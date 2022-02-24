import argparse
import os
import random
import time

import numpy as np
import pytorch_lightning as pl
import scipy.special
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from neptune.new.integrations.pytorch_lightning import NeptuneLogger
from sklearn.metrics import cohen_kappa_score

from filterbank_shape import FilterbankShape


def softXEnt(input, target):
    logprobs = torch.nn.functional.log_softmax(input, dim=1)
    return -(target * logprobs).sum() / input.shape[0]


class SeqSleepPL(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        # save input:
        self.save_hyperparameters(hparams)

        # settings:
        self.L = self.hparams.L  # sequence length
        self.nChan = self.hparams.nChannels
        self.learning_rate = self.hparams.learning_rate
        self.weight_decay = self.hparams.weight_decay
        self.total_steps = self.hparams.total_steps
        self.use_scheduler = self.hparams.use_scheduler

        self.nHidden = 64
        self.nFilter = 32
        self.attentionSize = 64
        self.dropOutProb = self.hparams.dropOutProb
        self.timeBins = 29

        # ---------------------------filterbank:--------------------------------
        filtershape = FilterbankShape()

        # triangular filterbank shape
        shape = torch.tensor(
            filtershape.lin_tri_filter_shape(
                nfilt=self.nFilter, nfft=256, samplerate=100, lowfreq=0, highfreq=50
            ),
            dtype=torch.float,
        )
        self.Wbl = nn.Parameter(shape, requires_grad=False)
        # filter weights:
        self.Weeg = nn.Parameter(torch.randn(self.nFilter, self.nChan))
        # ----------------------------------------------------------------------

        self.epochrnn = nn.GRU(
            self.nFilter, self.nHidden, 1, bidirectional=True, batch_first=True
        )

        # attention-layer:
        self.attweight_w = nn.Parameter(
            torch.randn(2 * self.nHidden, self.attentionSize)
        )
        self.attweight_b = nn.Parameter(torch.randn(self.attentionSize))
        self.attweight_u = nn.Parameter(torch.randn(self.attentionSize))

        # epoch sequence block:
        self.seqDropout = torch.nn.Dropout(self.dropOutProb, inplace=False)
        self.seqRnn = nn.GRU(
            self.nHidden * 2, self.nHidden, 1, bidirectional=True, batch_first=True
        )

        # output:
        self.fc = nn.Linear(2 * self.nHidden, 5)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--L", default=20, help="sequence length", type=int)
        parser.add_argument(
            "--learning_rate", default=1e-4, help="learning rate", type=float
        )
        parser.add_argument(
            "--weight_decay", default=1e-3, help="weight decay", type=float
        )
        parser.add_argument(
            "--dropOutProb", default=0.2, help="drop out probability", type=float
        )
        parser.add_argument(
            "--nChannels",
            default=1,
            help="number of channels to use as input",
            type=int,
        )
        parser.add_argument(
            "--use_scheduler",
            default="ReduceLROnPlateau",
            choices=["ReduceLROnPlateau", "CosineAnnealingLR", "OneCycleLR"],
            type=str,
        )
        return parser

    def forward(self, x):
        # assert (x.shape[0] / self.L).is_integer(), f'Shape: {x.shape}, expected (B, {self.timeBins}, 129, {self.nChan})'    #we need to pass a multiple of L epochs
        assert (
            x.shape[1] == self.timeBins
        ), f"Shape: {x.shape}, expected (B, {self.timeBins}, 129, {self.nChan})"
        assert (
            x.shape[2] == 129
        ), f"Shape: {x.shape}, expected (B, {self.timeBins}, 129, {self.nChan})"
        assert (
            x.shape[3] == self.nChan
        ), f"Shape: {x.shape}, expected (B, {self.timeBins}, 129, {self.nChan})"

        x = x.permute([0, 3, 1, 2])

        # import pdb; pdb.set_trace()
        # filtering:

        Wfb = torch.multiply(torch.sigmoid(self.Weeg[:, 0]), self.Wbl)
        x = torch.matmul(x, Wfb)  # filtering
        x = torch.reshape(x, [-1, self.timeBins, self.nFilter])
        # x=torch.einsum('btrc,ri,ic->btic',x,self.Wbl,torch.sigmoid(self.Weeg))
        # x=torch.reshape(x,(-1,self.timeBins,self.nFilter*self.nChan))

        # this uses Einstein notation. letting b:batch, t:time,r:frequency, i:filter, c:channel,
        # we are saying that the btic'th value in the output should be x_btrc*Wbl_ri*Weeb*ic, summing over
        # all frequencies. this is a generalization of
        # Wfb = torch.multiply(Weeg,Wbl)
        # x = torch.matmul(x, Wfb)
        # which works for n channels

        # biGRU:
        x, hn = self.epochrnn(x)
        x = self.seqDropout(x)

        # attention:
        v = torch.tanh(
            torch.matmul(torch.reshape(x, [-1, self.nHidden * 2]), self.attweight_w)
            + torch.reshape(self.attweight_b, [1, -1])
        )
        vu = torch.matmul(v, torch.reshape(self.attweight_u, [-1, 1]))
        exps = torch.reshape(torch.exp(vu), [-1, self.timeBins])
        alphas = exps / torch.reshape(torch.sum(exps, 1), [-1, 1])
        x = torch.sum(x * torch.reshape(alphas, [-1, self.timeBins, 1]), 1)

        # sequences of epochs:
        x = x.reshape(-1, self.L, self.nHidden * 2)
        x, hn = self.seqRnn(x)
        x = self.seqDropout(x)

        # return to epochs:
        x = x.reshape(-1, self.nHidden * 2)

        # out:
        x = self.fc(x)

        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        # Get LR scheduler
        if self.use_scheduler == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=50,
                threshold=0.0001,
                threshold_mode="rel",
                cooldown=0,
                min_lr=0,
                eps=1e-08,
                verbose=True,
            )

        elif self.use_scheduler == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.hparams.max_epochs, verbose=True
            )
        elif self.use_scheduler == "OneCycleLR":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                total_steps=self.total_steps,
                verbose=True,
            )
        elif self.use_scheduler is None:
            scheduler = None
        else:
            raise NotImplementedError(
                f"Learning rate scheduler {self.use_scheduler} is not implemented."
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/Kappa",
        }

    def epochMetrics(self, epochOutputs):
        epochPreds = np.concatenate(
            [out["pred_labels"].detach().clone().cpu() for out in epochOutputs]
        )
        trueLabels = np.concatenate(
            [out["labels"].detach().clone().cpu() for out in epochOutputs]
        )
        totLoss = (
            torch.sum(torch.stack([out["loss"] for out in epochOutputs]))
            / trueLabels.size
        )

        kappa = np.around(cohen_kappa_score(epochPreds, trueLabels), 4)
        acc = np.mean(epochPreds == trueLabels)
        return totLoss, kappa, acc

    def training_step(self, batch, batch_idx):
        xtemp, ytemp, idx = batch
        y_pred = self(xtemp)
        loss = (y_pred, ytemp)
        # loss = F.cross_entropy(y_pred, ytemp) # Softmax internally and categorical targets

        _, pred_labels = torch.max(y_pred, 1)  # pred_labels is 0-4
        _, true_labels = torch.max(ytemp, 1)  # true_labels is 0-4
        return {
            "loss": loss,
            "pred_labels": pred_labels,
            "labels": true_labels,
            "idx": idx,
        }

    def training_epoch_end(self, training_step_outputs):
        totLoss, kappa, acc = self.epochMetrics(training_step_outputs)

        self.log("train/Loss", totLoss)
        self.log("train/Kappa", kappa)

    def validation_step(self, batch, batch_idx):
        xtemp, ytemp, idx = batch
        y_pred = self(xtemp)
        loss = softXEnt(y_pred, ytemp)
        #        loss = F.cross_entropy(y_pred, ytemp)

        _, pred_labels = torch.max(y_pred, 1)  # pred_labels is 0-4
        _, true_labels = torch.max(ytemp, 1)  # true_labels is 0-4
        return {
            "loss": loss,
            "pred_labels": pred_labels,
            "labels": true_labels,
            "idx": idx,
        }

    def validation_epoch_end(self, validation_step_outputs):
        totLoss, kappa, acc = self.epochMetrics(validation_step_outputs)

        self.log("val/Loss", totLoss)
        self.log("val/Kappa", kappa)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, idx = batch
        return {"y_pred": torch.nn.functional.softmax(self(x), 1), "idx": idx}

    def custom_ensemble_test(self, testX, trainer):
        nTest = testX.shape[0]

        # We need to pad the end to make sure everything fits in N*L epochs
        missing = int(np.ceil(nTest / self.L) * self.L - nTest)

        paddedX = np.concatenate((testX, testX[0:missing]), axis=0)
        nPadded = paddedX.shape[0]

        probs = torch.zeros((self.L, nTest, 5))

        paddedX_tensor = torch.tensor(paddedX)
        with torch.no_grad():
            self.eval()

            for shift in range(0, self.L):
                rolledTest = torch.utils.data.TensorDataset(
                    torch.roll(paddedX_tensor, shift, 0), torch.arange(nPadded)
                )
                testLoader = torch.utils.data.DataLoader(
                    rolledTest,
                    batch_size=self.L * 5,
                    shuffle=False,
                    drop_last=False,
                    num_workers=8,
                )
                predictions = trainer.predict(self, testLoader, return_predictions=True)
                y_pred = torch.cat([pred.get("y_pred") for pred in predictions])
                idx = torch.cat([pred.get("idx") for pred in predictions])

                epochProbs = torch.zeros((nPadded, 5)).to(y_pred.device)
                epochProbs[idx, :] = y_pred

                probs[shift, :, :] = torch.roll(epochProbs[0:nTest, :], -shift, 0)

        # Collect predictions and remove padding
        probs = probs[:, 0:nTest, :]
        y_pred = torch.mean(torch.log(probs), dim=0)

        return {"ensemble_pred": y_pred, "rolled_probs": probs}


class EnsembleModel(pl.LightningModule):
    def __init__(self, models):
        super().__init__()

        self.models = models
        self.n_models = len(models)

        # Eval mode of models
        for model in self.models:
            model.eval()

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        model_predicts = [
            model.predict_step(batch, batch_idx, dataloader_idx)
            for model in self.models
        ]
        y_pred = torch.mean(
            torch.stack([model_pred["y_pred"] for model_pred in model_predicts]), dim=0
        )
        return {"y_pred": y_pred, "idx": model_predicts[0]["idx"]}

    def custom_ensemble_test(self, testX, trainer):
        model_predicts = [
            model.custom_ensemble_test(testX, trainer) for model in self.models
        ]
        ensemble_pred = torch.mean(
            torch.stack([model_pred["ensemble_pred"] for model_pred in model_predicts]),
            dim=0,
        )
        rolled_probs = torch.mean(
            torch.stack([model_pred["rolled_probs"] for model_pred in model_predicts]),
            dim=0,
        )
        return {"ensemble_pred": ensemble_pred, "rolled_probs": rolled_probs}
