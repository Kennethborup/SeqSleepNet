import argparse
import copy
import os
import socket
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from neptune.new.integrations.pytorch_lightning import NeptuneLogger
from neptune.new.types import File
from pytorch_lightning.callbacks import LearningRateMonitor
from sklearn.metrics import cohen_kappa_score

from loadMat4 import custom_collate_fn, sleepEEGcontainer1, trainingEEGDataset_1
from SeqSleepNet import EnsembleModel, SeqSleepPL
from utils import log_histograms, log_pseudo_histograms

######################################
# Parse arguments
######################################
parser = argparse.ArgumentParser()

# Add PROGRAM level args
# fmt: off
parser.add_argument("--seed", default=None, help="random seed", type=int)
parser.add_argument("--normalize", action="store_true", default=False, help="normalize input")
parser.add_argument("--early_stopping_delay", default=None, help="early stopping delay", type=int)
parser.add_argument("--experiment_name", default=None, help="name of experiment", type=str)
parser.add_argument("--swa", action="store_true", default=False, help="use stochastic weight averaging")
parser.add_argument("--cv_range", nargs="*", help="which CV to train", type=int, default=None)
parser.add_argument("--cv_weights", nargs="*", help="weights for trained CV models", type=str, default=None)
parser.add_argument("--cv_weights_folders", nargs="*", help="folders with weights for trained CV models (CV{i}.ckpt)", type=str, default=None)
parser.add_argument("--run_id", default=None, help="name of log to use")  # Alternatively export NEPTUNE_CUSTOM_RUN_ID="<custom_id>" in bash
parser.add_argument("--continue_training", action="store_true", default=False, help="continue training of teacher weights")
parser.add_argument("--ensemble_pseudo_labels", action="store_true", default=False, help="use custom ensemble predictions as pseudo-labels (avg. based)")
parser.add_argument("--confidence_threshold", default=0.0, help="minimum threshold for samples to be used as pseudolabels", type=float)
parser.add_argument("--hard_pseudo_labels", action="store_true", default=False, help="use one-hot encoded pseudo-labels (default uses soft pseudo-labels)")
parser.add_argument("--temperature", default=1, help="temperature for soft pseudo-labels", type=float)
parser.add_argument("--tags", nargs="*", help="tags to add to neptune logger", type=str)
parser.add_argument("--n_pseudo_nights", default=None, type=int, help="number of pseudo-nights to use during training (default: None, corresponds to all)")
parser.add_argument("--seed_nights", default=None, type=int, help="random seed when sampling n nights")
parser.add_argument("--use_test_nights", default=False, action="store_true", help="use 4 test nights as part of pseudo data")
parser.add_argument("--test_night_idx", default=None, type=int, help="use the single test night with idx = test_night_idx as unlabeled test night")
parser.add_argument("--derivation", default="eeg_lr", type=str, choices=["eeg_lr", "ear_eog1", "eeg_l"], help="name of derivation to use for training (only a single derivation can be used here)")
parser.add_argument("--skip_non_pseudo", default=False, action="store_true", help="skip CV-steps for subjects without unlabeled data")
parser.add_argument("--ensemble_test_type", default="avg", type=str, choices=["prod", "avg"], help="type of ensemble to use for student")
parser.add_argument("--load_pseudo_labels", default=None, type=str, help="dir of pseudo_labels to be loaded from disc (automatically appends CV{test_idx}.csv)")
parser.add_argument("--save_pseudo_labels", default=None, type=str, help="dir of pseudo_labels to be saved to disc (automatically appends CV{test_idx}.csv)")
parser.add_argument("--soft_training_data", default=False, action="store_true", help="use pseudo-labels on training data as well")
parser.add_argument("--no_training_data", default=False, action="store_true", help="use only unlabeled data as distillation data")
parser.add_argument("--data_dir", default=None, type=str, help="path to data directory")
parser.add_argument("--unlabeled_data_dir", default=None, type=str, help="path to unlabeled data directory")

group = parser.add_mutually_exclusive_group()
group.add_argument( "--all_pseudo_subjects", action="store_true", default=False, help="use all pseudo-subjects (default uses only pseudo-subjects in training fold)")
group.add_argument( "--only_test_pseudo_subjects", action="store_true", default=False, help="use only test pseudo-subjects (default uses only pseudo-subjects in training fold)")
group.add_argument( "--only_non_test_pseudo_subjects", action="store_true", default=False, help="use all other than test pseudo-subjects (default uses only pseudo-subjects in training fold)")
group.add_argument( "--no_pseudo_subjects", action="store_true", default=False, help="use no pseudo-subjects - need to be combined with use_test_nights")
# fmt: on

# Add model specific args
parser = SeqSleepPL.add_model_specific_args(parser)

# Add all the available trainer options to argparse. ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
parser = pl.Trainer.add_argparse_args(parser)

# Parse args
args = parser.parse_args()

# Customize args
if args.cv_range is None:
    args.cv_range = (1, 20)
elif len(args.cv_range) == 1:
    args.cv_range = (args.cv_range[0], args.cv_range[0])
elif len(args.cv_range) > 2:
    raise NotImplementedError(
        f"cv_range should be either the largest cv (one value) or the range (two values), not {args.cv_range}"
    )
if args.cv_weights is not None:
    assert ((args.cv_range[1] + 1) - args.cv_range[0]) == len(args.cv_weights)
args.seed = np.random.choice(range(0, 100)) if args.seed is None else args.seed
args.experiment_name = (
    f"SeqSleepNet"
    if args.experiment_name is None
    else args.experiment_name
    if args.experiment_name is None
    else args.experiment_name
)
args.tags = [] if args.tags is None else args.tags

# Print arguments
params = vars(args)
print("Experiment settings:")
for key, val in params.items():
    print(f'{" "*4}{key}: {val}')


######################################
# Import data
######################################
# Find data
matDir = (
    args.data_dir
    if os.path.basename(args.data_dir) == "mat"
    else os.path.join(args.data_dir, "mat")
)
loadedData = sleepEEGcontainer1.fromDirectory(matDir, deriv=args.derivation)
print("Data loaded")

# Unlabeled data
unlabeledDir = (
    args.unlabeled_data_dir
    if os.path.basename(args.unlabeled_data_dir) == "mat"
    else os.path.join(args.unlabeled_data_dir, "mat")
)
unlabeledData = sleepEEGcontainer1.fromDirectory(
    unlabeledDir, deriv="ear_ego1" if args.derivation == "ear_eog1" else args.derivation
)
print(f"Unlabeled data exist for subjects: {np.unique(unlabeledData.subjectName)}")
pseudo_subjects = np.unique(unlabeledData.subjectName)

# Normalize data
if args.normalize:
    loadedData.normalize()
    unlabeledData.normalize(  # Use mean and std from labeled data
        mean=loadedData.normalize_mean, std=loadedData.normalize_std
    )


######################################
# Perform Cross-Validation
######################################
# Loop over test sets and train
pl.seed_everything(args.seed)
allKappas = np.zeros((20, 2))
allKappas[:, 0] = np.arange(1, 21)

run_id = args.run_id
for test_idx in range(args.cv_range[0], args.cv_range[1] + 1):
    pl.seed_everything(args.seed)
    # Skip CV if no pseudo_labels and settings require it
    if args.skip_non_pseudo:
        if test_idx not in pseudo_subjects:
            print(f"\nSkipping CV{test_idx}")
            allKappas[test_idx - 1, 1] = np.nan
            continue
    else:
        print(f"\nRunning CV{test_idx}")

    # Initialize logger and save parameters
    neptune_logger = NeptuneLogger(
        project="your/project",  # FIXME: Add your project on neptune here
        name=args.experiment_name,
        base_namespace=f"CV{test_idx}",
        run=run_id,
        tags=["selftrain"] + args.tags,
        close_after_fit=False,
        monitoring_namespace=f"monitoring-cv{test_idx}",
    )
    neptune_logger.experiment["parameters"] = params

    # Get train and validation folds
    rest = np.delete(np.arange(1, 21), test_idx - 1)
    assert len(rest) == 19
    shuffled_order = np.random.permutation(rest)
    train_idx = shuffled_order[0:15]
    val_idx = shuffled_order[15:19]
    print(f"Training folds: {train_idx}")
    print(f"Validation folds: {val_idx}")

    # Get pseudo-folds
    if args.only_test_pseudo_subjects:
        unlabeled_train_idx = unlabeledData.filterSubjects(np.array([test_idx]))
    elif args.all_pseudo_subjects:
        unlabeled_train_idx = unlabeledData.filterSubjects(np.arange(1, 21))
    elif args.only_non_test_pseudo_subjects:
        unlabeled_train_idx = unlabeledData.filterSubjects(
            np.concatenate([train_idx, val_idx])
        )
        # If more than 9 subjects (= 108 nights) randomly choose only 9 subjects
        print(f"Unlabeled subjects allowed for run: {np.unique(unlabeled_train_idx)}")
        if len(unlabeled_train_idx) > 9:
            unlabeled_train_idx = np.random.choice(
                unlabeled_train_idx, 9, replace=False
            )
        print(f"Unlabeled subjects used for run: {np.unique(unlabeled_train_idx)}")
    elif args.no_pseudo_subjects:
        unlabeled_train_idx = np.array([])
    else:  # train subjects only
        unlabeled_train_idx = unlabeledData.filterSubjects(train_idx)
    print(f"Using pseudo data for subjects: {np.unique(unlabeled_train_idx)}")

    ######################################
    # Load correct data
    ######################################
    # Load data
    trainX, trainy, trainLabels = loadedData.returnBySubject(train_idx)
    valX, valy, valLabels = loadedData.returnBySubject(val_idx)

    trainLabels_tensor = torch.tensor(trainLabels - 1).type(torch.long)
    valLabels_tensor = torch.tensor(valLabels - 1).type(torch.long)

    # Pytorch datasets
    trainDataset = torch.utils.data.TensorDataset(
        torch.tensor(trainX), torch.tensor(trainy), torch.arange(trainLabels.size)
    )
    valDataset = torch.utils.data.TensorDataset(
        torch.tensor(valX), torch.tensor(valy), torch.arange(valLabels.size)
    )

    # DataLoaders
    trainSampler = torch.utils.data.DataLoader(
        trainingEEGDataset_1(trainDataset, args.L),
        batch_size=5,
        shuffle=True,
        drop_last=True,
        collate_fn=custom_collate_fn,
        num_workers=8 if not args.auto_lr_find else 0,
        pin_memory=True if args.gpus is not None else False,
    )

    valSampler = torch.utils.data.DataLoader(
        valDataset,
        batch_size=args.L * 5,
        shuffle=False,
        drop_last=True,
        num_workers=8 if not args.auto_lr_find else 0,
        pin_memory=True if args.gpus is not None else False,
    )

    ########### Get pseudo data ###########
    if not args.no_pseudo_subjects:
        print(f"Using pseudo data for subjects: {np.unique(unlabeled_train_idx)}")
        unlabeledX, _, _ = unlabeledData.returnBySubject(
            unlabeled_train_idx, nights=args.n_pseudo_nights, seed=args.seed_nights
        )
        unlabeledX = torch.tensor(unlabeledX)
        unlabeledIdx = torch.arange(unlabeledX.shape[0])
    else:
        unlabeledX = torch.tensor([])
        unlabeledIdx = torch.arange(0)

    # Get test_nights from labeled data
    if args.use_test_nights:
        testX, _, _ = loadedData.returnBySubject(
            test_idx, night_idx=args.test_night_idx
        )
        testX = torch.tensor(testX)
        testIdx = torch.arange(testX.shape[0])
    else:
        testX = torch.tensor([])
        testIdx = torch.arange(0)

    # Get 1A data for soft labeling
    if args.soft_training_data:
        soft_trainX = torch.tensor(trainX)
        soft_trainIdx = torch.arange(soft_trainX.shape[0])
    else:
        soft_trainX = torch.tensor([])
        soft_trainIdx = torch.arange(0)

    # Combine data
    pseudoX = torch.cat([unlabeledX, testX, soft_trainX], axis=0)
    pseudoIdx = torch.cat([unlabeledIdx, testIdx, soft_trainIdx])
    print(f"Amount of samples with soft labels: {len(pseudoIdx)}")

    # Construct dataset
    if len(pseudoIdx) != 0:
        pseudoDataset = torch.utils.data.TensorDataset(pseudoX, pseudoIdx)

        pseudoSampler = torch.utils.data.DataLoader(
            pseudoDataset,
            batch_size=args.L * 5,
            shuffle=False,
            drop_last=True,
            num_workers=8 if not args.auto_lr_find else 0,
            pin_memory=True if args.gpus is not None else False,
        )

        # Rescale amount of epochs/batches to be consistent:
        args.limit_train_batches = float(
            len(trainSampler) / (len(trainSampler) + len(pseudoSampler))
        )
    else:
        # Rescale amount of epochs/batches to be consistent:
        args.limit_train_batches = 1.0

    ######################################
    # Prepare model and training
    ######################################
    args.total_steps = len(trainSampler) * args.max_epochs

    # Loading "teacher" model
    if args.cv_weights_folders is not None:
        weight_paths = [
            os.path.join(w_folder, f"CV{test_idx}.ckpt")
            for w_folder in args.cv_weights_folders
        ]
    if args.cv_weights is not None:
        weight_paths = [args.cv_weights[(test_idx - 1) - args.cv_range[0]]]

    if len(weight_paths) == 1:
        teacher = SeqSleepPL.load_from_checkpoint(Path(weight_paths[0]), hparams=args)
        print(f"Weights loaded from {weight_paths}.")
    elif len(weight_paths) >= 2:
        teacher_models = [
            SeqSleepPL.load_from_checkpoint(Path(w_path), hparams=args)
            for w_path in weight_paths
        ]
        print(f"Weights loaded from {weight_paths}.")
        teacher_models = [
            teacher.to("cuda" if torch.cuda.is_available() else "cpu")
            for teacher in teacher_models
        ]
        teacher = EnsembleModel(teacher_models)
        print("Emsemble baseline created!")

    # Initializing "student" model
    if args.continue_training:
        assert (
            len(weight_paths) == 1
        ), "Can not continue training when baseline/teacher is an ensemble..."
        print("Continue training of provided weights")
        model = copy.deepcopy(teacher)
    else:
        print("Initializing new model")
        model = SeqSleepPL(args)

    print(f"Model on device: {model.device}")

    # Collect callbacks
    callback_list = []

    # Model checkpoints
    checkpoint_callback = pl.callbacks.ModelCheckpoint(  # Save last checkpoint
        filename=f"CV{test_idx}", save_top_k=None, monitor=None
    )
    callback_list.append(checkpoint_callback)

    # Early stopping
    if args.early_stopping_delay is not None:
        early_stopping = pl.callbacks.EarlyStopping(
            monitor="val/Kappa",
            min_delta=0.00,
            patience=args.early_stopping_delay,
            verbose=True,
            mode="max",
        )
        print("Using early stopping callback")
        callback_list.append(early_stopping)

    if args.swa:
        swa_callback = pl.callbacks.StochasticWeightAveraging(
            swa_epoch_start=0.8,
            annealing_epochs=10,
        )
        print(f"Using SWA callback, and ignores all schedulers: {args.use_scheduler}.")
        print("Using SWA callback.")
        callback_list.append(swa_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callback_list.append(lr_monitor)

    # Prepare trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=callback_list,
        logger=neptune_logger,
        benchmark=True,  # speeds up training if batch size is constant
        progress_bar_refresh_rate=0,
        weights_save_path="checkpoints/",  # needed since, trainer.logger.save_dir fails otherwise
    )

    # If no pseudo-data is available continue, else get pseudo-predictions
    if len(pseudoIdx) == 0:
        print(
            f"No pseudo data available for subject {test_idx}, training on normal training data."
        )
        combinedSampler = trainSampler

    else:
        if args.ensemble_pseudo_labels:  # Use ensemble test for predictions (slow)
            baseline_predicts = [
                baseline.custom_ensemble_test(pseudoX, trainer)
                for baseline in teacher_models
            ]
            teacher_ensemble_preds = torch.stack(
                [
                    torch.mean(baseline_pred["rolled_probs"], dim=0)
                    for baseline_pred in baseline_predicts
                ]
            ).mean(
                dim=0
            )  # using average prob. vectors
            y_pred = teacher_ensemble_preds.clone().detach()
        else:  # use only one prediction per epoch (fast)
            predictions = trainer.predict(
                teacher, combinedSampler, return_predictions=True
            )
            y_pred = torch.cat([pred.get("y_pred") for pred in predictions])

        # Log predicted pseudo-probabilities
        conf, pseudo_labels = torch.max(y_pred, 1)
        log_pseudo_histograms(
            neptune_logger, y_pred.cpu(), np.array(pseudo_labels.cpu()), test_idx
        )

        # Only use pseudo-labels with largest confidence above threshold
        if args.confidence_threshold > 0:
            confident_preds = conf > args.confidence_threshold
            y_pred = y_pred[confident_preds]
            print(
                f"{sum(confident_preds)} confident pseudo samples (at {args.confidence_threshold}) of {len(confident_preds)} pseudo samples"
            )
            print(
                f"Pseudo samples are distributed as {torch.bincount(pseudo_labels[confident_preds])}"
            )

        # Adapt pseudo-labels to soft or hard
        if args.hard_pseudo_labels:
            y_pred_hard = torch.zeros_like(y_pred)
            y_pred_hard[torch.arange(y_pred.shape[0]), torch.argmax(y_pred, 1)] = 1
            y_pred = y_pred_hard
        elif args.temperature != 1:
            y_pred = torch.nn.functional.softmax(
                torch.log(y_pred) / args.temperature, 1
            )

        if args.no_training_data or args.soft_training_data:
            print(
                f"Note, since no_training_data is {args.no_training_data} and soft_training_data is {args.soft_training_data}, no original 1A training data is included."
            )
            combinedDataset = torch.utils.data.TensorDataset(
                pseudoX[: y_pred.shape[0]], y_pred.cpu(), pseudoIdx[: y_pred.shape[0]]
            )
        else:
            combinedDataset = torch.utils.data.TensorDataset(
                torch.cat([torch.tensor(trainX), pseudoX[: y_pred.shape[0]]]),
                torch.cat([torch.tensor(trainy), y_pred.cpu()]),
                torch.cat(
                    [torch.arange(trainLabels.size), pseudoIdx[: y_pred.shape[0]]]
                ),
            )

        combinedSampler = torch.utils.data.DataLoader(
            trainingEEGDataset_1(combinedDataset, args.L),
            batch_size=5,
            shuffle=True,
            drop_last=True,
            collate_fn=custom_collate_fn,
            num_workers=8 if not args.auto_lr_find else 0,
            pin_memory=True if args.gpus is not None else False,
        )
        print(
            f"New dataset consist of {trainX.shape[0]*(not args.no_training_data)} training samples and {y_pred.shape[0]} pseudosamples."
        )

    # Perform auto_lr_find procedure and exit script afterwards. Can be used to find a *good* lr (highly dependent on weight_decay)
    if args.auto_lr_find is not False:
        print("Finding learning rate automatically.")
        trainer.logger = []
        lr_finder = trainer.tuner.lr_find(
            model, train_dataloader=trainSampler, val_dataloaders=valSampler
        )

        args.learning_rate = lr_finder.suggestion()
        params.update({"learning_rate": args.learning_rate})

        trainer.logger = neptune_logger
        neptune_logger.experiment["parameters"] = params
        fig = lr_finder.plot(suggest=True)
        neptune_logger.experiment["lr_finder"].upload(fig)
        print(f"Auto-found and updated learning rate: {model.learning_rate}")
        print("Exiting script. Find learning rate in logs.")
        sys.exit()

    ######################################
    # Train model
    ######################################
    trainer.fit(model, combinedSampler, valSampler)

    ######################################
    # Test model performance on test sample
    ######################################
    # Get test data
    testX, testy, testLabels = loadedData.returnBySubject(
        test_idx, night_idx=args.test_night_idx
    )

    # Perform ensemble testing and calculate Cohen's Kappa
    ensembleTesting = model.custom_ensemble_test(testX, trainer)
    if (
        args.ensemble_test_type == "avg"
    ):  # Note this changes the ensemble_pred evaluation from product based to average based
        ensembleTesting["ensemble_pred"] = torch.mean(
            ensembleTesting["rolled_probs"], dim=0
        )

    _, pred_class = torch.max(ensembleTesting["ensemble_pred"], 1)  # pred_class is 0-4
    kappa = cohen_kappa_score(torch.unsqueeze(pred_class + 1, 1), testLabels.T)

    # Calculate Cohen's Kappa if did not do ensemple
    rolledKappas = np.zeros(args.L)
    for iRoll in range(args.L):
        _, pred_class = torch.max(
            ensembleTesting["rolled_probs"][iRoll, :, :], 1
        )  # pred_class is 0-4
        rolledKappas[iRoll] = cohen_kappa_score(
            torch.unsqueeze(pred_class + 1, 1), testLabels.T
        )

    # Log histograms
    log_histograms(
        neptune_logger, ensembleTesting["ensemble_pred"], testLabels, test_idx
    )

    # Log test metrics
    print("rolledKappas:", rolledKappas)
    print("meanRolledKappa:", np.mean(rolledKappas))
    print("Consensus:", test_idx, kappa)
    neptune_logger.experiment["test/subjectKappa"].log(kappa)
    neptune_logger.experiment["test/meanRolledKappa"].log(np.mean(rolledKappas))

    # Save kappa for total log later
    allKappas[test_idx - 1, 1] = kappa

    # Continue to log in this run on neptune, but with newly initialized logger
    if os.getenv("NEPTUNE_CUSTOM_RUN_ID", False):
        run_id = None
    else:
        run_id = neptune_logger.experiment.get_run_url().split("/")[-1]

    ######################################
    # Logging allKappas file to CV-folder on neptune
    ######################################
    kappa_id = run_id if run_id is not None else os.getenv("NEPTUNE_CUSTOM_RUN_ID")
    kappa_file = (
        f"kappas/{str(kappa_id)}/{args.experiment_name}_allKappas_CV{test_idx}.csv"
    )
    try:
        prev_kappas = np.loadtxt(kappa_file, delimiter=",")
        prev_kappas[test_idx - 1, 1] = kappa
        np.savetxt(kappa_file, prev_kappas, delimiter=",")
    except:
        os.makedirs(os.path.dirname(kappa_file), exist_ok=True)
        np.savetxt(kappa_file, allKappas, delimiter=",")
    neptune_logger.experiment[f"CV{test_idx}/allKappas"].upload(
        File(kappa_file)
    )  # Log kappa_file to CV-folder on Neptune

    # Stop this CV-logger
    neptune_logger.experiment.stop()

######################################
# Perform final logging to 'total' folder
######################################
# Save all kappas to disk
kappa_id = run_id if run_id is not None else os.getenv("NEPTUNE_CUSTOM_RUN_ID")
kappa_file = f"kappas/{str(kappa_id)}/{args.experiment_name}_allKappas.csv"
try:
    prev_kappas = np.loadtxt(kappa_file, delimiter=",")
    allKappas[:, 1] += prev_kappas[:, 1]
except:
    os.makedirs(os.path.dirname(kappa_file), exist_ok=True)
np.savetxt(kappa_file, allKappas, delimiter=",")

# Initialize logger to log total metrics
neptune_logger = NeptuneLogger(
    project="kennethborup/ear-eeg-distill",
    name=args.experiment_name,
    tags=["selftrain"] + args.tags,
    run=run_id,
    proxies={"https": "http://proxyserv:3128"}
    if "genomedk" in socket.getfqdn().split(".")
    else None,  # needed on genomedk
    close_after_fit=False,
)

# Log metrics
print("allKappas", allKappas[:, 1])
print("meanKappas", np.mean(allKappas[:, 1]))
neptune_logger.experiment["total/allKappas"].upload(File(kappa_file))
neptune_logger.experiment["total/meanKappa"] = np.mean(allKappas[:, 1])

neptune_logger.experiment.stop()
