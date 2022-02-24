# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 20:46:50 2021

@author: mitur
"""

import os
import pickle
import random
from pathlib import Path

import h5py
import numpy as np
import torch


class sleepEEGcontainer1:
    def __init__(self, inputDict):
        self.Xlist = inputDict["Xlist"]
        self.ylist = inputDict["ylist"]
        self.labelList = inputDict["labelList"]
        self.subjectName = inputDict["subjectName"]
        self.subjectNight = inputDict["subjectNight"]

        self.n = len(self.Xlist)

        # self.normalize()

    def __repr__(self):
        return "Dataset with " + str(self.n) + " recordings"

    def normalize(self, mean=None, std=None):
        self.n = len(self.Xlist)
        assert self.n == len(self.ylist)
        assert self.n == len(self.labelList)

        # normalize data (for each frequency):
        if mean is None:
            allMeans = np.array(
                [np.mean(x, axis=(1, 2)) for x in self.Xlist if len(x) > 10]
            )  # len(x) > 10 er for at sikre at det ikke er tomme epoker
            totMean = np.mean(allMeans, 0).reshape((-1, 1, 1))
        else:
            totMean = mean
        self.normalize_mean = totMean
        Xlist = [x - totMean for x in self.Xlist]

        if std is None:
            allStds = np.array(
                [np.std(x, axis=(1, 2)) for x in self.Xlist if len(x) > 10]
            )
            totStd = np.mean(allStds, 0).reshape((-1, 1, 1))
        else:
            totStd = std
        self.normalize_std = totStd
        self.Xlist = [x / totStd for x in Xlist]

    @classmethod
    def fromDirectory(cls, matDir, deriv):
        loadedDict = loadMatData(matDir, deriv)
        return cls(loadedDict)

    def returnRecords(self, idxs, nights=None, seed=None, night_idx=None):
        idxs = idxs[0]  # input is a 1d tuple with one entry
        assert np.array(idxs).size

        # ignore empty idxs (i.e. fewer than 1000 entries):
        idxs = [idx for idx in idxs if self.Xlist[idx].size > 1000]
        if nights is not None:
            if seed is not None:
                rng = random.Random(seed)
                for _ in range(nights):
                    rng.shuffle(idxs)
            idxs = idxs[0:nights]
            print(f"Using {nights} pseudo-nights, with indices {idxs}")
        elif night_idx is not None:
            print(f"Nights available with indices {idxs}")
            idxs = idxs[night_idx]
            print(f"Using night {night_idx}, with index {idxs}")
        assert np.array(idxs).size

        # Get correct samples
        idxs = [idxs] if not isinstance(idxs, list) else idxs
        Xout = np.concatenate([self.Xlist[i] for i in idxs], axis=2)
        yout = np.concatenate([self.ylist[i] for i in idxs], axis=1)
        label_out = np.concatenate([self.labelList[i] for i in idxs], axis=1)

        # we want batch x 29 x 129 x 1: (måske b x 1 x 29 x 129)
        Xout = Xout.swapaxes(0, 2)
        Xout = np.expand_dims(Xout, 3)

        # we want batch x 5:
        yout = yout.T

        return Xout, yout, label_out

    def returnSubjectByNights(self, subject_idx):
        subject_idx = np.array([subject_idx])
        assert np.array(subject_idx).size
        idxs = self.filterSubjects(subject_idx)
        idxs = np.where(np.in1d(self.subjectName, idxs))[0]

        # ignore empty idxs (i.e. fewer than 1000 entries):
        idxs = [idx for idx in idxs if self.Xlist[idx].size > 1000]
        assert np.array(idxs).size
        # Get correct samples
        Xout = [self.Xlist[i] for i in idxs]
        yout = [self.ylist[i] for i in idxs]
        label_out = [self.labelList[i] for i in idxs]

        # we want batch x 29 x 129 x 1: (måske b x 1 x 29 x 129)
        Xout = [np.expand_dims(X.swapaxes(0, 2), 3) for X in Xout]

        # we want batch x 5:
        yout = [y.T for y in yout]

        return Xout, yout, label_out

    def returnBySubject(self, iSs, nights=None, seed=None, night_idx=None):
        iSs = np.array([iSs]) if not isinstance(iSs, np.ndarray) else iSs
        if len(iSs) == 0:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])

        # did the user ask for non-existent subjects:
        if not all(np.in1d(iSs, self.subjectName)):
            print("Error: requested subject not in data set")
            raise SystemExit(0)

        # find recordings for all subjects:
        recs = np.where(np.in1d(self.subjectName, iSs))

        Xout, yout, label_out = self.returnRecords(recs, nights, seed, night_idx)
        return Xout, yout, label_out

    def filterSubjects(self, subject_idxs):
        """
        Filters input to only contain subjects present in this dataset
        """
        return subject_idxs[np.in1d(subject_idxs, self.subjectName)]


def loadMatData(matDir, deriv):  # afladning/kanal skal bruge eeg_lr
    pickleName = os.path.join(matDir, deriv + "_" + "pickled.p")
    print("Pickle-name:", pickleName)
    if os.path.exists(pickleName):
        print("Loading pickled data")
        temp = pickle.load(open(pickleName, "rb"))
        Xlist = temp["Xlist"]
        ylist = temp["ylist"]
        labelList = temp["labelList"]
        subjectName = temp["subjectName"]
        subjectNight = temp["subjectNight"]

    else:
        Xlist, ylist, labelList = [], [], []
        subjectName, subjectNight = [], []
        counter = 0

        # Get subject-dirs
        p = Path(matDir)
        subjectDirs = [
            x for x in p.iterdir() if x.is_dir() and x != ".ipynb_checkpoints"
        ]

        for iS in range(len(subjectDirs)):
            try:
                subjectName_temp = int(str(subjectDirs[iS])[-2:])
            except:
                subjectName_temp = int(iS)

            # Get night-dirs
            p = subjectDirs[iS]
            nightDirs = [
                x for x in p.iterdir() if x.is_dir() and x != ".ipynb_checkpoints"
            ]
            for iN in range(len(nightDirs)):
                filename = os.path.join(nightDirs[iN], deriv + ".mat")
                temp = h5py.File(filename, "r")
                subjectName.append(subjectName_temp)
                subjectNight.append(iN)
                Xlist.append(np.array(temp["X"]))
                try:
                    ylist.append(np.array(temp["y"]))
                    labelList.append(np.array(temp["label"]))
                except:
                    # if there are no labels:
                    ylist.append(np.empty((0, 0)))
                    labelList.append(np.empty((0, 0)))

                counter += 1
                print(f"Count: {counter}", end="\r")

        print("Pickling data")
        pickle.dump(
            {
                "Xlist": Xlist,
                "ylist": ylist,
                "labelList": labelList,
                "subjectName": subjectName,
                "subjectNight": subjectNight,
            },
            open(pickleName, "wb"),
        )

    return {
        "Xlist": Xlist,
        "ylist": ylist,
        "labelList": labelList,
        "subjectName": subjectName,
        "subjectNight": subjectNight,
    }


if __name__ == "__main__":
    matDir = "D:\\OneDrive - Aarhus Universitet\\python\\data\\10x12_nights\\mat"
    deriv = "eeg_lr"
    loadedData = sleepEEGcontainer1.fromDirectory(matDir, deriv)
    x, y, labels = loadedData.returnBySubject([1])


class trainingEEGDataset_1(torch.utils.data.Dataset):
    # a wrapper for torch datasets, to make it possible to shuffle sequences
    def __init__(self, inputDataset: torch.utils.data.Dataset = None, L: int = None):

        self.dataSet = inputDataset
        self.L = L

        # bookkeeping idx's:
        self.seqIndices = None
        self.getCounter = 0
        self.reset()

    def reset(self):
        # reset bookkeeping idx's
        start = np.random.randint(0, self.L)
        seqRange = range(start, len(self.dataSet), self.L)
        seqRange = range(seqRange[0], seqRange[len(seqRange) - 1])
        self.seqIndices = np.reshape(seqRange, (-1, self.L))
        self.getCounter = 0

    def __len__(self):
        return int(np.floor(len(self.dataSet) / self.L))
        # return self.seqIndices.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if type(idx) in (tuple, list):
            print(len(idx))
            idx = idx[0]

        try:
            self.getCounter += len(idx)
        except:
            # if idx is a scalar, the other one fails
            self.getCounter += np.array(idx).size

        # because __len__ fluctuates, we need to make sure we don't try to access non-existing data:
        idx = idx % (self.seqIndices.shape[0])

        try:
            sample = self.dataSet[np.reshape(self.seqIndices[idx, :], (-1,))]
        except:
            print("Custom dataloader failed")
            print("maxIdx", np.max(idx))
            print("len:", self.__len__())
            print("self.dataSet", self.dataSet.shape)
            print("seqIndices.max", np.max(self.seqIndices))
            raise SystemExit(0)

        # if all idxs have been passed:
        if self.getCounter >= (self.seqIndices.shape[0] - 1):
            self.reset()

        return sample


def custom_collate_fn(batch):
    x = torch.cat([item[0] for item in batch])
    y = torch.cat([item[1] for item in batch])
    i = torch.cat([item[2] for item in batch])
    return x, y, i
