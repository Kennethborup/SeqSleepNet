import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score


def log_histograms(logger, probs, labels, CV):
    plot_data = pd.DataFrame(np.array(probs), columns=range(5))
    plot_data["true"] = (labels.T - 1).astype(int)

    for name, df in plot_data.groupby("true"):
        fig = plt.figure(figsize=(10, 8))
        for col in range(5):
            df[col].hist(bins=25, label=str(col), alpha=0.5)
        plt.xlim(0, 1)
        plt.title(f"True class: {name}")
        plt.legend()
        logger.experiment[f"CV{CV}/prob_histogram_{name}"].upload(fig)
        plt.clf()
        plt.close(fig)


def log_pseudo_histograms(logger, probs, labels, CV):
    plot_data = pd.DataFrame(np.array(probs), columns=range(5))
    plot_data["true"] = labels.T.astype(int)

    for name, df in plot_data.groupby("true"):
        fig = plt.figure(figsize=(10, 8))
        for col in range(5):
            df[col].hist(bins=25, label=str(col), alpha=0.5)
        plt.xlim(0, 1)
        plt.title(f"Predicted class: {name}")
        plt.legend()
        logger.experiment[f"CV{CV}/pseudo_prob_histogram_{name}"].upload(fig)
        plt.clf()
        plt.close(fig)
