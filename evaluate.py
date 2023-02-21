import argparse
import copy
import os
import pickle
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch

from loadMat4 import sleepEEGcontainer1
from SeqSleepNet import EnsembleModel, SeqSleepPL
from utils import find_data

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
parser.add_argument("--ensemble_pseudo_labels", action="store_true", default=False, help="use custom ensemble predictions as pseudo-labels")
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
parser.add_argument("--save_pred_file", default="predictions.p", type=str, help="file (and directory) for predictions to be saved")
parser.add_argument("--weight_space_ensemble", default=False, action="store_true")
parser.add_argument("--data_dir", default=None, type=str, help="path to data directory")

group = parser.add_mutually_exclusive_group()
group.add_argument("--all_pseudo_subjects", action="store_true", default=False, help="use all pseudo-subjects (default uses only pseudo-subjects in training fold)")
group.add_argument("--only_test_pseudo_subjects", action="store_true", default=False, help="use only test pseudo-subjects (default uses only pseudo-subjects in training fold)")
group.add_argument("--only_non_test_pseudo_subjects", action="store_true", default=False, help="use all other than test pseudo-subjects (default uses only pseudo-subjects in training fold)")
group.add_argument("--no_pseudo_subjects", action="store_true", default=False, help="use no pseudo-subjects - need to be combined with use_test_nights")
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
    f"SeqSleepNet" if args.experiment_name is None else args.experiment_name
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

# Get data container
loadedData = sleepEEGcontainer1.fromDirectory(matDir, deriv=args.derivation)
print("Data loaded")

# Normalize data
if args.normalize:
    loadedData.normalize()


######################################
# Perform Cross-Validation
######################################
# Loop over test sets and train
pl.seed_everything(args.seed)

predictions = {}
for test_idx in range(args.cv_range[0], args.cv_range[1] + 1):
    print(f"\nRunning CV{test_idx}")

    # Get train and validation folds
    rest = np.delete(np.arange(1, 21), test_idx - 1)
    assert len(rest) == 19
    shuffled_order = np.random.permutation(rest)
    train_idx = shuffled_order[0:15]
    val_idx = shuffled_order[15:19]
    print(f"Training folds: {train_idx}")
    print(f"Validation folds: {val_idx}")

    ######################################
    # Prepare model and training
    ######################################
    args.total_steps = 1
    # Loading "teacher" model
    weight_paths = [
        os.path.join(w_folder, f"CV{test_idx}.ckpt")
        for w_folder in args.cv_weights_folders
    ]
    teacher_models = [
        SeqSleepPL.load_from_checkpoint(Path(w_path), hparams=args)
        for w_path in weight_paths
    ]
    print(f"Weights loaded from {weight_paths}.")
    if args.weight_space_ensemble:
        # Get state dicts for all models
        state_dicts = [m.state_dict() for m in teacher_models]

        # Get mean of all weights across all baseline models
        ensemble_dict = copy.deepcopy(state_dicts[0])
        for key in state_dicts[0].keys():
            weights = torch.stack([sd[key] for sd in state_dicts])
            ensemble_dict[key] = weights.mean(dim=0)
            # print(f"{key:<35s}:  {ensemble_dict[key].shape}")

        # Load weights into ensemble model
        teacher = copy.deepcopy(teacher_models[0])
        teacher.load_state_dict(ensemble_dict)
        teacher.to("cuda" if torch.cuda.is_available() else "cpu")
        print("Weight space ensemble created")
    else:
        teacher_models = [
            teacher.to("cuda" if torch.cuda.is_available() else "cpu")
            for teacher in teacher_models
        ]
        teacher = EnsembleModel(teacher_models)
        print("Emsemble baseline created!")

    teacher.eval()

    # Prepare trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        benchmark=True,  # speeds up training if batch size is constant
        progress_bar_refresh_rate=1,
    )

    ######################################
    # Test model performance on test sample
    ######################################
    # Get test data
    testX, testy, testLabels = loadedData.returnBySubject(
        test_idx, night_idx=args.test_night_idx
    )

    # Pytorch datasets
    testDataset = torch.utils.data.TensorDataset(
        torch.tensor(testX), torch.arange(testLabels.size)
    )

    testSampler = torch.utils.data.DataLoader(
        testDataset,
        batch_size=args.L * 5,
        shuffle=False,
        drop_last=True,
        num_workers=4,
        pin_memory=False,
    )
    # Get Predictions
    model_predicts = [
        model.custom_ensemble_test(testX, trainer) for model in teacher_models
    ]

    # Add predictions to dict and save temp.
    predictions.update({f"CV{test_idx}": model_predicts})

    # create a binary pickle file
    prediction_file = open(
        os.path.join(args.cv_weights_folders[0], args.save_pred_file), "wb"
    )
    pickle.dump(
        predictions, prediction_file
    )  # write the python object (dict) to pickle file
    prediction_file.close()  # close file
    print(
        f"Saved predictions to file {os.path.join(args.cv_weights_folders[0], args.save_pred_file)}"
    )
