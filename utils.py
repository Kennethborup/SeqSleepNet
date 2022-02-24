import os

import matplotlib.pyplot as plt
import neptune.new as neptune
import numpy as np
import pandas as pd


def find_data(dirs=None):
    dirs = (
        [
            os.path.join(
                "/",
                "Users",
                "kennethborup",
                "OneDrive - Aarhus Universitet",
                "ear_eeg",
                "20x4_nights",
                "mat",
            ),  # Local
            "data/20x4_nights/mat",  # GenomeDK
            os.path.join(
                "/", "com", "ecent", "NOBACKUP", "sleepData", "20x4_nights", "mat"
            ),  # PRIME
        ]
        if dirs is None
        else dirs
    )

    for temp_dir in dirs:
        if os.path.exists(temp_dir):
            data_dir = temp_dir
            print(f"Directory of data: {data_dir}")
            return data_dir


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


def majority_vote_ensemble(ensemble_preds):
    disagree = []
    preds = []
    for pred in ensemble_preds.T:
        classes, counts = pred.unique(return_counts=True)
        disagree.append(1 if len(counts) > 1 else 0)

        if sum(counts == counts.max()) > 1:
            idx = np.random.choice(
                counts[counts == counts.max()], replace=False, size=1
            )[0]
        else:
            idx = counts.argmax()
        preds.append(classes[idx])

    preds = np.stack(preds)
    disagreement = np.mean(disagree)
    return preds, disagreement


def evaluate_single_model(model_pred, testLabels):
    prod_rolled_probs = torch.argmax(
        torch.mean(torch.log(model_pred["rolled_probs"]), dim=0), dim=1
    )  # equivalent to previous ensemble_testing
    avg_rolled_probs = torch.argmax(
        torch.mean(model_pred["rolled_probs"], dim=0), dim=1
    )  # use mean of probabilities instead of product (i.e. average prob. per class)

    rolledKappas = np.zeros(args.L)
    for iRoll in range(args.L):
        _, pred_class = torch.max(
            model_pred["rolled_probs"][iRoll, :, :], 1
        )  # pred_class is 0-4
        rolledKappas[iRoll] = cohen_kappa_score(
            torch.unsqueeze(pred_class + 1, 1), testLabels.T
        )

    return {
        "mean_rolled_kappa": np.mean(rolledKappas),
        "prod_rolled_probs_kappa": cohen_kappa_score(
            torch.unsqueeze(prod_rolled_probs + 1, 1), testLabels.T
        ),
        "avg_rolled_probs_kappa": cohen_kappa_score(
            torch.unsqueeze(avg_rolled_probs + 1, 1), testLabels.T
        ),
    }


def evaluate_ensemble_model(model_preds, testLabels):
    # Method 1a (Class from Prod. Ensemble)
    class_prod_rolled_probs = torch.stack(
        [
            torch.argmax(
                torch.mean(torch.log(model_pred["rolled_probs"]), dim=0), dim=1
            )
            for model_pred in model_preds
        ]
    )  # equivalent to previous ensemble_testing
    prod_majority_vote, prod_disagreement = majority_vote_ensemble(
        class_prod_rolled_probs
    )
    prod_majority_vote_kappa = cohen_kappa_score(prod_majority_vote + 1, testLabels.T)

    # Method 1b (Class from Avg. Ensemble)
    class_avg_rolled_probs = torch.stack(
        [
            torch.argmax(torch.mean(model_pred["rolled_probs"], dim=0), dim=1)
            for model_pred in model_preds
        ]
    )  # use mean of probabilities instead of product (i.e. average prob. per class)
    avg_majority_vote, avg_disagreement = majority_vote_ensemble(class_avg_rolled_probs)
    avg_majority_vote_kappa = cohen_kappa_score(avg_majority_vote + 1, testLabels.T)

    # Method 2a (Probs. from Prod. Ensemble)
    probs_prod_rolled_probs = torch.stack(
        [
            torch.mean(torch.log(model_pred["rolled_probs"]), dim=0)
            for model_pred in model_preds
        ]
    )
    probs_prod_class = torch.argmax(probs_prod_rolled_probs.mean(dim=0), dim=1)
    probs_prod_kappa = cohen_kappa_score(probs_prod_class + 1, testLabels.T)

    # Method 2b (Probs. from Avg. Ensemble)
    probs_avg_rolled_probs = torch.stack(
        [torch.mean(model_pred["rolled_probs"], dim=0) for model_pred in model_preds]
    )
    probs_avg_class = torch.argmax(probs_avg_rolled_probs.mean(dim=0), dim=1)
    probs_avg_kappa = cohen_kappa_score(probs_avg_class + 1, testLabels.T)

    return {
        "product": {
            "majority_vote": prod_majority_vote_kappa,
            "disagreement": prod_disagreement,
            "prob_ensemble": probs_prod_kappa,
        },
        "average": {
            "majority_vote": avg_majority_vote_kappa,
            "disagreement": avg_disagreement,
            "prob_ensemble": probs_avg_kappa,
        },
    }


def download_logs(local_file):
    project = neptune.get_project(
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxYTU2NTA3ZC0zMzFmLTQzMmEtOTRmMi02YzU2YWQwZGE1NDYifQ==",
        name="kennethborup/ear-eeg-distill",
    )

    # Get IDs of all runs with subjectkappa stored and not currently logging
    project_df = project.fetch_runs_table().to_pandas()
    project_df = project_df.loc[project_df["sys/state"] != "running"]
    ids = project_df["sys/id"]
    ids = project_df.loc[
        project_df["test/subjectKappa"].notna(), "sys/id"
    ].sort_values()

    # Lists of metadata and data to download from each run
    metadata = [
        "sys/id",
        "sys/tags",
        "parameters/temperature",
        "parameters/confidence_threshold",
        "parameters/seed",
        "parameters/n_pseudo_nights",
        "parameters/cv_range",
        "parameters/seed_nights",
    ]
    series_file = "total/allKappas"

    # Local file to save results/data in
    results_file = local_file

    # Load previous results if results_file exists
    if os.path.exists(results_file):
        print(f"{results_file} exists... Will add new results to this file...")
        runs_full = pd.read_csv(results_file, index_col=0)
    else:
        print(f"{results_file} does not exist... Will create new file for results...")
        runs_full = pd.DataFrame([])

    # Remove id of all loaded runs
    if runs_full.shape[0] > 0:
        ids = ids.loc[~ids.isin(runs_full["id"])]

    # Run through all ids and get information
    counter = 0
    for run_id in ids:
        # Get run-object from logger
        run = neptune.init(
            project="kennethborup/ear-eeg-distill",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxYTU2NTA3ZC0zMzFmLTQzMmEtOTRmMi02YzU2YWQwZGE1NDYifQ==",
            mode="read-only",
            run=run_id,
        )

        run_df = {}  # Placeholder for run-data

        # Get all meta-data for run
        for name in metadata:
            try:
                # Get data
                dat = run[name].fetch()
                # Refactor tag-column
                dat = (
                    ", ".join(
                        [
                            tag.replace("_", " ").title()
                            for tag in dat
                            if tag not in ["selftrain"]
                        ]
                    )
                    if name == "sys/tags"
                    else dat
                )
            except:
                dat = np.NaN

            # Rename metadata to exclude "folder" and add to run dictionary
            run_df.update({name.split("/")[-1]: dat})

        # Download series_file to disk and load it
        try:
            run[series_file].download("/tmp/allKappas.csv")
            subject_kappas = pd.read_csv("/tmp/allKappas.csv", header=None, usecols=[1])
            subject_kappas = subject_kappas.transpose().add_prefix(
                f"test/subjectKappa_CV"
            )
            kappa_dict = subject_kappas.to_dict(orient="records")[0]

            # Add subject and average kappa to run_df
            run_df.update(kappa_dict)
            run_df.update({f"avg_test/subjectKappa": np.mean(subject_kappas.values)})
        except:
            print(f"Failed to download total/allKappas for {run_id}... Skipping run...")
            continue

        # Add to full dataframe
        runs_full = runs_full.append(run_df, ignore_index=True)

        # Save to csv occasionally if many runs
        if counter % 10 == 0:
            runs_full.to_csv(results_file)
        counter += 1

    # Fix some formatting
    runs_full["n_pseudo_nights"] = (
        runs_full["n_pseudo_nights"].replace({"None": None}).astype("float")
    )
    runs_full["seed_nights"] = (
        runs_full["seed_nights"].replace({"None": np.nan}).astype("float")
    )

    # Remove duplicates if they should occur
    runs_full = runs_full.drop_duplicates()

    # Save to csv again
    runs_full.to_csv(results_file)
