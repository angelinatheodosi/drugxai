import os
import numpy as np
import pandas as pd
from scipy import stats
import utils
from tuning_models import tune_model


DATASETS = [
    ("clintox",       "ClinTox"),
    ("carcinogens",   "Carcinogens_Lagunin"),
    ("skin_reaction", "Skin_Reaction"),
]
FEATURE_TYPES = ["rdkit", "chemberta"]
SPLITS = ["scaffold", "random"]
SEEDS = [42, 1, 2, 3, 4]
SEED_STEP = 100      # replacement seed = seed + SEED_STEP when a split has a single-class valid/test set
MAX_SEED_TRIES = 10
RESULTS_CSV = "results/split_comparison.csv"                  # one row per run (dataset, features, split, seed)
SUMMARY_CSV = "results/split_comparison_summary.csv"          # scaffold vs random per metric, dataset, features


def run_comparison():
    # Best model type per (dataset, features), chosen by valid PR-AUC on the scaffold split
    # Only the model type is kept because hyperparameters are re-tuned on every split
    best_models = utils.load_best_models()

    all_results = []
    for name, tdc_name in DATASETS:
        for fname in FEATURE_TYPES:
            path = f"data/{name}_{fname}.csv"
            if not os.path.exists(path):
                print(f"File {path} not found. Skipping...")
                continue
            # only replace the split column because features are pre-computed and fixed for each dataset and feature type
            df = pd.read_csv(path)
            feat_cols = utils.get_feature_columns(df)
            mtype = best_models[(name, fname)]["mtype"]

            for method in SPLITS:
                for base_seed in SEEDS:
                    # If valid or test has only one class, move to seed + SEED_STEP until a valid split is found, up to MAX_SEED_TRIES times
                    seed = base_seed
                    for _ in range(MAX_SEED_TRIES):
                        split_df = utils.load_tdc_dataset(tdc_name, split_method=method, seed=seed)
                        df["split"] = df["Drug_ID"].map(split_df.set_index("Drug_ID")["split"])
                        if (df.loc[df["split"] == "valid", "Y"].nunique() == 2
                                and df.loc[df["split"] == "test", "Y"].nunique() == 2):
                            break
                        seed += SEED_STEP
                    else:
                        raise RuntimeError(f"No valid {method} split found for {name} starting from seed={base_seed}")

                    train_df = df[df["split"] == "train"]
                    valid_df = df[df["split"] == "valid"]
                    test_df  = df[df["split"] == "test"]

                    X_train = np.clip(train_df[feat_cols].replace([np.inf, -np.inf], np.nan).values, -1e30, 1e30)
                    y_train = train_df["Y"].values
                    X_valid = np.clip(valid_df[feat_cols].replace([np.inf, -np.inf], np.nan).values, -1e30, 1e30)
                    y_valid = valid_df["Y"].values
                    X_test  = np.clip(test_df[feat_cols].replace([np.inf, -np.inf], np.nan).values, -1e30, 1e30)
                    y_test  = test_df["Y"].values

                    winner = tune_model(mtype, X_train, y_train, X_valid, y_valid, X_test, y_test)
                    all_results.append({
                        "dataset":  name,
                        "features": fname,
                        "model":    mtype,
                        "split":    method,
                        "seed":     seed,
                        "base_seed": base_seed,      # seed != base_seed means the base seed was replaced
                        "test_pos_rate": y_test.mean(),
                        **winner,
                    })

    os.makedirs("results", exist_ok=True)
    results = pd.DataFrame(all_results)
    results.to_csv(RESULTS_CSV, index=False)
    print(f"Saved: {RESULTS_CSV}")
    return results


# Scaffold vs random per (metric, dataset, features): mean and std over seeds
# diff = random − scaffold and Welch t-test p-value
def summarize(results):
    results = results.copy()
    # lift = PR-AUC / positive rate
    results["test_PR_lift"] = results["test_PR-AUC"] / results["test_pos_rate"]

    rows = []
    for met in ["test_ROC-AUC", "test_PR_lift", "valid_ROC-AUC"]:
        for (d, f), g in results.groupby(["dataset", "features"]):
            s, ra = g[g["split"] == "scaffold"][met], g[g["split"] == "random"][met]
            rows.append({
                "metric": met, "dataset": d, "features": f,
                "scaffold_mean": s.mean(),  "scaffold_std": s.std(),
                "random_mean":   ra.mean(), "random_std":   ra.std(),
                "diff":          ra.mean() - s.mean(),
                "p":             stats.ttest_ind(ra, s, equal_var=False).pvalue,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    results = run_comparison()
    summary = summarize(results)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved: {SUMMARY_CSV}")
