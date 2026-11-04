import os
import numpy as np
import pandas as pd
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


if __name__ == "__main__":
    # Best model type per (dataset, features), chosen by valid PR-AUC on the scaffold split.
    # Only the model type is kept because hyperparameters are re-tuned on every split.
    best_models = utils.load_best_models()

    all_results = []
    for name, tdc_name in DATASETS:
        for fname in FEATURE_TYPES:
            path = f"data/{name}_{fname}.csv"
            if not os.path.exists(path):
                print(f"File {path} not found. Skipping...")
                continue
            # We only replace the split column because features are pre-computed and fixed for each dataset and feature type.
            df = pd.read_csv(path)
            feat_cols = utils.get_feature_columns(df)
            mtype = best_models[(name, fname)]["mtype"]

            for method in SPLITS:
                for base_seed in SEEDS:
                    # If valid or test has only one class, ROC/PR-AUC are undefined and tuning is blind,
                    # so move to seed + SEED_STEP (large step to avoid reusing another seed in SEEDS).
                    seed = base_seed
                    for _ in range(MAX_SEED_TRIES):
                        split_df = utils.load_tdc_dataset(tdc_name, split_method=method, seed=seed)
                        df["split"] = df["Drug_ID"].map(split_df.set_index("Drug_ID")["split"])
                        if (df.loc[df["split"] == "valid", "Y"].nunique() == 2
                                and df.loc[df["split"] == "test", "Y"].nunique() == 2):
                            break
                        print(f"{name} {method} seed={seed}: single-class valid/test, trying seed={seed + SEED_STEP}")
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
                        "base_seed": base_seed,
                        "test_pos_rate": y_test.mean(),  
                        **winner,
                    })
                    print(f"{name:13s} {fname:9s} {mtype:8s} {method:8s} seed={seed:<2d} "
                          f"test ROC-AUC={winner['test_ROC-AUC']:.3f} PR-AUC={winner['test_PR-AUC']:.3f}")

    os.makedirs("results", exist_ok=True)
    results = pd.DataFrame(all_results)
    results.to_csv("results/split_comparison.csv", index=False)
    print("Saved: results/split_comparison.csv")

    # Mean ± std over seeds, and random − scaffold difference on the test set
    metrics = ["test_ROC-AUC", "test_PR-AUC", "test_pos_rate"]
    summary = (results.groupby(["dataset", "features", "model", "split"])[metrics]
               .agg(["mean", "std"]).round(3))
    summary.to_csv("results/split_comparison_summary.csv")
    print(summary)

    means = results.groupby(["dataset", "features", "model", "split"])[metrics[:2]].mean().unstack("split")
    for m in metrics[:2]:
        means[(m, "random - scaffold")] = means[(m, "random")] - means[(m, "scaffold")]
    print("\nMean test scores and difference (random − scaffold):")
    print(means.round(3))
