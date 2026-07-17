import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.base import clone
import os
import utils


def safe_ap(y, p):
    return average_precision_score(y, p) if len(np.unique(y)) >= 2 else np.nan

def safe_auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) >= 2 else np.nan


if __name__ == "__main__":
    dataset_names = ["clintox", "carcinogens", "skin_reaction"]
    feature_types = ["rdkit", "chemberta"]

    all_results = []
    for fname in feature_types:
        for name in dataset_names:
            path = f"data/{name}_{fname}.csv"

            if not os.path.exists(path):
                print(f"File {path} not found. Skipping...")
                continue
            df = pd.read_csv(path)

            train_df = df[df["split"] == "train"]
            valid_df = df[df["split"] == "valid"]
            test_df  = df[df["split"] == "test"]

            feat_cols = utils.get_feature_columns(df)

            X_train = np.clip(train_df[feat_cols].replace([np.inf, -np.inf], np.nan).values, -1e30, 1e30)
            y_train = train_df["Y"].values
            X_valid = np.clip(valid_df[feat_cols].replace([np.inf, -np.inf], np.nan).values, -1e30, 1e30)
            y_valid = valid_df["Y"].values
            X_test  = np.clip(test_df[feat_cols].replace([np.inf, -np.inf], np.nan).values, -1e30, 1e30)
            y_test  = test_df["Y"].values

            X_combined = np.vstack([X_train, X_valid])
            y_combined = np.concatenate([y_train, y_valid])
            split_index = [-1] * len(X_train) + [0] * len(X_valid)
            ps = PredefinedSplit(test_fold=split_index)

            experiments = [
                (
                    "logistic",
                    utils.build_pipeline("logistic", {}, y_train),
                    {"model__C": [0.001, 0.01, 0.1, 1, 10, 100], "model__penalty": ["l1", "l2"]},
                    12,  # exhaustive: 6 × 2 = 12 combinations
                ),
                (
                    "rf",
                    utils.build_pipeline("rf", {}, y_train),
                    {
                        "model__n_estimators":     [100, 200, 300, 500],
                        "model__max_depth":        [3, 5, 10, None],
                        "model__min_samples_leaf": [2, 5, 10],
                        "model__max_features":     ["sqrt", "log2", 0.3],
                    },
                    20,  # 144 combinations total
                ),
                (
                    "xgb",
                    utils.build_pipeline("xgb", {}, y_train),
                    {
                        "model__n_estimators":     [100, 200, 300],
                        "model__max_depth":        [3, 5, 7],
                        "model__learning_rate":    [0.01, 0.05, 0.1, 0.2],
                        "model__subsample":        [0.7, 0.8, 1.0],
                        "model__colsample_bytree": [0.7, 0.8, 1.0],
                    },
                    30,  # 324 combinations total
                ),
            ]

            for mtype, pipe, param_dist, n_iter in experiments:
                search = RandomizedSearchCV(
                    pipe, param_dist,
                    n_iter=n_iter,
                    scoring="average_precision",
                    cv=ps,
                    refit=False,          # don't refit on train+valid; we refit on train only below
                    random_state=42,
                    n_jobs=-1,
                )
                search.fit(X_combined, y_combined)

                # Refit the winning config on TRAIN ONLY, so valid stays a clean hold-out
                best = clone(pipe).set_params(**search.best_params_)
                best.fit(X_train, y_train)
                p_valid = best.predict_proba(X_valid)[:, 1]
                p_test  = best.predict_proba(X_test)[:, 1]

                all_results.append({
                    "dataset":       name,
                    "features":      fname,
                    "model":         mtype,
                    "best_params":   str(search.best_params_),
                    "valid_PR-AUC":  safe_ap(y_valid, p_valid),
                    "valid_ROC-AUC": safe_auc(y_valid, p_valid),
                    "test_PR-AUC":   safe_ap(y_test, p_test),
                    "test_ROC-AUC":  safe_auc(y_test, p_test),
                })

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(all_results).to_csv("results/tuning_results.csv", index=False)
    print("Saved: results/tuning_results.csv")
