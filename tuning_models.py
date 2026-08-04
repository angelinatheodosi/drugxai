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

            # Combine train and valid sets for hyperparameter tuning, and create a PredefinedSplit rule to ensure that the validation set is used for scoring.
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
                        "model__min_samples_leaf": [1, 2, 5, 10],  # 1 = baseline default
                        "model__max_features":     ["sqrt", "log2", 0.3],
                    },
                    20,  # 192 combinations total; grid includes the baseline config
                ),
                (
                    "xgb",
                    utils.build_pipeline("xgb", {}, y_train),
                    {
                        "model__n_estimators":     [100, 200, 300],
                        "model__max_depth":        [3, 5, 6, 7],       # 6 = baseline default
                        "model__learning_rate":    [0.01, 0.05, 0.1, 0.2, 0.3],  # 0.3 = baseline default
                        "model__subsample":        [0.7, 0.8, 1.0],
                        "model__colsample_bytree": [0.7, 0.8, 1.0],
                    },
                    30,  # 540 combinations total; grid includes the baseline config
                ),
            ]

            for mtype, pipe, param_dist, n_iter in experiments:
                search = RandomizedSearchCV(
                    pipe, param_dist,
                    n_iter=n_iter,
                    scoring="average_precision",
                    cv=ps,
                    refit=False,          # don't refit on train+valid, manual refit on train only below
                    random_state=42,
                    n_jobs=-1,
                )
                # Find the best hyperparameters
                search.fit(X_combined, y_combined)

                # Compare the best tuned model to the baseline model by refitting both on the training set and evaluating on the validation set
                tuned_params    = search.best_params_
                baseline_params = {f"model__{k}": v
                                   for k, v in utils.BASELINE_PARAMS[mtype].items()}

                candidates = []
                for cand, params in [("tuned", tuned_params), ("baseline", baseline_params)]:
                    estimator = clone(pipe).set_params(**params)
                    estimator.fit(X_train, y_train)
                    pv = estimator.predict_proba(X_valid)[:, 1]
                    pt = estimator.predict_proba(X_test)[:, 1]
                    candidates.append({
                        "selected":      cand,
                        "best_params":   str(params),
                        "valid_PR-AUC":  safe_ap(y_valid, pv),
                        "valid_ROC-AUC": safe_auc(y_valid, pv),
                        "test_PR-AUC":   safe_ap(y_test, pt),
                        "test_ROC-AUC":  safe_auc(y_test, pt),
                    })

                # Keep the candidate with the higher validation PR-AUC (NaN treated as -inf)
                winner = max(candidates,
                             key=lambda c: c["valid_PR-AUC"] if not np.isnan(c["valid_PR-AUC"])
                             else -np.inf)

                all_results.append({
                    "dataset":  name,
                    "features": fname,
                    "model":    mtype,
                    **winner,
                })

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(all_results).to_csv("results/model_results.csv", index=False)
    print("Saved: results/model_results.csv")
