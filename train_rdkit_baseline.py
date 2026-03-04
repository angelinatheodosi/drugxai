import os
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score


# Extract the feature columns
def get_feature_columns(df: pd.DataFrame, smiles_col="Drug", label_col="Y"):
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    feat_cols = [c for c in numeric_cols if c != label_col]
    return feat_cols

# Train and evaluate the model one time
def train_eval_once(df: pd.DataFrame, model_type="logistic", seed=0):
    smiles_col = "Drug"
    label_col = "Y"
    feat_cols = get_feature_columns(df, smiles_col=smiles_col, label_col=label_col)

    # TDC's split from CSV
    if "split" not in df.columns:
        raise ValueError("Split column not found.")
    
    # Split data into train, validation, and test sets
    train_df = df[df["split"] == "train"].copy()
    valid_df = df[df["split"] == "valid"].copy()
    test_df  = df[df["split"] == "test"].copy()
    
    # Replace inf values with NaN
    X_train = train_df[feat_cols].replace([np.inf, -np.inf], np.nan)
    y_train = train_df[label_col].values

    X_valid = valid_df[feat_cols].replace([np.inf, -np.inf], np.nan)
    y_valid = valid_df[label_col].values

    X_test = test_df[feat_cols].replace([np.inf, -np.inf], np.nan)
    y_test = test_df[label_col].values

    # Select model
    if model_type == "logistic":
        model = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed, solver="liblinear")
    elif model_type == "rf":
        model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=seed, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


    clf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    clf.fit(X_train, y_train)

    p_valid = clf.predict_proba(X_valid)[:, 1]
    p_test  = clf.predict_proba(X_test)[:, 1]
    
    # In case of single class in a split, return NaN
    def compute_auc(y, p):
        if len(np.unique(y)) < 2:
            return np.nan
        return roc_auc_score(y, p)

    def compute_ap(y, p):
        if len(np.unique(y)) < 2:
            return np.nan
        return average_precision_score(y, p)

    out = {
        "model": model_type,
        "seed": seed,
        "n_train": len(train_df),
        "n_valid": len(valid_df),
        "n_test": len(test_df),
        "valid_roc_auc": compute_auc(y_valid, p_valid),
        "valid_pr_auc": compute_ap(y_valid, p_valid),
        "test_roc_auc": compute_auc(y_test, p_test),
        "test_pr_auc": compute_ap(y_test, p_test),
    }
    return out

# Train and evaluate the model multiple times with different seeds
def run_many_seeds(df: pd.DataFrame, model_type="logistic", seeds=(0, 1, 2, 3, 4)):
    rows = []
    for s in seeds:
        rows.append(train_eval_once(df, model_type=model_type, seed=s))
    result = pd.DataFrame(rows)

    # summary mean, standard deviation (ignore NaN if a split had 1 class)
    summary = {
        "valid_roc_auc_mean": np.nanmean(result["valid_roc_auc"]),
        "valid_roc_auc_std":  np.nanstd(result["valid_roc_auc"]),
        "valid_pr_auc_mean":  np.nanmean(result["valid_pr_auc"]),
        "valid_pr_auc_std":   np.nanstd(result["valid_pr_auc"]),
        "test_roc_auc_mean":  np.nanmean(result["test_roc_auc"]),
        "test_roc_auc_std":   np.nanstd(result["test_roc_auc"]),
        "test_pr_auc_mean":   np.nanmean(result["test_pr_auc"]),
        "test_pr_auc_std":    np.nanstd(result["test_pr_auc"]),
    }
    return result, summary


if __name__ == "__main__":
    # Paths to the CSVs created by extract_features.py
    datasets = [
        ("ClinTox", "data/clintox_rdkit2d.csv"),
        ("Carcinogens", "data/carcinogens_rdkit2d.csv")
    ]

    for name, path in datasets:
        if not os.path.exists(path):
            print(f"The file {path} was not found.")
            continue

        print(f"\n{'~'*50}")
        print(f" Processing Dataset: {name}")
        print(f"{'~'*50}")

        try:
            # Load preprocessed CSV (with TDC splits)
            df = pd.read_csv(path)
            if "Drug" not in df.columns:
                 print(f"Error: Column 'Drug' not found in {path}")
                 continue
            
            for mtype in ["logistic", "rf"]:
                print(f"\n> Training {mtype.upper()} model... \n")
                results_df, summary = run_many_seeds(df, model_type=mtype)

                if mtype == "logistic":
                    print(f"Label Column: Y")
                    row = results_df.iloc[0]
                    print(f"Train/Valid/Test sizes: {row['n_train']} / {row['n_valid']} / {row['n_test']}")
                
                print(f"Results for {mtype.upper()} (Mean, Std)")
                print(f"Validation ROC-AUC: {summary['valid_roc_auc_mean']:.4f} | {summary['valid_roc_auc_std']:.4f}")
                print(f"Validation PR-AUC:  {summary['valid_pr_auc_mean']:.4f} | {summary['valid_pr_auc_std']:.4f}")
                print(f"Test ROC-AUC:       {summary['test_roc_auc_mean']:.4f} | {summary['test_roc_auc_std']:.4f}")
                print(f"Test PR-AUC:        {summary['test_pr_auc_mean']:.4f} | {summary['test_pr_auc_std']:.4f}")

        except Exception as e:
            print(f"Error processing {name}: {e}")
