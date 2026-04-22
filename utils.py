import numpy as np
import pandas as pd
from tdc.single_pred import Tox
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

# Load dataset from TDC and perform scaffold split
def load_tdc_dataset(tdc_name):
    data = Tox(name=tdc_name)
    split_data = data.get_split(method="scaffold")

    # Combine split sets
    df = pd.concat([split_data["train"], split_data["valid"], split_data["test"]], axis=0)
    df["split"] = (["train"] * len(split_data["train"])
                   + ["valid"] * len(split_data["valid"])
                   + ["test"] * len(split_data["test"]))
    return df.reset_index(drop=True)

# Identify numeric feature columns by excluding label and split columns
def get_feature_columns(df: pd.DataFrame, label_col="Y"):
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    feat_cols = [c for c in numeric_cols if c != label_col and c != "split"]
    return feat_cols

# Train and evaluate a model
def run_experiment(df: pd.DataFrame, model_type="logistic"):
    label_col = "Y"
    feat_cols = get_feature_columns(df, label_col=label_col)

    if "split" not in df.columns:
        raise ValueError("Split column not found in DataFrame.")

    # Isolate data subsets
    train_df = df[df["split"] == "train"].copy()
    valid_df = df[df["split"] == "valid"].copy()
    test_df  = df[df["split"] == "test"].copy()

    # Cleaning: replace inf with NaN and cap extreme values
    X_train = np.clip(train_df[feat_cols].replace([np.inf, -np.inf], np.nan).values, -1e30, 1e30)
    y_train = train_df[label_col].values

    X_valid = np.clip(valid_df[feat_cols].replace([np.inf, -np.inf], np.nan).values, -1e30, 1e30)
    y_valid = valid_df[label_col].values

    X_test = np.clip(test_df[feat_cols].replace([np.inf, -np.inf], np.nan).values, -1e30, 1e30)
    y_test = test_df[label_col].values

    if model_type == "logistic":
        model = LogisticRegression(max_iter=10000, class_weight="balanced", random_state=42, solver="liblinear")
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model)
        ]
    elif model_type == "rf":
        model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model)
        ]
    elif model_type == "xgb":
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        spw = neg / pos if pos > 0 else 1.0
        model = XGBClassifier(n_estimators=100, scale_pos_weight=spw, random_state=42,eval_metric="logloss")
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model)
        ]
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

   
    clf = ImbPipeline(steps=steps)
    

    clf.fit(X_train, y_train)

    # Predict probabilities for evaluation
    p_valid = clf.predict_proba(X_valid)[:, 1]
    p_test  = clf.predict_proba(X_test)[:, 1]

    # In case of single class in a split, return NaN
    def compute_auc(y, p):
        if len(np.unique(y)) < 2: return np.nan
        return roc_auc_score(y, p)

    def compute_ap(y, p):
        if len(np.unique(y)) < 2: return np.nan
        return average_precision_score(y, p)

    return {
        "ROC-AUC": compute_auc(y_test, p_test),
        "PR-AUC": compute_ap(y_test, p_test),
        "valid_ROC-AUC": compute_auc(y_valid, p_valid),
        "valid_PR-AUC": compute_ap(y_valid, p_valid),
        "n_train": len(train_df),
        "n_valid": len(valid_df),
        "n_test": len(test_df)
    }
