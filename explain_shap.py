import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

os.makedirs("results/shap", exist_ok=True)

# Best model per (dataset, feature_type)
best_models = {
    ("clintox",       "rdkit"):    {"mtype": "rf",      "params": {"n_estimators": 100, "min_samples_leaf": 2,  "max_features": 0.3,    "max_depth": None}},
    ("clintox",       "chemberta"):{"mtype": "logistic", "params": {"C": 0.1, "penalty": "l2"}},
    ("carcinogens",   "rdkit"):    {"mtype": "rf",      "params": {"n_estimators": 300, "min_samples_leaf": 10, "max_features": "sqrt", "max_depth": 10}},
    ("carcinogens",   "chemberta"):{"mtype": "rf",      "params": {"n_estimators": 100, "min_samples_leaf": 2,  "max_features": "log2", "max_depth": 3}},
    ("skin_reaction", "rdkit"):    {"mtype": "rf",      "params": {"n_estimators": 300, "min_samples_leaf": 5,  "max_features": "log2", "max_depth": 3}},
    ("skin_reaction", "chemberta"):{"mtype": "rf",      "params": {"n_estimators": 500, "min_samples_leaf": 5,  "max_features": "log2", "max_depth": 3}},
}

for (dataset, ftype), config in best_models.items():
    print(f"\n--- {dataset} | {ftype} ---")

    df = pd.read_csv(f"data/{dataset}_{ftype}.csv")
    feat_cols = [c for c in df.columns if c not in ["Y", "split", "Drug", "Drug_ID"]]

    train_df = df[df["split"] == "train"]
    test_df  = df[df["split"] == "test"]

    X_train = np.clip(train_df[feat_cols].replace([np.inf, -np.inf], np.nan).values, -1e10, 1e10)
    y_train = train_df["Y"].values
    X_test  = np.clip(test_df[feat_cols].replace([np.inf, -np.inf], np.nan).values, -1e10, 1e10)

    mtype  = config["mtype"]
    params = config["params"]


    if mtype == "rf":
        clf = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1, **params))
        ])
    elif mtype == "logistic":
        clf = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=10000, class_weight="balanced",
                                         random_state=42, solver="liblinear", **params))
        ])
    elif mtype == "xgb":
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        spw = neg / pos if pos > 0 else 1.0
        clf = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(scale_pos_weight=spw, random_state=42,
                                    eval_metric="logloss", **params))
        ])

    clf.fit(X_train, y_train)

    # Prepare imputed (+ scaled) data for SHAP
    imputer = clf.named_steps["imputer"]
    model   = clf.named_steps["model"]
    X_test_imp = imputer.transform(X_test)

    if mtype == "logistic":
        scaler     = clf.named_steps["scaler"]
        X_test_imp = scaler.transform(X_test_imp)

    # SHAP explainer
    if isinstance(model, LogisticRegression):
        explainer   = shap.LinearExplainer(model, shap.maskers.Independent(X_test_imp))
        sv          = explainer.shap_values(X_test_imp)
    else:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_imp)
        if isinstance(shap_values, list):
            sv = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            sv = shap_values[:, :, 1]
        else:
            sv = shap_values

    # Beeswarm plot
    plt.figure()
    shap.summary_plot(sv, X_test_imp, feature_names=feat_cols, show=False, max_display=15)
    plt.title(f"{dataset} | {ftype}")
    plt.tight_layout()
    plt.savefig(f"results/shap/{dataset}_{ftype}_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Top 15 features CSV
    mean_abs = np.abs(sv).mean(axis=0)
    top_df = (pd.DataFrame({"feature": feat_cols, "mean_abs_shap": mean_abs})
                .sort_values("mean_abs_shap", ascending=False)
                .head(15)
                .reset_index(drop=True))
    top_df.to_csv(f"results/shap/{dataset}_{ftype}_top_features.csv", index=False)
    print(f"  Top feature: {top_df.iloc[0]['feature']} ({top_df.iloc[0]['mean_abs_shap']:.4f})")

print("\nDone. Results in results/shap/")
