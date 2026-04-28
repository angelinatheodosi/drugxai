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
from xgboost import XGBClassifier, DMatrix

os.makedirs("results/shap/local", exist_ok=True)

# Same best models with explain_shap.py
best_models = {
    ("clintox",       "rdkit"):    {"mtype": "rf",      "params": {"n_estimators": 100, "min_samples_leaf": 2,  "max_features": 0.3,    "max_depth": None}},
    ("clintox",       "chemberta"):{"mtype": "xgb",     "params": {}},
    ("carcinogens",   "rdkit"):    {"mtype": "rf",      "params": {"n_estimators": 300, "min_samples_leaf": 10, "max_features": "sqrt", "max_depth": 10}},
    ("carcinogens",   "chemberta"):{"mtype": "rf",      "params": {"n_estimators": 100, "min_samples_leaf": 2,  "max_features": "log2", "max_depth": 3}},
    ("skin_reaction", "rdkit"):    {"mtype": "xgb",     "params": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05, "subsample": 1.0, "colsample_bytree": 0.8}},
    ("skin_reaction", "chemberta"):{"mtype": "rf",      "params": {"n_estimators": 500, "min_samples_leaf": 5,  "max_features": "log2", "max_depth": 3}},
}

for (dataset, ftype), config in best_models.items():
    print(f"\n--- {dataset} | {ftype} ---")

    df = pd.read_csv(f"data/{dataset}_{ftype}.csv")
    feat_cols = [c for c in df.columns if c not in ["Y", "split", "Drug", "Drug_ID"] and pd.api.types.is_numeric_dtype(df[c])]

    train_df = df[df["split"] == "train"]
    test_df  = df[df["split"] == "test"].reset_index(drop=True)

    X_train = np.clip(train_df[feat_cols].replace([np.inf, -np.inf], np.nan).values, -1e10, 1e10)
    y_train = train_df["Y"].values
    X_test  = np.clip(test_df[feat_cols].replace([np.inf, -np.inf], np.nan).values, -1e10, 1e10)
    y_test  = test_df["Y"].values

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
                                    eval_metric="logloss", base_score=0.5, **params))
        ])

    clf.fit(X_train, y_train)


    imputer = clf.named_steps["imputer"]
    model   = clf.named_steps["model"]
    X_test_imp = imputer.transform(X_test)

    if mtype == "logistic":
        scaler     = clf.named_steps["scaler"]
        X_test_imp = scaler.transform(X_test_imp)

    # SHAP
    if isinstance(model, LogisticRegression):
        explainer   = shap.LinearExplainer(model, shap.maskers.Independent(X_test_imp))
        sv          = explainer.shap_values(X_test_imp)
        base_val    = explainer.expected_value
    elif isinstance(model, XGBClassifier):
        # XGBoost 2.0+ native SHAP (avoids SHAP library incompatibility)
        contribs = model.get_booster().predict(DMatrix(X_test_imp), pred_contribs=True)
        sv       = contribs[:, :-1]   # last column is bias term
        base_val = float(contribs[0, -1])
    else:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_imp)
        if isinstance(shap_values, list):
            sv       = shap_values[1]
            base_val = explainer.expected_value[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            sv       = shap_values[:, :, 1]
            base_val = explainer.expected_value[1] if hasattr(explainer.expected_value, '__len__') else explainer.expected_value
        else:
            sv       = shap_values
            base_val = explainer.expected_value

    # Predicted probabilities from X_test before imputation
    probs = clf.predict_proba(X_test)[:, 1]

    # Molecule selection
    # True Positive (y=1) and False Positive (y=0), and their highest predicted probability
    tp_candidates = np.where(y_test == 1)[0]
    fp_candidates = np.where(y_test == 0)[0]

    if len(tp_candidates) == 0 or len(fp_candidates) == 0:
        print("Not enough sample, skipping.")
        continue

    tp_idx = tp_candidates[np.argmax(probs[tp_candidates])]
    fp_idx = fp_candidates[np.argmax(probs[fp_candidates])]

    # Waterfall plots
    for idx, label in [(tp_idx, "true_positive"), (fp_idx, "false_positive")]:
        exp = shap.Explanation(
            values=sv[idx],
            base_values=float(base_val),
            data=X_test_imp[idx],
            feature_names=feat_cols
        )

        plt.figure()
        shap.plots.waterfall(exp, max_display=10, show=False)
        plt.title(f"{dataset} | {ftype} | {label.replace('_', ' ')}", fontsize=10)
        plt.tight_layout()
        plt.savefig(f"results/shap/local/{dataset}_{ftype}_{label}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {label} (idx={idx}, y={y_test[idx]}, prob={probs[idx]:.3f})")

print("\nDone. Local plots in results/shap/local/")
