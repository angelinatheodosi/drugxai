import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from xgboost import DMatrix
import utils

if __name__ == "__main__":
    os.makedirs("results/shap/local", exist_ok=True)

    best_models = utils.load_best_models()

    for (dataset, ftype), config in best_models.items():
        df = pd.read_csv(f"data/{dataset}_{ftype}.csv")
        feat_cols = utils.get_feature_columns(df)

        train_df = df[df["split"] == "train"]
        test_df  = df[df["split"] == "test"].reset_index(drop=True)

        X_train = np.clip(train_df[feat_cols].replace([np.inf, -np.inf], np.nan).values, -1e30, 1e30)
        y_train = train_df["Y"].values
        X_test  = np.clip(test_df[feat_cols].replace([np.inf, -np.inf], np.nan).values, -1e30, 1e30)
        y_test  = test_df["Y"].values

        mtype  = config["mtype"]
        params = config["params"]

        clf = utils.build_best_pipeline(mtype, params, y_train)
        clf.fit(X_train, y_train)

        imputer = clf.named_steps["imputer"]
        model   = clf.named_steps["model"]
        X_train_proc = imputer.transform(X_train)
        X_test_proc  = imputer.transform(X_test)
        if mtype == "logistic":
            X_train_proc = clf.named_steps["scaler"].transform(X_train_proc)
            X_test_proc  = clf.named_steps["scaler"].transform(X_test_proc)

        # SHAP (background = train, instances explained = test)
        if mtype == "logistic":
            explainer = shap.LinearExplainer(model, shap.maskers.Independent(X_train_proc))
            sv        = explainer.shap_values(X_test_proc)
            base_val  = explainer.expected_value
        elif mtype == "xgb":
            # XGBoost 2.0+ native SHAP
            contribs = model.get_booster().predict(DMatrix(X_test_proc), pred_contribs=True)
            sv       = contribs[:, :-1]
            base_val = float(contribs[0, -1])
        else:
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_proc)
            if isinstance(shap_values, list):
                sv       = shap_values[1]
                base_val = explainer.expected_value[1]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                sv       = shap_values[:, :, 1]
                base_val = (explainer.expected_value[1]
                            if hasattr(explainer.expected_value, "__len__")
                            else explainer.expected_value)
            else:
                sv       = shap_values
                base_val = explainer.expected_value

        probs = clf.predict_proba(X_test)[:, 1]

        tp_candidates = np.where(y_test == 1)[0]
        fp_candidates = np.where(y_test == 0)[0]

        if len(tp_candidates) == 0 or len(fp_candidates) == 0:
            print("  Not enough samples, skipping.")
            continue

        tp_idx = tp_candidates[np.argmax(probs[tp_candidates])]
        fp_idx = fp_candidates[np.argmax(probs[fp_candidates])]

        for idx, label in [(tp_idx, "true_positive"), (fp_idx, "false_positive")]:
            exp = shap.Explanation(
                values=sv[idx],
                base_values=float(base_val),
                data=X_test_proc[idx],
                feature_names=feat_cols
            )
            plt.figure()
            shap.plots.waterfall(exp, max_display=10, show=False)
            plt.title(f"{dataset} | {ftype} | {label.replace('_', ' ')}", fontsize=10)
            plt.tight_layout()
            plt.savefig(f"results/shap/local/{dataset}_{ftype}_{label}.png",
                        dpi=150, bbox_inches="tight")
            plt.close()

    print("Done. Local plots in results/shap/local/")
