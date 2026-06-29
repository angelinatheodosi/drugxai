import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from xgboost import DMatrix
import utils

if __name__ == "__main__":
    os.makedirs("results/shap", exist_ok=True)

    best_models = utils.load_best_models()

    for (dataset, ftype), config in best_models.items():
        df = pd.read_csv(f"data/{dataset}_{ftype}.csv")
        feat_cols = utils.get_feature_columns(df)

        train_df = df[df["split"] == "train"]
        test_df  = df[df["split"] == "test"]

        X_train = np.clip(train_df[feat_cols].replace([np.inf, -np.inf], np.nan).values, -1e30, 1e30)
        y_train = train_df["Y"].values
        X_test  = np.clip(test_df[feat_cols].replace([np.inf, -np.inf], np.nan).values, -1e30, 1e30)

        mtype  = config["mtype"]
        params = config["params"]

        clf = utils.build_best_pipeline(mtype, params, y_train)
        clf.fit(X_train, y_train)

        imputer = clf.named_steps["imputer"]
        model   = clf.named_steps["model"]
        X_test_proc = imputer.transform(X_test)
        if mtype == "logistic":
            X_test_proc = clf.named_steps["scaler"].transform(X_test_proc)

        # SHAP explainer
        if mtype == "logistic":
            explainer = shap.LinearExplainer(model, shap.maskers.Independent(X_test_proc))
            sv = explainer.shap_values(X_test_proc)
        elif mtype == "xgb":
            # XGBoost 2.0+ native SHAP
            contribs = model.get_booster().predict(DMatrix(X_test_proc), pred_contribs=True)
            sv = contribs[:, :-1]
        else:
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_proc)
            if isinstance(shap_values, list):
                sv = shap_values[1]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                sv = shap_values[:, :, 1]
            else:
                sv = shap_values

        # Beeswarm plot
        plt.figure()
        shap.summary_plot(sv, X_test_proc, feature_names=feat_cols, show=False, max_display=15)
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

    print("Done. Results in results/shap/")
