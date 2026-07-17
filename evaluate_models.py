import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc as sklearn_auc,
    precision_recall_curve, average_precision_score,
    classification_report, roc_auc_score,
)
import utils

if __name__ == "__main__":
    os.makedirs("results/evaluation", exist_ok=True)

    DATASET_LABELS = {
        "carcinogens":   "Carcinogens",
        "clintox":       "ClinTox",
        "skin_reaction": "Skin Reaction",
    }
    MODEL_LABELS = {"logistic": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost"}
    DATASETS = ["carcinogens", "clintox", "skin_reaction"]
    FEATURE_COLORS = {"rdkit": "#4C72B0", "chemberta": "#DD8452"}

    best_models = utils.load_best_models()
    report_rows = []
    roc_data = {d: {} for d in DATASETS}

    for (dataset, ftype), config in best_models.items():
        mtype  = config["mtype"]
        params = config["params"]
        title  = f"{DATASET_LABELS[dataset]} | {ftype.capitalize()} | {MODEL_LABELS[mtype]}"

        df = pd.read_csv(f"data/{dataset}_{ftype}.csv")
        feat_cols = utils.get_feature_columns(df)

        train_df = df[df["split"] == "train"]
        test_df  = df[df["split"] == "test"].reset_index(drop=True)

        X_train = np.clip(train_df[feat_cols].replace([np.inf, -np.inf], np.nan).values, -1e30, 1e30)
        y_train = train_df["Y"].values
        X_test  = np.clip(test_df[feat_cols].replace([np.inf, -np.inf], np.nan).values, -1e30, 1e30)
        y_test  = test_df["Y"].values

        if len(np.unique(y_test)) < 2:
            print(f"[Skip] {dataset}/{ftype}: single class in test set.")
            continue

        clf = utils.build_pipeline(mtype, params, y_train)
        clf.fit(X_train, y_train)

        probs = clf.predict_proba(X_test)[:, 1]
        preds = clf.predict(X_test)

        # Confusion matrix
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ConfusionMatrixDisplay(confusion_matrix(y_test, preds)).plot(
            ax=axes[0], colorbar=False)
        axes[0].set_title("Raw counts")
        ConfusionMatrixDisplay(
            confusion_matrix(y_test, preds, normalize="true")
        ).plot(ax=axes[1], colorbar=False, values_format=".2f")
        axes[1].set_title("Normalized")
        fig.suptitle(title, fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"results/evaluation/{dataset}_{ftype}_confusion.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc_val = sklearn_auc(fpr, tpr)
        roc_data[dataset][ftype] = (fpr, tpr, auc_val, mtype)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc_val:.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve — {DATASET_LABELS[dataset]} | {ftype.capitalize()}")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"results/evaluation/{dataset}_{ftype}_roc.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, probs)
        ap = average_precision_score(y_test, probs)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(recall, precision, linewidth=2, label=f"AP = {ap:.3f}")
        baseline = y_test.mean()
        ax.axhline(baseline, color="gray", linestyle="--", alpha=0.6,
                   label=f"Baseline (prevalence = {baseline:.2f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"PR Curve — {DATASET_LABELS[dataset]} | {ftype.capitalize()}")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"results/evaluation/{dataset}_{ftype}_pr_curve.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

        # Classification report
        rep = classification_report(y_test, preds, output_dict=True, zero_division=0)
        report_rows.append({
            "Dataset":          DATASET_LABELS[dataset],
            "Features":         ftype.capitalize(),
            "Model":            MODEL_LABELS[mtype],
            "Precision (0)":    round(rep["0"]["precision"], 4),
            "Recall (0)":       round(rep["0"]["recall"], 4),
            "F1 (0)":           round(rep["0"]["f1-score"], 4),
            "Precision (1)":    round(rep["1"]["precision"], 4),
            "Recall (1)":       round(rep["1"]["recall"], 4),
            "F1 (1)":           round(rep["1"]["f1-score"], 4),
            "Macro F1":         round(rep["macro avg"]["f1-score"], 4),
            "ROC-AUC":          round(roc_auc_score(y_test, probs), 4),
            "PR-AUC":           round(ap, 4),
        })

    # Summary ROC plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, dataset in zip(axes, DATASETS):
        for ftype, color in FEATURE_COLORS.items():
            if ftype in roc_data[dataset]:
                fpr, tpr, auc_val, mtype = roc_data[dataset][ftype]
                ax.plot(fpr, tpr, color=color, linewidth=2,
                        label=f"{ftype.capitalize()} (AUC={auc_val:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_title(DATASET_LABELS[dataset], fontsize=12, fontweight="bold")
        ax.set_xlabel("FPR")
        if ax is axes[0]:
            ax.set_ylabel("TPR")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("ROC Curves — Best Tuned Models (RDKit vs ChemBERTa)", fontsize=13)
    plt.tight_layout()
    plt.savefig("results/evaluation/roc_all.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Save classification report CSV
    pd.DataFrame(report_rows).to_csv("results/evaluation/classification_report.csv", index=False)
    print("Saved: results/evaluation/")
    print("Done. All evaluation results in results/evaluation/")
