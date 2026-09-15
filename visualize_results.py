import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

os.makedirs("results/figures", exist_ok=True)

df = pd.read_csv("results/model_results.csv")

DATASET_LABELS = {
    "carcinogens": "Carcinogens",
    "clintox":     "ClinTox",
    "skin_reaction": "Skin Reaction",
}
MODEL_LABELS = {"logistic": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost"}
FEATURE_COLORS = {"rdkit": "#4C72B0", "chemberta": "#DD8452"}
DATASETS = ["carcinogens", "clintox", "skin_reaction"]
MODELS   = ["logistic", "rf", "xgb"]

# Grouped bar chart per dataset (ROC-AUC & PR-AUC)
# First loop for ROC-AUC, second loop for PR-AUC
for metric in ["test_ROC-AUC", "test_PR-AUC"]:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    metric_label = metric.replace("test_", "") # clear label for y-axis

    # Loop through datasets and plot bars for each model and feature type
    for ax, dataset in zip(axes, DATASETS):
        sub = df[df["dataset"] == dataset] # keeps only the lines from specific dataset
        x = np.arange(len(MODELS))
        width = 0.35

        for i, ftype in enumerate(["rdkit", "chemberta"]):
            vals = [sub[(sub["model"] == m) & (sub["features"] == ftype)][metric].values
                    for m in MODELS]
            vals = [v[0] if len(v) else np.nan for v in vals]
            bars = ax.bar(x + (i - 0.5) * width, vals, width,
                          label=ftype.capitalize(),
                          color=FEATURE_COLORS[ftype], alpha=0.85)
            for bar, val in zip(bars, vals):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f"{val:.2f}", ha="center", va="bottom", fontsize=7.5)

        ax.set_title(DATASET_LABELS[dataset], fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=8.5, rotation=10)
        ax.set_ylabel(metric_label if ax == axes[0] else "")
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle(f"Selected Models — Test {metric_label} (RDKit vs ChemBERTa)", fontsize=13, y=1.01)
    plt.tight_layout()
    fname = f"results/figures/bar_{metric_label.replace('-','_').lower()}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


# Heatmaps (ROC-AUC & PR-AUC)
for metric in ["test_ROC-AUC", "test_PR-AUC"]:
    metric_label = metric.replace("test_", "")

    pivot = df.pivot_table(index=["dataset", "features"], columns="model", values=metric)
    pivot = pivot.loc[[(d, f) for d in DATASETS for f in ["rdkit", "chemberta"]
                        if (d, f) in pivot.index]]

    row_labels = [f"{DATASET_LABELS[d]}\n({f.capitalize()})"
                  for d, f in pivot.index]
    col_labels  = [MODEL_LABELS[m] for m in MODELS if m in pivot.columns]

    data = pivot[[m for m in MODELS if m in pivot.columns]].values

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(data, vmin=0.4, vmax=1.0, cmap="YlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label=metric_label)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=9, color="black" if val < 0.85 else "white")

    # Separator line between rdkit and chemberta rows
    for k in range(1, len(DATASETS)):
        ax.axhline(k * 2 - 0.5, color="white", linewidth=2)

    ax.set_title(f"Test {metric_label} — Selected Models", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fname = f"results/figures/heatmap_{metric_label.replace('-','_').lower()}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


# Best model per (dataset, features) table
# Selection is by validation PR-AUC among the tuned models
best_rows = []
for (dataset, ftype), group in df.groupby(["dataset", "features"]):
    best = group.loc[group["valid_PR-AUC"].idxmax()]
    best_rows.append({
        "Dataset":       DATASET_LABELS.get(dataset, dataset),
        "Features":      ftype.capitalize(),
        "Best Model":    MODEL_LABELS.get(best["model"], best["model"]),
        "Config":        best["selected"],  # tuned or baseline
        "Valid PR-AUC":  round(best["valid_PR-AUC"], 4),
        "Test PR-AUC":   round(best["test_PR-AUC"], 4),
        "Test ROC-AUC":  round(best["test_ROC-AUC"], 4),
    })

best_df = pd.DataFrame(best_rows)
best_df.to_csv("results/figures/best_models_summary.csv", index=False)
print("Saved: results/figures/best_models_summary.csv")
print("Done. All figures in results/figures/")
