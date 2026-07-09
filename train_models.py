import os
import pandas as pd
import utils

if __name__ == "__main__":
    datasets = ["clintox", "carcinogens", "skin_reaction"]
    feature_sets = ["rdkit", "chemberta"]
    
    all_results = []

    for dname in datasets:
        for fname in feature_sets:
            path = f"data/{dname}_{fname}.csv"
            if not os.path.exists(path):
                print(f"[Skip] {path} not found.")
                continue

            df = pd.read_csv(path)

            # Train and evaluate all three classifiers
            for mtype in ["logistic", "rf", "xgb"]:
                res = utils.run_experiment(df, mtype)
                res.update({"dataset": dname, "features": fname, "model": mtype})
                all_results.append(res)

    results_df = pd.DataFrame(all_results)
    
    # Sort
    results_df = results_df.sort_values(["dataset", "features", "model"]).reset_index(drop=True)



    metric_cols = ["ROC-AUC", "PR-AUC", "valid_ROC-AUC", "valid_PR-AUC"]
    keep_cols = ["dataset", "features", "model"] + metric_cols
    results_df = results_df[keep_cols]

    table_df = results_df.pivot_table(
        index=["dataset", "features"],
        columns="model",
        values=metric_cols,
        aggfunc="first"
    )

    # Flatten column names for a cleaner CSV export
    table_df.columns = [f"{metric}_{model}" for metric, model in table_df.columns]

    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/baseline_comparison_results.csv", index=False)
    table_df.to_csv("results/baseline_comparison_table.csv")

    print("Saved: results/baseline_comparison_results.csv")
    print("Saved: results/baseline_comparison_table.csv")
