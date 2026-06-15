import os
import pandas as pd
import utils

if __name__ == "__main__":
    datasets = ["clintox", "carcinogens", "skin_reaction"]
    feature_sets = ["rdkit", "chemberta"]
    
    all_results = []

    # Iterate through each dataset and feature set combination
    for dname in datasets:
        print(f"~"*50)
        print(f" COMPARING FEATURES FOR DATASET: {dname.upper()}")
        print(f"~"*50)
        
        for fname in feature_sets:
            # Check if the feature file exists before proceeding
            path = f"data/{dname}_{fname}.csv"
            if not os.path.exists(path):
                print(f"  [Skipping] {fname} - file not found: {path}")
                continue
            
            print(f"  > Feature Set: {fname.upper()}")
            df = pd.read_csv(path)
            
            # Train and evaluate Logistic Regression and Random Forest models
            for mtype in ["logistic", "rf", "xgb"]:
                for use_smote in [False, True]:
                    res = utils.run_experiment(df, mtype, use_smote=use_smote)
                    res.update({"dataset": dname, "features": fname, "model": mtype, "smote": use_smote})
                    all_results.append(res)
                    smote_tag = "+SMOTE" if use_smote else "      "
                    print(f"    - {mtype.upper()} {smote_tag}: Test ROC-AUC = {res['ROC-AUC']:.4f} | Valid = {res['valid_ROC-AUC']:.4f}")

    results_df = pd.DataFrame(all_results)
    
    # Sort
    results_df = results_df.sort_values(["dataset", "features", "model"]).reset_index(drop=True)

    print("~" * 50)
    print(" FINAL COMPARISON OVERVIEW (TEST METRICS)")
    print("~" * 50)

    metric_cols = ["ROC-AUC", "PR-AUC", "valid_ROC-AUC", "valid_PR-AUC"]
    keep_cols = ["dataset", "features", "model", "smote"] + metric_cols
    results_df = results_df[keep_cols + ["n_train", "n_valid", "n_test"]]

    table_df = results_df[keep_cols].pivot_table(
        index=["dataset", "features", "smote"],
        columns="model",
        values=metric_cols,
        aggfunc="first"
    )

    # Flatten column names for a cleaner CSV export
    table_df.columns = [f"{metric}_{model}" for metric, model in table_df.columns]
    print(table_df)

    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/baseline_comparison_results.csv", index=False)
    table_df.to_csv("results/baseline_comparison_table.csv")

    print("\nSuccessfully Saved:")
    print(" - results/baseline_comparison_results.csv (full rows)")
    print(" - results/baseline_comparison_table.csv (pivot table)")
