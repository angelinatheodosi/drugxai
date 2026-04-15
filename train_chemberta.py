import os
import pandas as pd
import utils

if __name__ == "__main__":
    datasets = [
        ("ClinTox", "data/clintox_chemberta.csv"),
        ("Carcinogens", "data/carcinogens_chemberta.csv")
    ]

    all_res = []

    # Iterate through each dataset and its corresponding feature file
    for name, path in datasets:
        # Verify file existence before processing
        if not os.path.exists(path):
            print(f"File not found: {path}. Skipping...")
            continue

        print(f"\n" + "~"*60)
        print(f" PROCESSING DATASET: {name} (ChemBERTa Features)")
        print(f"~"*60)

        df = pd.read_csv(path)
        
        # Train and evaluate Logistic Regression and Random Forest models using 5-fold/seed cross-validation
        for mtype in ["logistic", "rf"]:
            print(f"\n> Training {mtype.upper()} model...")
            res = utils.run_experiment(df, mtype)

            print(f"Train/Valid/Test sizes: {res['n_train']} / {res['n_valid']} / {res['n_test']}")

            # Print evaluation metrics (ROC-AUC and PR-AUC) for test and validation sets
            print(f"Test ROC-AUC:  {res['ROC-AUC']:.4f}")
            print(f"Test PR-AUC:   {res['PR-AUC']:.4f}")

            print(f"Valid ROC-AUC: {res['valid_ROC-AUC']:.4f}")
            print(f"Valid PR-AUC:  {res['valid_PR-AUC']:.4f}")
            
            res.update({"dataset": name, "model": mtype, "features": "chemberta"})
            all_res.append(res)

    results_df = pd.DataFrame(all_res)
    results_df.to_csv("results_chemberta.csv", index=False)
    print(f"\n[Success] ChemBERTa results saved to: results_chemberta.csv")
