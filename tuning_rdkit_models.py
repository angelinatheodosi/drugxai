import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

df_clintox = pd.read_csv("data/clintox_rdkit.csv")
df_carcinogens = pd.read_csv("data/carcinogens_rdkit.csv")
df_skin_reaction = pd.read_csv("data/skin_reaction_rdkit.csv")

dataset = {"clintox" : df_clintox, 
           "carcinogens": df_carcinogens, 
           "skin_reaction": df_skin_reaction}


for name, df in dataset.items():
    train_df = df[df["split"] == "train"]
    valid_df = df[df["split"] == "valid"]
    test_df = df[df["split"] == "test"]

    feat_cols = [c for c in df.columns if c not in ["Y", "split", "Drug", "Drug_ID"]]

    X_train = np.clip(train_df[feat_cols].replace([np.inf, -np.inf], np.nan).values, -1e30, 1e30)
    y_train = train_df["Y"].values
    X_valid = np.clip(valid_df[feat_cols].replace([np.inf, -np.inf], np.nan).values, -1e30, 1e30)
    y_valid = valid_df["Y"].values
    X_test = np.clip(test_df[feat_cols].replace([np.inf, -np.inf], np.nan).values, -1e30, 1e30)
    y_test = test_df["Y"].values

    X_combined = np.vstack([X_train, X_valid])
    y_combined = np.concatenate([y_train, y_valid])
    split_index = [-1] * len(X_train) + [0] * len(X_valid)
    ps = PredefinedSplit(test_fold=split_index)
    