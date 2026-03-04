import os
import pandas as pd
import numpy as np
import torch
import utils
from molfeat.trans.pretrained.hf_transformers import PretrainedHFTransformer, HFModel

# Convert torch tensors to numpy arrays
def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# Extracts molecular embeddings using a pretrained Transformer model (ChemBERTa)
def extract_features_transformer(df, output_path, batch_size=128):
    model_id = "seyonec/ChemBERTa-zinc-base-v1"
    print(f"\n Extracting transformer features with: {model_id} (mean pooling)")

    try:
        # Initialize HuggingFace model
        hf_model = HFModel.from_pretrained(model=model_id, tokenizer=model_id)
        transformer = PretrainedHFTransformer(kind=hf_model, notation="smiles", pooling="mean")

        smiles = df["Drug"].astype(str).tolist()
        all_embeds = []
        embedding_dim = None
        
        # Process smiles in batches
        for i in range(0, len(smiles), batch_size):
            batch = smiles[i:i + batch_size]
            try:
                emb = to_numpy(transformer.transform(batch))
                if embedding_dim is None:
                    embedding_dim = emb.shape[1]
                all_embeds.append(emb)
            except Exception:
                # Fallback to per-molecule processing if batch transformation fails
                print(f"Batch {i//batch_size} failed, fallback to per-molecule...")
                row_embeds = []
                if embedding_dim is None:
                    for s in batch:
                        try:
                            e = to_numpy(transformer.transform([s]))
                            embedding_dim = e.shape[1]
                            break
                        except Exception: continue
                
                for s in batch:
                    try:
                        e = to_numpy(transformer.transform([s]))
                        row_embeds.append(e[0])
                    except Exception:
                        # Use NaN values for failed SMILES (maintain row consistency)
                        row_embeds.append(np.full((embedding_dim,), np.nan))
                all_embeds.append(np.vstack(row_embeds))

        # Combine all embeddings and merge with the original dataframe
        embeddings = np.vstack(all_embeds)
        if embeddings.shape[0] != len(df):
            raise RuntimeError(f"Row mismatch: {embeddings.shape[0]} vs {len(df)}")

        feat_cols = [f"feat_{i}" for i in range(embeddings.shape[1])]
        feat_df = pd.DataFrame(embeddings, columns=feat_cols)
        final_df = pd.concat([df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)

        final_df.to_csv(output_path, index=False)
        print(f"Successfully saved to: {output_path}")
        return True

    except Exception as e:
        print(f"Error with ChemBERTa: {e}")
        return False


# Load dataset and extract transformer features
def process_dataset(dataset_name, tdc_name, output_prefix):
    print(f"\nProcessing {dataset_name}...")
    df = utils.load_tdc_dataset(tdc_name)
    os.makedirs("data", exist_ok=True)
    extract_features_transformer(df, f"data/{output_prefix}_chemberta.csv")


if __name__ == "__main__":
    process_dataset("ClinTox", "ClinTox", "clintox")
    process_dataset("Carcinogens", "Carcinogens_Lagunin", "carcinogens")
