import pandas as pd
import numpy as np
import os
import utils
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

# Get all the available RDKit descriptors
all_descriptor_names = sorted([name for name, _ in Descriptors._descList])

# Initialize the calculator
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(all_descriptor_names)

# Index of Ipc in the descriptor list. The default Ipc (avg=False) grows
# factorially with molecule size (values up to ~1e12), which destabilizes the
# models and SHAP. We override it with the averaged version (avg=True), which is
# bounded (~2-3) and numerically stable.
IPC_IDX = all_descriptor_names.index("Ipc")

def extract_rdkit_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        # If mol is None, return NaN for all features, to keep the same shape
        if mol is None:
            return [np.nan] * len(all_descriptor_names)

        # Use the calculator for batch computation of all descriptors
        values = list(calculator.CalcDescriptors(mol))
        # Replace the exploding Ipc with its averaged, bounded version
        values[IPC_IDX] = Descriptors.Ipc(mol, avg=True)
        return values
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        return [np.nan] * len(all_descriptor_names)

def process_dataset(dataset_name, tdc_name, output_filename):
    print(f"\nProcessing {dataset_name} using TDC '{tdc_name}'...")
    
    # Load the dataset
    df = utils.load_tdc_dataset(tdc_name)
    print(f"Total records: {len(df)}")
    
    # Extract RDKit2D features (optimized batch calculation)
    features = [extract_rdkit_features(s) for s in df['Drug']]
    
    feat_df = pd.DataFrame(features, columns=all_descriptor_names)
    
        
    final_df = pd.concat([df, feat_df], axis=1)
    
    # Save the final dataframe
    os.makedirs('data', exist_ok=True)
    output_path = os.path.join('data', output_filename)
    final_df.to_csv(output_path, index=False)
    print(f"Successfully saved to: {output_path} | shape={final_df.shape}")

if __name__ == "__main__":
    process_dataset('ClinTox', 'ClinTox', 'clintox_rdkit.csv')
    process_dataset('Carcinogens', 'Carcinogens_Lagunin', 'carcinogens_rdkit.csv')
    process_dataset('SkinReaction', 'Skin_Reaction', 'skin_reaction_rdkit.csv')
