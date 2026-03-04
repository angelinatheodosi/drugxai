import pandas as pd
import numpy as np
import os
from tdc.single_pred import Tox
from rdkit import Chem
from rdkit.Chem import Descriptors

# Get all available RDKit descriptors
descriptor_names = [name for name, _ in Descriptors._descList]

# Create a dictionary of descriptor calculators
descriptor_calculators = {name: func for name, func in Descriptors._descList}
col_names = descriptor_names

def extract_rdkit_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        # If mol is None, return NaN for all features, to keep the same shape
        if mol is None:
            print(f"Could not parse SMILES: {smiles}")
            return [np.nan] * len(col_names)
        
        # Calculate all descriptors for the molecule
        features = []
        for name in col_names:
            try:
                val = descriptor_calculators[name](mol)
                features.append(val)
            except:
                features.append(np.nan)
        return features
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        return [np.nan] * len(col_names)

def process_and_save(dataset_name, output_filename):
    print(f"\nLoading {dataset_name} with TDC Scaffold Split")
    data = Tox(name=dataset_name)
    
    # TDC scaffold split
    split_data = data.get_split(method='scaffold')
    train_df = split_data['train']
    valid_df = split_data['valid']
    test_df  = split_data['test']

    # Add split column, to keep track of the split
    train_df['split'] = 'train'
    valid_df['split'] = 'valid'
    test_df['split'] = 'test'

    # Combine into one DataFrame
    df = pd.concat([train_df, valid_df, test_df], axis=0).reset_index(drop=True)
    print(f"Total records in {dataset_name}: {len(df)}")
    
    # Extract Features
    print("Extracting RDKit2D features from SMILES...")
    features = []
    for smiles in df['Drug']:
        features.append(extract_rdkit_features(smiles))
    
    feat_df = pd.DataFrame(features, columns=col_names)
    
    # Cleaning (Inf values to NaN)
    print("Converting infinite values to NaNs...")
    feat_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
    # Concatenate and Save
    final_df = pd.concat([df, feat_df], axis=1)
    
    os.makedirs('data', exist_ok=True)
    output_path = os.path.join('data', output_filename)
    final_df.to_csv(output_path, index=False)
    print(f"Successfully saved data with TDC splits to: {output_path}")

if __name__ == "__main__":
    process_and_save('ClinTox', 'clintox_rdkit2d.csv')
    process_and_save('Carcinogens_Lagunin', 'carcinogens_rdkit2d.csv')
