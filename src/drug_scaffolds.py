from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from tqdm import tqdm
from random import Random
from collections import defaultdict
random = Random(42)
import pandas as pd

path = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/DRP_Training_Data/"
file_name = "CTRP_AAC_SMILES.txt"
all_data = pd.read_csv(path + file_name, engine='c', sep=',')

# all_data = all_data[[key_column, class_column, cpd_name_column, smiles_column,
#                           target_column]]

s = all_data['cpd_smiles'].unique()
# scaffolds = defaultdict(set)
# idx2mol = dict(zip(list(range(len(s))), s))
cpd_scaffold_dict = {}
error_smiles = 0
for i, smiles in tqdm(enumerate(s), total=len(s)):
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smiles), includeChirality=False)
        cpd_scaffold_dict[smiles] = scaffold
        # scaffolds[scaffold].add(i)
    except:
        print(smiles + ' returns RDKit error and is thus omitted...')
        error_smiles += 1

cpd_scaffold_df = pd.DataFrame.from_dict(cpd_scaffold_dict, orient="index", columns=["scaffold"])

final_data = all_data.merge(cpd_scaffold_df, left_on="cpd_smiles", right_index=True)

# Save
final_data.to_csv(path + file_name)
