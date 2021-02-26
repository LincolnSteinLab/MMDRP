from rdkit import Chem
from rdkit.Chem import AllChem
# import gdsctools
import pandas as pd
import multiprocessing
# from itertools import product
import pickle

# path = "/content/gdrive/My Drive/Python/RNAi/Train_Data/"
# PATH = "/u/ftaj/anaconda3/envs/Python/Data/DRP_Training_Data/"
# path = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/ChEMBL/"


# chembl = pd.read_csv(path+"chembl_27_chemreps.txt", engine='c', sep='\t')

# chembl_smiles = chembl['canonical_smiles']


# for i in range(619999, chembl.shape[0], 10000):
#     chembl['mols'][i:i+10000] = chembl['canonical_smiles'][i:i+10000].apply(lambda x: SmilesToMol(x))
#
# pd.Series.last_valid_index(chembl['mols']) == chembl.shape[0] - 1

# pd.DataFrame.to_pickle(chembl, PATH+"chembl_SmilesToMolDF.pkl")

# chembl = pd.read_pickle(PATH+"chembl_SmilesToMolDF.pkl")

# chembl.shape[0]


def SmilesToMol(smiles, default=None):
    try:
        return Chem.MolFromSmiles(smiles)
    except (ValueError, TypeError):
        return default


def MorganFingerPrint_512(mol):
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
    except (ValueError, TypeError):
        return None


def MorganFingerPrint_1024(mol):
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    except (ValueError, TypeError):
        return None


def MorganFingerPrint_2048(mol):
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    except (ValueError, TypeError):
        return None


def MorganFingerPrint_4096(mol):
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=4096)
    except (ValueError, TypeError):
        return None


if __name__ == '__main__':
    # print("Starting Morgan 1024")
    # pool = multiprocessing.Pool(4)
    # results = pool.map(MorganFingerPrint_1024, range(0, chembl.shape[0]))
    # for i in range(len(results)):
    #     if results[i]:
    #         results[i] = results[i].ToBitString()
    #
    # # final = [i.ToBitString() for i in results]
    # with open(PATH+"Morgan_1024_list.pkl", "wb") as fp:   #Pickling
    #     pickle.dump(results, fp)
    # pool.close()
    # print("Finished Morgan 1024")

    # path = "/u/ftaj/anaconda3/envs/Python/Data/DRP_Training_Data/"
    # path = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/ChEMBL/"
    path = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/DRP_Training_Data/"
    # file_name = "chembl_27_chemreps.txt"
    # file_name = "CTRP_AUC_SMILES.txt"
    file_name = "CTRP_IC50_SMILES.txt"
    # file_name = "GDSC1_AUC_SMILES.txt"
    # file_name = "GDSC2_AUC_SMILES.txt"
    # Read and subset the data
    print("Reading file...")
    # all_data = pd.read_csv(path + file_name, engine='c', sep='\t')
    all_data = pd.read_csv(path + file_name, engine='c', sep=',')
    # full_train = all_data[["ccl_name", "cpd_name", "cpd_smiles", "area_under_curve"]]
    full_train = all_data[["ccl_name", "cpd_name", "cpd_smiles", "ic50"]]
    # full_train = all_data[["canonical_smiles"]]

    pool = multiprocessing.Pool(4)

    ### 512
    # pool_results = pool.map(SmilesToMol, list(full_train["cpd_smiles"]))
    print("Converting SMILES to Mol")
    # pool_results = pool.map(SmilesToMol, list(full_train["canonical_smiles"]))
    pool_results = pool.map(SmilesToMol, list(full_train["cpd_smiles"]))
    # pool.close()
    for cur_func, width in zip([MorganFingerPrint_512, MorganFingerPrint_1024,
                                MorganFingerPrint_2048, MorganFingerPrint_4096],
                               ["512", "1024", "2048", "4096"]):
        # cur_func = locals()["MorganFingerPrint_" + width]()
        print("Starting Morgan", width, "...")
        results = pool.map(cur_func, pool_results)
        for i in range(len(results)):
            if results[i]:
                results[i] = results[i].ToBitString()
        full_train['morgan'] = results
        print("Saving file...")
        # pd.DataFrame.to_pickle(full_train,
        #                        "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/DRP_Training_Data/ChEMBL_Morgan_"
        #                        + width + ".pkl")
        # full_train.to_hdf("CTRP_AUC_MORGAN_" + width + ".hdf", key="df")
        full_train.to_hdf("CTRP_IC50_MORGAN_" + width + ".hdf", key="df")

    pool.close()

    # full_train.to_hdf("GDSC1_AUC_MORGAN.hdf", key="df")
    # full_train.to_hdf("GDSC2_AUC_MORGAN.hdf", key="df")
    # final = [i.ToBitString() for i in results]
    # with open(path+"Morgan_2048_list.pkl", "wb") as fp:   #Pickling
    #     pickle.dump(results, fp)
    # pool.close()
    # print("Finished Morgan 2048")

    # print("Starting Morgan 4096")
    # pool = multiprocessing.Pool(4)
    # results = pool.map(MorganFingerPrint_4096, range(0, chembl.shape[0]))
    # for i in range(len(results)):
    #     if results[i]:
    #         results[i] = results[i].ToBitString()
    #
    # # final = [i.ToBitString() for i in results]
    # with open(PATH+"Morgan_4096_list.pkl", "wb") as fp:   #Pickling
    #     pickle.dump(results, fp)
    # pool.close()
    # print("Finished Morgan 4096")

# with open(PATH+"Morgan_1024_list.pkl", 'rb') as f:
#     mynewlist = pickle.load(f)
#     len(mynewlist)
#     temp = [i.ToBitString() for i in mynewlist[0:10]]
# mynewlist[0].ToBitString()
# chembl['morgan_2_1024'] = None
# chembl['morgan_2_2048'] = None
# chembl['morgan_2_4096'] = None
#
# for i in range(chembl.shape[0]):
#     try:
#         chembl['morgan_2_1024'][i] = AllChem.GetMorganFingerprintAsBitVect(chembl['mols'][i], 2, nBits=1024)
#     except (ValueError, TypeError):
#         continue
#     try:
#         chembl['morgan_2_2048'][i] = AllChem.GetMorganFingerprintAsBitVect(chembl['mols'][i], 2, nBits=2048)
#     except (ValueError, TypeError):
#         continue
#     try:
#         chembl['morgan_2_4096'][i] = AllChem.GetMorganFingerprintAsBitVect(chembl['mols'][i], 2, nBits=4096)
#     except (ValueError, TypeError):
#         continue
#
# pd.DataFrame.to_pickle(chembl, PATH+"chembl_SmilesToMolDF.pkl")
