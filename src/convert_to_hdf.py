import pandas as pd
import glob

PATH = "/Data/DRP_Training_Data/"

cur_filenames = glob.glob(PATH + "*AAC_MORGAN*") + glob.glob(PATH + "*No_NA_ProteinQuant.csv") +\
                                                  glob.glob(PATH + "*Expression.csv") +\
                                                  glob.glob(PATH + "*CopyNumber.csv") +\
                                                  glob.glob(PATH + "*CGC_Mutations_by_Cell.csv")

for file_name in cur_filenames:
    # Read as CSV
    print("Reading:", file_name)
    cur_file = pd.read_csv(file_name)
    # Save as HDF(5)
    hdf_name = file_name.split('.')[0] + ".hdf"
    print("Saving as", hdf_name)
    cur_file.to_hdf(hdf_name, key='df', mode='w')
