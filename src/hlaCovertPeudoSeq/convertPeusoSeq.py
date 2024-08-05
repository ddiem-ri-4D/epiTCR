import numpy as np
import pandas as pd
from utils import hla_old, hla_new

# Read the final dataset containing HLA data
data_cv = pd.read_csv('../data/finalData/finalWithHLA.csv')

# Replace old HLA names with new HLA names
data_cv['HLA'] = data_cv['HLA'].replace(hla_old, hla_new)

# Read the HLA pseudo sequence data
data_hla = pd.read_csv("../data/hlaCovertPeudoSeq/HLAWithPseudoSeq.csv")

# Create a DataFrame with new HLA names
data_mhc = pd.DataFrame(hla_new, columns=["HLA name"])

# Merge the HLA data with pseudo sequences based on HLA name
result = pd.merge(data_mhc, data_hla, on=["HLA name"])

# Extract lists of HLA names and their corresponding pseudo sequences
lst_hla = np.array(result["HLA name"])
lst_imgt = np.array(result["HLA PSEUDO SEQ"])

# Replace HLA names in the dataset with their pseudo sequences
data_cv['MHC'] = data_cv['HLA'].replace(lst_hla, lst_imgt)

# Save the updated dataset to a new CSV file
data_cv.to_csv('../data/finalData/finalWithHLAConverted.csv', index=False)
