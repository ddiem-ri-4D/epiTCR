import numpy as np
import pandas as pd
from utils import *

DATA_CV = pd.read_csv('../data/finalData/finalWithHLA.csv')

DATA_CV['HLA'] = DATA_CV['HLA'].replace(hla_old, hla_new)

data_hla = pd.read_csv("../data/hlaCovertPeudoSeq/HLAWithPseudoSeq.csv")
data_mhc = pd.DataFrame(hla_new, columns = ["HLA name"])
result = pd.merge(data_mhc, data_hla, on=["HLA name"])

lst_HLA = np.array(result["HLA name"])
lst_imgt = np.array(result["HLA PSEUDO SEQ"])

DATA_CV['MHC'] = DATA_CV['HLA'].replace(lst_HLA, lst_imgt)

DATA_CV.to_csv('../data/finalData/finalWithHLAConverted.csv', index=False)
