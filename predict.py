import numpy as np
import sys
import pickle
import pandas as pd

from imblearn.under_sampling import RandomUnderSampler

from tqdm import tqdm
from IPython.display import display
import src.modules.processor as Processor
import src.modules.model as Model

from argparse import ArgumentParser

#Args parse
parser = ArgumentParser(description="Specifying Input Parameters")
parser.add_argument("-te", "--testfile", help="Specify the full path of the file with TCR sequences")
parser.add_argument("-mf", "--modelfile", help="Specify the full path of the file with trained model")
parser.add_argument("-c", "--chain", default="ce", help="Specify the chain (s) to use (ce, cem). Default: ce")
parser.add_argument("-o", "--outfile", default=sys.stdout, help="Specify output file")
args = parser.parse_args()

modelfile = args.modelfile
chain = args.chain
test = pd.read_csv(args.testfile)

assert chain in ["ce","cem"]
if chain not in ["ce","cem"]:
    print("Invalid chain. You can select ce (cdr3b+epitope), cem (cdr3b+epitope+mhc)")


if chain=='ce':
    pX_test, py_test = Processor.dataRepresentationBlosum62WithoutMHCb(test), test[["binder"]]

    model_rf = pickle.load(open(modelfile, 'rb'))
    print('Evaluating..')
    auc_test, acc_test, sens_test, spec_test = Model.predicMLModel(model_rf, test, pX_test, py_test, args.outfile)
    print('Done!')

else:
    pX_test_mhc, py_test_mhc = Processor.dataRepresentationBlosum62WithMHCb(test), test[["binder"]]

    model_rf_mhc = pickle.load(open(modelfile, 'rb'))
    print('Evaluating..')
    auc_test, acc_test, sens_test, spec_test = Model.predicMLModel(model_rf_mhc, test, pX_test_mhc, py_test_mhc, args.outfile)
    print('Done!')
