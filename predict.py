import numpy as np
import sys
import pickle
import pandas as pd

from imblearn.under_sampling import RandomUnderSampler

import src.modules.processor as Processor
import src.modules.model as Model

from argparse import ArgumentParser

# Args parse
parser = ArgumentParser(description="Specifying Input Parameters")
parser.add_argument("-te", "--testfile", help="Specify the full path of the file with TCR sequences")
parser.add_argument("-mf", "--modelfile", help="Specify the full path of the file with trained model")
parser.add_argument("-c", "--chain", default="ce", help="Specify the chain(s) to use (ce, cem). Default: ce")
parser.add_argument("-o", "--outfile", default=sys.stdout, help="Specify output file")
args = parser.parse_args()

chain = args.chain
modelfile = args.modelfile
test_data = pd.read_csv(args.testfile)

assert chain in ["ce", "cem"], "Invalid chain. You can select ce (cdr3b+epitope), cem (cdr3b+epitope+mhc)"

if chain == 'ce':
    X_test, y_test = Processor.dataRepresentationBlosum62WithoutMHCb(test_data), test_data[["binder"]]

    rf_model = pickle.load(open(modelfile, 'rb'))
    print('Evaluating...')
    auc_test, acc_test, sens_test, spec_test = Model.predicMLModel(rf_model, test_data, X_test, y_test, args.outfile)
    print('Done!')

else:
    X_test_mhc, y_test_mhc = Processor.dataRepresentationBlosum62WithMHCb(test_data), test_data[["binder"]]

    rf_model_mhc = pickle.load(open(modelfile, 'rb'))
    print('Evaluating...')
    auc_test, acc_test, sens_test, spec_test = Model.predicMLModel(rf_model_mhc, test_data, X_test_mhc, y_test_mhc, args.outfile)
    print('Done!')
