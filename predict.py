import numpy as np
import sys
import pickle
import pandas as pd
from argparse import ArgumentParser

import src.modules.processor as Processor

# Argument parsing
parser = ArgumentParser(description="Specifying Input Parameters")
parser.add_argument("-te", "--testfile", required=True, help="Specify the full path of the file with TCR sequences")
parser.add_argument("-mf", "--modelfile", required=True, help="Specify the full path of the file with trained model")
parser.add_argument("-c", "--chain", default="ce", help="Specify the chain(s) to use (ce, cem). Default: ce")
parser.add_argument("-o", "--outfile", default="output.csv", help="Specify output file")
args = parser.parse_args()

chain = args.chain
modelfile = args.modelfile
outfile = args.outfile

# Load test data
test_data = pd.read_csv(args.testfile)

assert chain in ["ce", "cem"], "Invalid chain. You can select ce (cdr3b+epitope), cem (cdr3b+epitope+mhc)"

if chain == 'ce':
    X_test = Processor.dataRepresentationBlosum62WithoutMHCb(test_data)
    y_test = test_data[["binder"]]

    rf_model = pickle.load(open(modelfile, 'rb'))
    print('Evaluating Random Forest without MHC...')
    
    y_rf_test_proba = rf_model.predict_proba(X_test)[:, 1]
    df_test_rf = pd.DataFrame({'predict_proba': y_rf_test_proba})
    df_prob_test_rf = pd.concat([test_data.reset_index(drop=True), df_test_rf], axis=1)
    df_prob_test_rf['binder_pred'] = (df_prob_test_rf['predict_proba'] >= 0.5).astype(int)
    
    df_prob_test_rf.to_csv(outfile, index=False)
    print('Done!')

else:
    X_test_mhc = Processor.dataRepresentationBlosum62WithMHCb(test_data)
    y_test_mhc = test_data[["binder"]]

    rf_model_mhc = pickle.load(open(modelfile, 'rb'))
    print('Evaluating Random Forest with MHC...')
    
    y_rf_test_proba_mhc = rf_model_mhc.predict_proba(X_test_mhc)[:, 1]
    df_test_rf_mhc = pd.DataFrame({'predict_proba': y_rf_test_proba_mhc})
    df_prob_test_rf_mhc = pd.concat([test_data.reset_index(drop=True), df_test_rf_mhc], axis=1)
    df_prob_test_rf_mhc['binder_pred'] = (df_prob_test_rf_mhc['predict_proba'] >= 0.5).astype(int)
    
    df_prob_test_rf_mhc.to_csv(outfile, index=False)
    print('Done!')
