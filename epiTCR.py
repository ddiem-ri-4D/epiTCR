import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from argparse import ArgumentParser
import src.modules.processor as Processor
import src.modules.model as Model

# Argument parsing
parser = ArgumentParser(description="Specifying Input Parameters")
parser.add_argument("-tr", "--trainfile", required=True, help="Specify the full path of the training file with TCR sequences")
parser.add_argument("-te", "--testfile", required=True, help="Specify the full path of the test file with TCR sequences")
parser.add_argument("-c", "--chain", default="ce", help="Specify the chain(s) to use (ce, cem). Default: ce")
parser.add_argument("-o", "--outfile", default="output.csv", help="Specify output file")
args = parser.parse_args()

chain = args.chain

if chain not in ["ce", "cem"]:
    sys.exit("Invalid chain. You can select ce (cdr3b+epitope) or cem (cdr3b+epitope+mhc)")

print('Loading and encoding the dataset...')
train_data = pd.read_csv(args.trainfile)
test_data = pd.read_csv(args.testfile)

models = [
    ('Random Forest - without MHC', RandomForestClassifier(bootstrap=False, max_features=15, n_estimators=300, n_jobs=-1, random_state=42)),
    ('Random Forest - with MHC', RandomForestClassifier(max_features=20, n_estimators=300, n_jobs=-1, random_state=42))
]

if chain == 'ce':
    X_train, y_train = Processor.dataRepresentationDownsamplingWithoutMHCb(train_data)
    X_test = Processor.dataRepresentationBlosum62WithoutMHCb(test_data)
    y_test = test_data[["binder"]]

    print('Training Random Forest without MHC...')
    rf_model = models[0][1].fit(X_train, np.ravel(y_train))
    Model.saveByPickle(rf_model, "./models/rdforestWithoutMHCModel.pickle")

    print('Evaluating Random Forest without MHC...')
    y_rf_test_proba = rf_model.predict_proba(X_test)[:, 1]
    df_test_rf = pd.DataFrame({'predict_proba': y_rf_test_proba})
    df_prob_test_rf = pd.concat([test_data.reset_index(drop=True), df_test_rf], axis=1)
    df_prob_test_rf['binder_pred'] = (df_prob_test_rf['predict_proba'] >= 0.5).astype(int)
    df_prob_test_rf.to_csv(args.outfile, index=False)

else:
    X_train_mhc, y_train_mhc = Processor.dataRepresentationDownsamplingWithMHCb(train_data)
    X_test_mhc = Processor.dataRepresentationBlosum62WithMHCb(test_data)
    y_test_mhc = test_data[["binder"]]

    print('Training Random Forest with MHC...')
    rf_model_mhc = models[1][1].fit(X_train_mhc, np.ravel(y_train_mhc))
    Model.saveByPickle(rf_model_mhc, "./models/rdforestWithMHCModel.pickle")

    print('Evaluating Random Forest with MHC...')
    y_rf_test_proba_mhc = rf_model_mhc.predict_proba(X_test_mhc)[:, 1]
    df_test_rf_mhc = pd.DataFrame({'predict_proba': y_rf_test_proba_mhc})
    df_prob_test_rf_mhc = pd.concat([test_data.reset_index(drop=True), df_test_rf_mhc], axis=1)
    df_prob_test_rf_mhc['binder_pred'] = (df_prob_test_rf_mhc['predict_proba'] >= 0.5).astype(int)
    df_prob_test_rf_mhc.to_csv(args.outfile, index=False)

print('Done!')
