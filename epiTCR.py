import numpy as np
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
import src.modules.processor as Processor
import src.modules.model as Model
from argparse import ArgumentParser

# Args parse
parser = ArgumentParser(description="Specifying Input Parameters")
parser.add_argument("-tr", "--trainfile", help="Specify the full path of the training file with TCR sequences")
parser.add_argument("-te", "--testfile", help="Specify the full path of the file with TCR sequences")
parser.add_argument("-c", "--chain", default="ce", help="Specify the chain(s) to use (ce, cem). Default: ce")
parser.add_argument("-o", "--outfile", default=sys.stdout, help="Specify output file")

args = parser.parse_args()

chain = args.chain

print('Loading and encoding the dataset...')

if chain not in ["ce", "cem"]:
    print("Invalid chain. You can select ce (cdr3b+epitope), cem (cdr3b+epitope+mhc)")
assert chain in ["ce", "cem"]

train_data = pd.read_csv(args.trainfile)
test_data = pd.read_csv(args.testfile)

under_sampler = RandomUnderSampler(random_state=42)

models = [
    ('Random Forest - without MHC', RandomForestClassifier(bootstrap=False, max_features=15,
                                                           n_estimators=300, n_jobs=-1, random_state=42)),
    ('Random Forest - with MHC', RandomForestClassifier(max_features=20,
                                                        n_estimators=300, n_jobs=-1, random_state=42))
]

if chain == 'ce':
    X_train, y_train = Processor.dataRepresentationDownsamplingWithoutMHCb(train_data)
    X_test, y_test = Processor.dataRepresentationBlosum62WithoutMHCb(test_data), test_data[["binder"]]

    print('Training...')
    rf_model = models[0][1].fit(X_train, np.ravel(y_train))
    Model.saveByPickle(rf_model, "./models/rdforestWithoutMHCModel.pickle")

    print('Evaluating...')
    auc_test, acc_test, sens_test, spec_test = Model.predicMLModel(rf_model, test_data, X_test, y_test, args.outfile)

else:
    X_train_mhc, y_train_mhc = Processor.dataRepresentationDownsamplingWithMHCb(train_data)
    X_test_mhc, y_test_mhc = Processor.dataRepresentationBlosum62WithMHCb(test_data), test_data[["binder"]]

    print('Training...')
    rf_model_mhc = models[1][1].fit(X_train_mhc, np.ravel(y_train_mhc))
    Model.saveByPickle(rf_model_mhc, "./models/rdforestWithMHCModel.pickle")

    print('Evaluating...')
    auc_test, acc_test, sens_test, spec_test = Model.predicMLModel(rf_model_mhc, test_data, X_test_mhc, y_test_mhc, args.outfile)

print('Done!')
