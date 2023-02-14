import numpy as np
import sys
import pickle
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler

from IPython.display import display
import src.modules.processor as Processor
import src.modules.model as Model

from argparse import ArgumentParser

#Args parse
#Nonoverlapping peptides Model

parser = ArgumentParser(description="Specifying Input Parameters")
parser.add_argument("-tr", "--trainfile", help="Specify the full path of the training file with TCR sequences")
parser.add_argument("-te", "--testfile", help="Specify the full path of the file with TCR sequences")
parser.add_argument("-c", "--chain", default="ce", help="Specify the chain(s) to use (ce, cem). Default: ce")
parser.add_argument("-o", "--outfile", default=sys.stdout, help="Specify output file")
# parser.add_argument("-sm", "--savemodel", help="Specify the full path of the file with save model")

args = parser.parse_args()

chain = args.chain

print('Loading and encoding the dataset..')

if chain not in ["ce","cem"]:
    print("Invalid chain. You can select ce (cdr3b+epitope), cem (cdr3b+epitope+mhc)")
assert chain in ["ce","cem"]

train = pd.read_csv(args.trainfile)
test = pd.read_csv(args.testfile)

clf_sm = RandomUnderSampler(random_state=42)

lst_models = [ ('Random Forest - without MHC', RandomForestClassifier(bootstrap=False, max_features=15,
                         n_estimators=300, n_jobs=-1, random_state=42)),
               ('Random Forest - with MHC', RandomForestClassifier(max_features=20,
                         n_estimators=300, n_jobs=-1, random_state=42))]

if chain=='ce':
    pX_train, py_train = Processor.dataRepresentationDownsamplingWithoutMHCb(train)
    pX_test, py_test = Processor.dataRepresentationBlosum62WithoutMHCb(test), test[["binder"]]

    print('Training..')
    model_rf = lst_models[0][1].fit(pX_train, np.ravel(py_train))
    Model.saveByPickle(model_rf, "./models/rdforestWithoutMHCNonOverlapingModel.pickle")
    # Model.saveByPickle(model_rf, args.savemodel)

    print('Evaluating..')
    auc_test, acc_test, sens_test, spec_test = Model.predicMLModel(model_rf, test, pX_test, py_test, args.outfile)

    print('Done!')

else:
    pX_train_mhc, py_train_mhc = Processor.dataRepresentationDownsamplingWithMHCb(train)
    pX_test_mhc, py_test_mhc = Processor.dataRepresentationBlosum62WithMHCb(test), test[["binder"]]

    print('Training..')
    model_rf_mhc = lst_models[1][1].fit(pX_train_mhc, np.ravel(py_train_mhc))
    Model.saveByPickle(model_rf_mhc, "./models/rdforestWithMHCNonOverlapingModel.pickle")
    # Model.saveByPickle(model_rf_mhc, args.savemodel)

    print('Evaluating..')
    auc_test, acc_test, sens_test, spec_test = Model.predicMLModel(model_rf_mhc, test, pX_test_mhc, py_test_mhc, args.outfile)

    print('Done!')
