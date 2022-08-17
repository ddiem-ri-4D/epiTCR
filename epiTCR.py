import numpy as np
import os, sys
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import classification_report,roc_curve,auc, f1_score, plot_roc_curve
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
import warnings

from tqdm import tqdm
from IPython.display import display
import src.modules.processor as Processor
import src.modules.utils as Utils
# import src.modules.model as Model
import src.modules.plot as Plot

from argparse import ArgumentParser

class epitcrModel:
    def __init__(self, pmodel, pX, py):
        self.model = pmodel
        self.model.fit(pX, py)
    
    def predict(self, pnew_data):
        yhat_class = self.model.predict(pnew_data)
        return yhat_class 

    def predict_proba(self, pnew_data):
        yhat_class = self.model.predict_proba(pnew_data)
        return yhat_class 

#Args parse
parser = ArgumentParser(description="Specifying Input Parameters")
parser.add_argument("-tr", "--trainfile", help="Specify the full path of the training file with TCR sequences")
parser.add_argument("-te", "--testfile", help="Specify the full path of the file with TCR sequences")
parser.add_argument("-c", "--chain", default="nm", help="Specify the chain(s) to use (nm, m). Default: nm")
parser.add_argument("-o", "--outfile", default=sys.stdout, help="Specify output file")
args = parser.parse_args()

chain = args.chain

print('Loading and encoding the dataset..')

if chain not in ["nm","m"]:
    print("Invalid chain. You can select nm (without mhc), m (with mhc)")

train = pd.read_csv(args.trainfile)
test = pd.read_csv(args.testfile)

clf_sm = RandomUnderSampler(random_state=42)

lst_models = [ ('Random Forest', RandomForestClassifier(bootstrap=False, max_depth=90, max_features=5,
                         n_estimators=400, n_jobs=-1, random_state=42))]
lst_models_mhc = [ ('Random Forest', RandomForestClassifier(bootstrap=False, max_depth=90, max_features='auto',
                         min_samples_split=10, n_estimators=600, n_jobs=-1, random_state=42))]

if(chain=='nm'):
    X_train = train.iloc[:, :2]
    y_train = train.iloc[:, 2:]

    X_test = test.iloc[:, :2]
    y_test = test.iloc[:, 2:]

    X_res, y_res = clf_sm.fit_resample(X_train, y_train)
    pX_res = Processor.data_representation(X_res)
    py_res = y_res.copy()

    pX_test = Processor.data_representation(X_test)
    py_test = y_test.copy()

    print('Training..')

    rf_tcr = lst_models[0][1]
    model_rf = epitcrModel(rf_tcr, pX_res, np.ravel(py_res))

    print('Evaluating..')

    y_rf_test_proba = model_rf.predict_proba(pX_test)
    df_test = pd.DataFrame(data = y_rf_test_proba, columns = ["tmp", "predict_proba"])

    df_test = df_test.iloc[:, 1:]

    df_prob_test = pd.concat([test, df_test], axis=1)
    df_prob_test.to_csv('output_prediction.csv', index=False)
    # df_prob_test.to_csv(args.outfile, index=False)
    print('Done!')


elif chain=="m":
    X_train_mhc = train.iloc[:, :3]
    y_train_mhc = train.iloc[:, 3:]

    X_test_mhc = test.iloc[:, :3]
    y_test_mhc = test.iloc[:, 3:]

    X_res_mhc, y_res_mhc = clf_sm.fit_resample(X_train_mhc, y_train_mhc)
    pX_res_mhc = Processor.data_representation_mhc(X_res_mhc)
    py_res_mhc = y_res_mhc.copy()

    pX_test_mhc = Processor.data_representation_mhc(X_test_mhc)
    py_test_mhc = y_test_mhc.copy()

    print('Training..')

    rf_tcr_mhc = lst_models_mhc[0][1]
    model_rf_mhc = epitcrModel(rf_tcr_mhc, pX_res_mhc, np.ravel(py_res_mhc))

    print('Evaluating..')

    y_rf_test_proba_mhc = model_rf_mhc.predict_proba(pX_test_mhc)
    df_test_mhc = pd.DataFrame(data = y_rf_test_proba_mhc, columns = ["tmp", "predict_proba"])

    df_test_mhc = df_test_mhc.iloc[:, 1:]

    df_prob_test_mhc = pd.concat([test, df_test_mhc], axis=1)
    df_prob_test_mhc.to_csv('output_prediction.csv', index=False)
    # df_prob_test_mhc.to_csv(args.outfile, index=False)
    print('Done!')