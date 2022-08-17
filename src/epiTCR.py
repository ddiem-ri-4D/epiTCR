import numpy as np
import time
import os, sys
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import classification_report,roc_curve,auc, f1_score, plot_roc_curve
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

from tqdm import tqdm
from IPython.display import display
import modules.processor as Processor
import modules.utils as Utils
import modules.model as Model
import modules.plot as Plot

from argparse import ArgumentParser

class epitcrModel:
    def __init__(self, pmodel, pX, py):
        self.model = pmodel
        self.model.fit(pX, py)
    
    def predict(self, pnew_data):
        yhat_class = self.model.predict(pnew_data)
        return yhat_class 

    
    def info(self):
        print(self.model)
    
    def rocAuc(self, X, y_true):
        ax = plt.subplot()
        n = X.shape[0]
        mean_fpr = np.linspace(0, 1, n)
        tprs = []
        aucs = []
        viz = plot_roc_curve(self.model, X, y_true,
                                name='Random Forest',
                                alpha=.3, lw=2, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
        title="ROC curve comparison")
        ax.legend(loc="lower right")
        plt.show()
    
    def predict_proba(self, pnew_data):
        yhat_class = self.model.predict_proba(pnew_data)
        return yhat_class 

output_dir = '../data/convert-data/tmp'

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

    # pX_res = pd.read_csv("../data/convert-data/tmp/without-mhc/res/X.csv")
    # py_res = pd.read_csv("../data/convert-data/tmp/without-mhc/res/y.csv")

    # pX_test = pd.read_csv("../data/convert-data/tmp/without-mhc/test/X.csv")
    # py_test = pd.read_csv("../data/convert-data/tmp/without-mhc/test/y.csv")

    print('Training..')

    rf_tcr = lst_models[0][1]
    model_rf = epitcrModel(rf_tcr, pX_res, np.ravel(py_res))

    print('Evaluating..')

    y_rf_test_proba = model_rf.predict_proba(pX_test)
    df_test = pd.DataFrame(data = y_rf_test_proba, columns = ["tmp", "predict_proba"])

    df_test = df_test.iloc[:, 1:]

    df_prob_test = pd.concat([test, df_test], axis=1)
    # df_prob_test['binder_pred'] = np.where(df_prob_test['predict_proba'] >= 0.5, 1, 0)
    df_prob_test.to_csv('output_prediction.csv', index=False)
    print('Done!')


elif chain=="m":
    X_train_mhc = train.iloc[:, :3]
    y_train_mhc = train.iloc[:, 3:]

    X_test_mhc = test.iloc[:, :3]
    y_test_mhc = test.iloc[:, 3:]

    X_res_mhc, y_res_mhc = clf_sm.fit_resample(X_train_mhc, y_train_mhc)
    pX_res_mhc = Processor.data_representation_mhc(X_res_mhc)
    py_res_mhc = y_res.copy()

    pX_test_mhc = Processor.data_representation_mhc(X_test_mhc)
    py_test_mhc = y_test_mhc.copy()

    # pX_res_mhc = pd.read_csv("../data/convert-data/tmp/with-mhc/res/X.csv")
    # py_res_mhc = pd.read_csv("../data/convert-data/tmp/with-mhc/res/y.csv")

    # pX_test_mhc = pd.read_csv("../data/convert-data/tmp/with-mhc/test/X.csv")
    # py_test_mhc = pd.read_csv("../data/convert-data/tmp/with-mhc/test/y.csv")

    print('Training..')

    rf_tcr_mhc = lst_models_mhc[0][1]
    model_rf_mhc = epitcrModel(rf_tcr_mhc, pX_res_mhc, np.ravel(py_res_mhc))

    print('Evaluating..')

    y_rf_test_proba_mhc = model_rf_mhc.predict_proba(pX_test_mhc)
    df_test_mhc = pd.DataFrame(data = y_rf_test_proba_mhc, columns = ["tmp", "predict_proba"])

    df_test_mhc = df_test_mhc.iloc[:, 1:]

    df_prob_test_mhc = pd.concat([test, df_test_mhc], axis=1)
    # df_prob_test_mhc['binder_pred'] = np.where(df_prob_test_mhc['predict_proba'] >= 0.5, 1, 0)
    df_prob_test_mhc.to_csv('output_prediction.csv', index=False)
    print('Done!')