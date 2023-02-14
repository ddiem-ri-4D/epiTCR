import numpy as np
import sys
import time
import pickle
import pandas as pd
from sklearn import svm
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sklearn.metrics as metrics

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.metrics import classification_report,roc_curve,auc, f1_score, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,roc_auc_score,classification_report,roc_curve,auc, f1_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import plotly.graph_objects as go

from imblearn.under_sampling import RandomUnderSampler
from sklearn.svm import SVC

import modules.processor as Processor
import modules.model as Model
import modules.model as Model

from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

from tqdm import tqdm
from IPython.display import display

####--------------    --------------####


df = pd.read_csv("../../data/finalData/finalWithoutHLA.csv")
train = pd.read_csv("../../data/splitData/withoutMHC/train/train.csv")

tmp = df["epitope"].value_counts()
tmp2 = tmp.to_frame(name="count")
values = ["GLCTLVAML", "NLVPMVATV", "GILGFVFTL", "TPRVTGGGAM", "ELAGIGILTV", "AVFDRKSDAK", "KLGGALQAK"]

values_GLCTLVAML=["GLCTLVAML"]
values_NLVPMVATV=["NLVPMVATV"]
values_GILGFVFTL=["GILGFVFTL"]
values_TPRVTGGGAM=["TPRVTGGGAM"]
values_ELAGIGILTV=["ELAGIGILTV"]
values_AVFDRKSDAK=["AVFDRKSDAK"]
values_KLGGALQAK=["KLGGALQAK"]

data_GLCTLVAML = df[df["epitope"].isin(values_GLCTLVAML)]
data_GLCTLVAML = data_GLCTLVAML.reset_index(drop=True)

data_NLVPMVATV = df[df["epitope"].isin(values_NLVPMVATV)]
data_NLVPMVATV = data_NLVPMVATV.reset_index(drop=True)

data_GILGFVFTL = df[df["epitope"].isin(values_GILGFVFTL)]
data_GILGFVFTL = data_GILGFVFTL.reset_index(drop=True)

data_TPRVTGGGAM = df[df["epitope"].isin(values_TPRVTGGGAM)]
data_TPRVTGGGAM = data_TPRVTGGGAM.reset_index(drop=True)

data_ELAGIGILTV = df[df["epitope"].isin(values_ELAGIGILTV)]
data_ELAGIGILTV = data_ELAGIGILTV.reset_index(drop=True)

data_AVFDRKSDAK = df[df["epitope"].isin(values_AVFDRKSDAK)]
data_AVFDRKSDAK = data_AVFDRKSDAK.reset_index(drop=True)

data_KLGGALQAK = df[df["epitope"].isin(values_KLGGALQAK)]
data_KLGGALQAK = data_KLGGALQAK.reset_index(drop=True)

data_final = train[train["epitope"].isin(values)]
data_final = data_final.reset_index(drop=True)


####--------------    --------------####


X_GLCTLVAML   = data_GLCTLVAML.iloc[:, lambda data_GLCTLVAML: [0, 1]]
X_NLVPMVATV   = data_NLVPMVATV.iloc[:, lambda data_NLVPMVATV: [0, 1]]
X_GILGFVFTL   = data_GILGFVFTL.iloc[:, lambda data_GILGFVFTL: [0, 1]]
X_TPRVTGGGAM  = data_TPRVTGGGAM.iloc[:, lambda data_TPRVTGGGAM: [0, 1]]
X_ELAGIGILTV  = data_ELAGIGILTV.iloc[:, lambda data_ELAGIGILTV: [0, 1]]
X_AVFDRKSDAK   = data_AVFDRKSDAK.iloc[:, lambda data_AVFDRKSDAK: [0, 1]]
X_KLGGALQAK = data_KLGGALQAK.iloc[:, lambda data_KLGGALQAK: [0, 1]]

y_GLCTLVAML   = data_GLCTLVAML.iloc[:, lambda data_GLCTLVAML: [2]]
y_NLVPMVATV   = data_NLVPMVATV.iloc[:, lambda data_NLVPMVATV: [2]]
y_GILGFVFTL   = data_GILGFVFTL.iloc[:, lambda data_GILGFVFTL: [2]]
y_TPRVTGGGAM  = data_TPRVTGGGAM.iloc[:, lambda data_TPRVTGGGAM: [2]]
y_ELAGIGILTV  = data_ELAGIGILTV.iloc[:, lambda data_ELAGIGILTV: [2]]
y_AVFDRKSDAK   = data_AVFDRKSDAK.iloc[:, lambda data_AVFDRKSDAK: [2]]
y_KLGGALQAK = data_KLGGALQAK.iloc[:, lambda data_KLGGALQAK: [2]]

X_final = data_final.iloc[:, lambda data_final: [0, 1]]
y_final = data_final.iloc[:, lambda data_final: [2]]



####--------------    --------------####

X_train_GLCTLVAML  , X_test_GLCTLVAML  , y_train_GLCTLVAML  , y_test_GLCTLVAML   = train_test_split(X_GLCTLVAML  , y_GLCTLVAML  , test_size=0.2, random_state=42)
X_train_NLVPMVATV  , X_test_NLVPMVATV  , y_train_NLVPMVATV  , y_test_NLVPMVATV   = train_test_split(X_NLVPMVATV  , y_NLVPMVATV  , test_size=0.2, random_state=42)
X_train_GILGFVFTL  , X_test_GILGFVFTL  , y_train_GILGFVFTL  , y_test_GILGFVFTL   = train_test_split(X_GILGFVFTL  , y_GILGFVFTL  , test_size=0.2, random_state=42)
X_train_TPRVTGGGAM , X_test_TPRVTGGGAM , y_train_TPRVTGGGAM , y_test_TPRVTGGGAM  = train_test_split(X_TPRVTGGGAM , y_TPRVTGGGAM , test_size=0.2, random_state=42)
X_train_ELAGIGILTV , X_test_ELAGIGILTV , y_train_ELAGIGILTV , y_test_ELAGIGILTV  = train_test_split(X_ELAGIGILTV , y_ELAGIGILTV , test_size=0.2, random_state=42)
X_train_AVFDRKSDAK  , X_test_AVFDRKSDAK  , y_train_AVFDRKSDAK  , y_test_AVFDRKSDAK   = train_test_split(X_AVFDRKSDAK  , y_AVFDRKSDAK  , test_size=0.2, random_state=42)
X_train_KLGGALQAK, X_test_KLGGALQAK, y_train_KLGGALQAK, y_test_KLGGALQAK = train_test_split(X_KLGGALQAK, y_KLGGALQAK, test_size=0.2, random_state=42)
X_train_final  , X_test_final , y_train_final  , y_test_final   = train_test_split(X_final, y_final  , test_size=0.2, random_state=42)

pX_res_GLCTLVAML  , py_res_GLCTLVAML  = Processor.dataRepresentationDownsamplingWithoutMHCb(pd.concat([X_train_GLCTLVAML , y_train_GLCTLVAML ], axis=1))
pX_res_NLVPMVATV  , py_res_NLVPMVATV  = Processor.dataRepresentationDownsamplingWithoutMHCb(pd.concat([X_train_NLVPMVATV , y_train_NLVPMVATV ], axis=1))
pX_res_GILGFVFTL  , py_res_GILGFVFTL  = Processor.dataRepresentationDownsamplingWithoutMHCb(pd.concat([X_train_GILGFVFTL , y_train_GILGFVFTL ], axis=1))
pX_res_TPRVTGGGAM , py_res_TPRVTGGGAM = Processor.dataRepresentationDownsamplingWithoutMHCb(pd.concat([X_train_TPRVTGGGAM, y_train_TPRVTGGGAM], axis=1))
pX_res_ELAGIGILTV , py_res_ELAGIGILTV = Processor.dataRepresentationDownsamplingWithoutMHCb(pd.concat([X_train_ELAGIGILTV, y_train_ELAGIGILTV], axis=1))
pX_res_AVFDRKSDAK , py_res_AVFDRKSDAK = Processor.dataRepresentationDownsamplingWithoutMHCb( pd.concat([X_train_AVFDRKSDAK, y_train_AVFDRKSDAK], axis=1))
pX_res_KLGGALQAK  , py_res_KLGGALQAK  = Processor.dataRepresentationDownsamplingWithoutMHCb( pd.concat([X_train_KLGGALQAK , y_train_KLGGALQAK ], axis=1))
pX_res_final      , py_res_final      = Processor.dataRepresentationDownsamplingWithoutMHCb(      pd.concat([X_train_final     , y_train_final     ], axis=1))

pX_test_GLCTLVAML   = Processor.dataRepresentationBlosum62WithoutMHCb(X_test_GLCTLVAML)
pX_test_NLVPMVATV   = Processor.dataRepresentationBlosum62WithoutMHCb(X_test_NLVPMVATV)
pX_test_GILGFVFTL   = Processor.dataRepresentationBlosum62WithoutMHCb(X_test_GILGFVFTL)
pX_test_TPRVTGGGAM  = Processor.dataRepresentationBlosum62WithoutMHCb(X_test_TPRVTGGGAM)
pX_test_ELAGIGILTV  = Processor.dataRepresentationBlosum62WithoutMHCb(X_test_ELAGIGILTV)
pX_test_AVFDRKSDAK = Processor.dataRepresentationBlosum62WithoutMHCb(X_test_AVFDRKSDAK)
pX_test_KLGGALQAK  = Processor.dataRepresentationBlosum62WithoutMHCb(X_test_KLGGALQAK )
pX_test_final      = Processor.dataRepresentationBlosum62WithoutMHCb(X_test_final)

py_test_GLCTLVAML   = y_test_GLCTLVAML  .copy()
py_test_NLVPMVATV   = y_test_NLVPMVATV  .copy()
py_test_GILGFVFTL   = y_test_GILGFVFTL  .copy()
py_test_TPRVTGGGAM  = y_test_TPRVTGGGAM .copy()
py_test_ELAGIGILTV  = y_test_ELAGIGILTV .copy()
py_test_AVFDRKSDAK= y_test_AVFDRKSDAK.copy()
py_test_KLGGALQAK = y_test_KLGGALQAK .copy()
py_test_final   = y_test_final.copy()


####--------------    --------------####


lst_models = [
    ('Random Forest', RandomForestClassifier(bootstrap=False, max_features=15, n_estimators=300, n_jobs=-1, random_state=42)),
]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


####--------------    --------------####


model_GLCTLVAML  = lst_models[0][1].fit(pX_res_GLCTLVAML , np.ravel(py_res_GLCTLVAML ))
model_NLVPMVATV  = lst_models[0][1].fit(pX_res_NLVPMVATV , np.ravel(py_res_NLVPMVATV ))
model_GILGFVFTL  = lst_models[0][1].fit(pX_res_GILGFVFTL , np.ravel(py_res_GILGFVFTL ))
model_TPRVTGGGAM = lst_models[0][1].fit(pX_res_TPRVTGGGAM, np.ravel(py_res_TPRVTGGGAM))
model_ELAGIGILTV = lst_models[0][1].fit(pX_res_ELAGIGILTV, np.ravel(py_res_ELAGIGILTV))
model_AVFDRKSDAK = lst_models[0][1].fit(pX_res_AVFDRKSDAK, np.ravel(py_res_AVFDRKSDAK))
model_KLGGALQAK  = lst_models[0][1].fit(pX_res_KLGGALQAK , np.ravel(py_res_KLGGALQAK ))
model_final      = lst_models[0][1].fit(pX_res_final     , np.ravel(py_res_final     ))

auc_GLCTLVAML , acc_GLCTLVAML , sens_GLCTLVAML , spec_GLCTLVAML   = Model.predicMLModel(model_GLCTLVAML , X_test_GLCTLVAML .reset_index(drop=True), pX_test_GLCTLVAML .reset_index(drop=True), py_test_GLCTLVAML .reset_index(drop=True), '../../data/pred7DominantPeptide/predictGLCTLVAML.csv')
auc_NLVPMVATV , acc_NLVPMVATV , sens_NLVPMVATV , spec_NLVPMVATV   = Model.predicMLModel(model_NLVPMVATV , X_test_NLVPMVATV .reset_index(drop=True), pX_test_NLVPMVATV .reset_index(drop=True), py_test_NLVPMVATV .reset_index(drop=True), '../../data/pred7DominantPeptide/predictNLVPMVATV.csv')
auc_GILGFVFTL , acc_GILGFVFTL , sens_GILGFVFTL , spec_GILGFVFTL   = Model.predicMLModel(model_GILGFVFTL , X_test_GILGFVFTL .reset_index(drop=True), pX_test_GILGFVFTL .reset_index(drop=True), py_test_GILGFVFTL .reset_index(drop=True), '../../data/pred7DominantPeptide/predictGILGFVFTL.csv')
auc_TPRVTGGGAM, acc_TPRVTGGGAM, sens_TPRVTGGGAM, spec_TPRVTGGGAM  = Model.predicMLModel(model_TPRVTGGGAM, X_test_TPRVTGGGAM.reset_index(drop=True), pX_test_TPRVTGGGAM.reset_index(drop=True), py_test_TPRVTGGGAM.reset_index(drop=True), '../../data/pred7DominantPeptide/predictTPRVTGGGAM.csv')
auc_ELAGIGILTV, acc_ELAGIGILTV, sens_ELAGIGILTV, spec_ELAGIGILTV  = Model.predicMLModel(model_ELAGIGILTV, X_test_ELAGIGILTV.reset_index(drop=True), pX_test_ELAGIGILTV.reset_index(drop=True), py_test_ELAGIGILTV.reset_index(drop=True), '../../data/pred7DominantPeptide/predictELAGIGILTV.csv')
auc_AVFDRKSDAK, acc_AVFDRKSDAK, sens_AVFDRKSDAK, spec_AVFDRKSDAK  = Model.predicMLModel(model_AVFDRKSDAK, X_test_AVFDRKSDAK.reset_index(drop=True), pX_test_AVFDRKSDAK.reset_index(drop=True), py_test_AVFDRKSDAK.reset_index(drop=True), '../../data/pred7DominantPeptide/predictAVFDRKSDAK.csv')
auc_KLGGALQAK , acc_KLGGALQAK , sens_KLGGALQAK , spec_KLGGALQAK   = Model.predicMLModel(model_KLGGALQAK , X_test_KLGGALQAK .reset_index(drop=True), pX_test_KLGGALQAK .reset_index(drop=True), py_test_KLGGALQAK .reset_index(drop=True), '../../data/pred7DominantPeptide/predictKLGGALQAK.csv')
auc_final     , acc_final     , sens_final     , spec_final       = Model.predicMLModel(model_final     , X_test_final     .reset_index(drop=True), pX_test_final     .reset_index(drop=True), py_test_final     .reset_index(drop=True), '../../data/pred7DominantPeptide/predictfinal.csv')

accuracy_rf_final, classify_metrics_rf_final, fpr_rf_final, tpr_rf_final, auc_score_rf_final, f1_rf_final                                     = Model.modelRun(model_final,       pX_res_final,       py_res_final,       pX_test_final, py_test_final)
accuracy_rf_GLCTLVAML  , classify_metrics_rf_GLCTLVAML  , fpr_rf_GLCTLVAML  , tpr_rf_GLCTLVAML  , auc_score_rf_GLCTLVAML  , f1_rf_GLCTLVAML   = Model.modelRun(model_GLCTLVAML  , pX_res_GLCTLVAML  , py_res_GLCTLVAML  , pX_test_GLCTLVAML  , py_test_GLCTLVAML  )
accuracy_rf_NLVPMVATV  , classify_metrics_rf_NLVPMVATV  , fpr_rf_NLVPMVATV  , tpr_rf_NLVPMVATV  , auc_score_rf_NLVPMVATV  , f1_rf_NLVPMVATV   = Model.modelRun(model_NLVPMVATV  , pX_res_NLVPMVATV  , py_res_NLVPMVATV  , pX_test_NLVPMVATV  , py_test_NLVPMVATV  )
accuracy_rf_GILGFVFTL  , classify_metrics_rf_GILGFVFTL  , fpr_rf_GILGFVFTL  , tpr_rf_GILGFVFTL  , auc_score_rf_GILGFVFTL  , f1_rf_GILGFVFTL   = Model.modelRun(model_GILGFVFTL  , pX_res_GILGFVFTL  , py_res_GILGFVFTL  , pX_test_GILGFVFTL  , py_test_GILGFVFTL  )
accuracy_rf_TPRVTGGGAM , classify_metrics_rf_TPRVTGGGAM , fpr_rf_TPRVTGGGAM , tpr_rf_TPRVTGGGAM , auc_score_rf_TPRVTGGGAM , f1_rf_TPRVTGGGAM  = Model.modelRun(model_TPRVTGGGAM , pX_res_TPRVTGGGAM , py_res_TPRVTGGGAM , pX_test_TPRVTGGGAM , py_test_TPRVTGGGAM )
accuracy_rf_ELAGIGILTV , classify_metrics_rf_ELAGIGILTV , fpr_rf_ELAGIGILTV , tpr_rf_ELAGIGILTV , auc_score_rf_ELAGIGILTV , f1_rf_ELAGIGILTV  = Model.modelRun(model_ELAGIGILTV , pX_res_ELAGIGILTV , py_res_ELAGIGILTV , pX_test_ELAGIGILTV , py_test_ELAGIGILTV )
accuracy_rf_AVFDRKSDAK, classify_metrics_rf_AVFDRKSDAK, fpr_rf_AVFDRKSDAK, tpr_rf_AVFDRKSDAK, auc_score_rf_AVFDRKSDAK, f1_rf_AVFDRKSDAK       = Model.modelRun(model_AVFDRKSDAK , pX_res_AVFDRKSDAK,  py_res_AVFDRKSDAK,  pX_test_AVFDRKSDAK, py_test_AVFDRKSDAK)
accuracy_rf_KLGGALQAK , classify_metrics_rf_KLGGALQAK , fpr_rf_KLGGALQAK , tpr_rf_KLGGALQAK , auc_score_rf_KLGGALQAK , f1_rf_KLGGALQAK        = Model.modelRun(model_KLGGALQAK ,  pX_res_KLGGALQAK ,  py_res_KLGGALQAK ,  pX_test_KLGGALQAK , py_test_KLGGALQAK )


####--------------    --------------####


#linestyle="dotted"
fig = plt.figure(figsize=(10,9))
ax  = fig.add_subplot(111)

ax.plot(fpr_rf_final, tpr_rf_final, label = 'MLibTCR for 7 peptides (AUC = {0})'.format(round(auc_score_rf_final,3)),linewidth=2)
ax.plot(fpr_rf_GLCTLVAML  , tpr_rf_GLCTLVAML  , label = 'GLCTLVAML  (AUC = {0})'.format(round(auc_score_rf_GLCTLVAML  ,3)),linewidth=2,)
ax.plot(fpr_rf_NLVPMVATV  , tpr_rf_NLVPMVATV  , label = 'NLVPMVATV  (AUC = {0})'.format(round(auc_score_rf_NLVPMVATV  ,3)),linewidth=2,)
ax.plot(fpr_rf_GILGFVFTL  , tpr_rf_GILGFVFTL  , label = 'GILGFVFTL  (AUC = {0})'.format(round(auc_score_rf_GILGFVFTL  ,3)),linewidth=2)
ax.plot(fpr_rf_TPRVTGGGAM , tpr_rf_TPRVTGGGAM , label = 'TPRVTGGGAM (AUC = {0})'.format(round(auc_score_rf_TPRVTGGGAM ,3)),linewidth=2)
ax.plot(fpr_rf_ELAGIGILTV , tpr_rf_ELAGIGILTV , label = 'ELAGIGILTV (AUC = {0})'.format(round(auc_score_rf_ELAGIGILTV ,3)),linewidth=2)
ax.plot(fpr_rf_AVFDRKSDAK, tpr_rf_AVFDRKSDAK , label =  'AVFDRKSDAK (AUC = {0})'.format(round(auc_score_rf_AVFDRKSDAK,3) ),linewidth=2)
ax.plot(fpr_rf_KLGGALQAK , tpr_rf_KLGGALQAK ,  label =  'KLGGALQAK  (AUC = {0})'.format(round(auc_score_rf_KLGGALQAK ,3) ),linewidth=2)

# ax.plot([0, 1], [0, 1], linestyle="dashed", lw=1, color="k", label="Random guess", alpha=0.8)

plt.legend(loc="best")
# plt.title("ROC-AUC for 7 highly false positive peptides", fontsize=11)
plt.xlabel("1 - Specificity", fontsize=12)
plt.ylabel("Sensitivity", fontsize=12)

plt.savefig("../../analysis/figures/ROCAUC7HighlyFPpeptides.png")
plt.savefig("../../analysis/figures/ROCAUC7HighlyFPpeptides.pdf")

plt.rcParams.update({'font.size': 12})

plt.show()