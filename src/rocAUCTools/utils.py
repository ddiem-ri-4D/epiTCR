import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from sklearn.metrics import classification_report, confusion_matrix, plot_roc_curve
from sklearn.metrics import classification_report

# import src.modules.processor as Processor
# import src.modules.model as Model

def confusionMatrix(y_true, y_pred):
    target_names = ['Non-binder', 'Binder']
    print(classification_report(y_true, y_pred, target_names=target_names))
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred), index=target_names, columns=target_names)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='g')
    plt.title('Confusion matrix')
    plt.xlabel('Predicted values')
    plt.ylabel('Actual values')
    plt.show()

def _rocAuc(y_true, y_score):
    y_pred01_proba = y_score.to_numpy()
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred01_proba)
    auc = metrics.roc_auc_score(y_true, y_pred01_proba)
    plt.plot(fpr,tpr,label="AUC = "+str(auc))
    print ("AUC : ", auc)
    plt.legend(loc=4)
    plt.show()

def rocAuc(y_true, y_score):
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    auc = metrics.roc_auc_score(y_true, y_score)
    plt.plot(fpr,tpr,label="AUC="+str(auc))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random guess')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.grid()
    plt.show()

def cal_fpr_and_tpr_with_one_cutoff(y_true, y_prred, cutoff):
    tp = np.where(np.logical_and(y_prred >= cutoff, y_true == 1))[0]
    fp = np.where(np.logical_and(y_prred >= cutoff, y_true == 0))[0]
    tpr = len(tp)/len(y_true[y_true==1])
    fpr = len(fp)/len(y_true[y_true==0])
    return fpr,tpr
def cal_fpr_and_tpr_in_different_cutoffs(y_true, y_prred, cutoffs):
    fpr_in_different_cutoffs = []
    tpr_in_different_cutoffs = []
    for cutoff in cutoffs:
        fpr, tpr = cal_fpr_and_tpr_with_one_cutoff(y_true, y_prred, cutoff)
        fpr_in_different_cutoffs.append(fpr)
        tpr_in_different_cutoffs.append(tpr)
    return np.array(fpr_in_different_cutoffs), np.array(tpr_in_different_cutoffs)
def cal_fprs_and_tprs_in_different_sets(y_trues, y_preds, cutoffs):
    fprs_in_different_sets = []
    tprs_in_different_sets = []
    for y_true, y_pred in zip(y_trues, y_preds):
        fprs_in_a_set, tprs_in_a_set = cal_fpr_and_tpr_in_different_cutoffs(y_true, y_pred, cutoffs)
        fprs_in_different_sets.append(fprs_in_a_set)
        tprs_in_different_sets.append(tprs_in_a_set)
    fpr_mean = np.mean(fprs_in_different_sets, axis=0)
    tpr_mean = np.mean(tprs_in_different_sets, axis=0)
    tpr_std= np.std(tprs_in_different_sets, axis=0)
    return fprs_in_different_sets, tprs_in_different_sets