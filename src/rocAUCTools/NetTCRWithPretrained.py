import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score

import utils as Utils

test01_pred = pd.read_csv("../../data/predToolsData/NetTCR/Pretrained/test01_pred.csv")
test02_pred = pd.read_csv("../../data/predToolsData/NetTCR/Pretrained/test02_pred.csv")
test03_pred = pd.read_csv("../../data/predToolsData/NetTCR/Pretrained/test03_pred.csv")
test04_pred = pd.read_csv("../../data/predToolsData/NetTCR/Pretrained/test04_pred.csv")
test05_pred = pd.read_csv("../../data/predToolsData/NetTCR/Pretrained/test05_pred.csv")
test06_pred = pd.read_csv("../../data/predToolsData/NetTCR/Pretrained/test06_pred.csv")
test07_pred = pd.read_csv("../../data/predToolsData/NetTCR/Pretrained/test07_pred.csv")
test08_pred = pd.read_csv("../../data/predToolsData/NetTCR/Pretrained/test08_pred.csv")
test09_pred = pd.read_csv("../../data/predToolsData/NetTCR/Pretrained/test09_pred.csv")
test10_pred = pd.read_csv("../../data/predToolsData/NetTCR/Pretrained/test10_pred.csv")
test11_pred = pd.read_csv("../../data/predToolsData/NetTCR/Pretrained/test11_pred.csv")
test12_pred = pd.read_csv("../../data/predToolsData/NetTCR/Pretrained/test12_pred.csv")
test13_pred = pd.read_csv("../../data/predToolsData/NetTCR/Pretrained/test13_pred.csv")
test14_pred = pd.read_csv("../../data/predToolsData/NetTCR/Pretrained/test14_pred.csv")
test15_pred = pd.read_csv("../../data/predToolsData/NetTCR/Pretrained/test15_pred.csv")

y_test01 = test01_pred["binder"].to_numpy()
y_test02 = test02_pred["binder"].to_numpy()
y_test03 = test03_pred["binder"].to_numpy()
y_test04 = test04_pred["binder"].to_numpy()
y_test05 = test05_pred["binder"].to_numpy()
y_test06 = test06_pred["binder"].to_numpy()
y_test07 = test07_pred["binder"].to_numpy()
y_test08 = test08_pred["binder"].to_numpy()
y_test09 = test09_pred["binder"].to_numpy()
y_test10 = test10_pred["binder"].to_numpy()
y_test11 = test11_pred["binder"].to_numpy()
y_test12 = test12_pred["binder"].to_numpy()
y_test13 = test13_pred["binder"].to_numpy()
y_test14 = test14_pred["binder"].to_numpy()
y_test15 = test15_pred["binder"].to_numpy()

y_test01_pred = test01_pred["binder_pred"].to_numpy()
y_test02_pred = test02_pred["binder_pred"].to_numpy()
y_test03_pred = test03_pred["binder_pred"].to_numpy()
y_test04_pred = test04_pred["binder_pred"].to_numpy()
y_test05_pred = test05_pred["binder_pred"].to_numpy()
y_test06_pred = test06_pred["binder_pred"].to_numpy()
y_test07_pred = test07_pred["binder_pred"].to_numpy()
y_test08_pred = test08_pred["binder_pred"].to_numpy()
y_test09_pred = test09_pred["binder_pred"].to_numpy()
y_test10_pred = test10_pred["binder_pred"].to_numpy()
y_test11_pred = test11_pred["binder_pred"].to_numpy()
y_test12_pred = test12_pred["binder_pred"].to_numpy()
y_test13_pred = test13_pred["binder_pred"].to_numpy()
y_test14_pred = test14_pred["binder_pred"].to_numpy()
y_test15_pred = test15_pred["binder_pred"].to_numpy()

tn01, fp01, fn01, tp01 = Utils.confusion_matrix(y_test01, y_test01_pred).ravel()
tn02, fp02, fn02, tp02 = Utils.confusion_matrix(y_test02, y_test02_pred).ravel()
tn03, fp03, fn03, tp03 = Utils.confusion_matrix(y_test03, y_test03_pred).ravel()
tn04, fp04, fn04, tp04 = Utils.confusion_matrix(y_test04, y_test04_pred).ravel()						 
tn05, fp05, fn05, tp05 = Utils.confusion_matrix(y_test05, y_test05_pred).ravel()
tn06, fp06, fn06, tp06 = Utils.confusion_matrix(y_test06, y_test06_pred).ravel()
tn07, fp07, fn07, tp07 = Utils.confusion_matrix(y_test07, y_test07_pred).ravel()
tn08, fp08, fn08, tp08 = Utils.confusion_matrix(y_test08, y_test08_pred).ravel()
tn09, fp09, fn09, tp09 = Utils.confusion_matrix(y_test09, y_test09_pred).ravel()
tn10, fp10, fn10, tp10 = Utils.confusion_matrix(y_test10, y_test10_pred).ravel()
tn11, fp11, fn11, tp11 = Utils.confusion_matrix(y_test11, y_test11_pred).ravel()
tn12, fp12, fn12, tp12 = Utils.confusion_matrix(y_test12, y_test12_pred).ravel()
tn13, fp13, fn13, tp13 = Utils.confusion_matrix(y_test13, y_test13_pred).ravel()
tn14, fp14, fn14, tp14 = Utils.confusion_matrix(y_test14, y_test14_pred).ravel()
tn15, fp15, fn15, tp15 = Utils.confusion_matrix(y_test15, y_test15_pred).ravel()

accuracy01 = float(accuracy_score(y_test01, y_test01_pred).ravel())
accuracy02 = float(accuracy_score(y_test02, y_test02_pred).ravel())
accuracy03 = float(accuracy_score(y_test03, y_test03_pred).ravel())
accuracy04 = float(accuracy_score(y_test04, y_test04_pred).ravel())
accuracy05 = float(accuracy_score(y_test05, y_test05_pred).ravel())
accuracy06 = float(accuracy_score(y_test06, y_test06_pred).ravel())
accuracy07 = float(accuracy_score(y_test07, y_test07_pred).ravel())
accuracy08 = float(accuracy_score(y_test08, y_test08_pred).ravel())
accuracy09 = float(accuracy_score(y_test09, y_test09_pred).ravel())
accuracy10 = float(accuracy_score(y_test10, y_test10_pred).ravel())
accuracy11 = float(accuracy_score(y_test11, y_test11_pred).ravel())
accuracy12 = float(accuracy_score(y_test12, y_test12_pred).ravel())
accuracy13 = float(accuracy_score(y_test13, y_test13_pred).ravel())
accuracy14 = float(accuracy_score(y_test14, y_test14_pred).ravel())
accuracy15 = float(accuracy_score(y_test15, y_test15_pred).ravel())

sensitivity01 = tp01/(tp01+fn01)
sensitivity02 = tp02/(tp02+fn02)
sensitivity03 = tp03/(tp03+fn03)
sensitivity04 = tp04/(tp04+fn04)
sensitivity05 = tp05/(tp05+fn05)
sensitivity06 = tp06/(tp06+fn06)
sensitivity07 = tp07/(tp07+fn07)
sensitivity08 = tp08/(tp08+fn08)
sensitivity09 = tp09/(tp09+fn09)
sensitivity10 = tp10/(tp10+fn10)
sensitivity11 = tp11/(tp11+fn11)
sensitivity12 = tp12/(tp12+fn12)
sensitivity13 = tp13/(tp13+fn13)
sensitivity14 = tp14/(tp14+fn14)
sensitivity15 = tp15/(tp15+fn15)

specificity01 = tn01/(tn01+fp01)
specificity02 = tn02/(tn02+fp02)
specificity03 = tn03/(tn03+fp03)
specificity04 = tn04/(tn04+fp04)
specificity05 = tn05/(tn05+fp05)
specificity06 = tn06/(tn06+fp06)
specificity07 = tn07/(tn07+fp07)
specificity08 = tn08/(tn08+fp08)
specificity09 = tn09/(tn09+fp09)
specificity10 = tn10/(tn10+fp10)
specificity11 = tn11/(tn11+fp11)
specificity12 = tn12/(tn12+fp12)
specificity13 = tn13/(tn13+fp13)
specificity14 = tn14/(tn14+fp14)
specificity15 = tn15/(tn15+fp15)

auc01 = metrics.roc_auc_score(y_test01, test01_pred["prediction"])
auc02 = metrics.roc_auc_score(y_test02, test02_pred["prediction"])
auc03 = metrics.roc_auc_score(y_test03, test03_pred["prediction"])
auc04 = metrics.roc_auc_score(y_test04, test04_pred["prediction"])
auc05 = metrics.roc_auc_score(y_test05, test05_pred["prediction"])
auc06 = metrics.roc_auc_score(y_test06, test06_pred["prediction"])
auc07 = metrics.roc_auc_score(y_test07, test07_pred["prediction"])
auc08 = metrics.roc_auc_score(y_test08, test08_pred["prediction"])
auc09 = metrics.roc_auc_score(y_test09, test09_pred["prediction"])
auc10 = metrics.roc_auc_score(y_test10, test10_pred["prediction"])
auc11 = metrics.roc_auc_score(y_test11, test11_pred["prediction"])
auc12 = metrics.roc_auc_score(y_test12, test12_pred["prediction"])
auc13 = metrics.roc_auc_score(y_test13, test13_pred["prediction"])
auc14 = metrics.roc_auc_score(y_test14, test14_pred["prediction"])
auc15 = metrics.roc_auc_score(y_test15, test15_pred["prediction"])

data = {'acc': [accuracy01, accuracy02, accuracy03, accuracy04, accuracy05,
                accuracy06,accuracy07,accuracy08,accuracy09,accuracy10,
                accuracy11,accuracy12,accuracy13,accuracy14,accuracy15], 
        'sens': [sensitivity01, sensitivity02,sensitivity03, sensitivity04,sensitivity05,
                 sensitivity06,sensitivity07,sensitivity08,sensitivity09,sensitivity10,
                 sensitivity11,sensitivity12,sensitivity13,sensitivity14,sensitivity15],
        'spec': [specificity01, specificity02,specificity03, specificity04,specificity05,
                 specificity06,specificity07,specificity08,specificity09,specificity10,
                 specificity11,specificity12,specificity13,specificity14,specificity15], 
        'auc': [auc01, auc02,auc03,auc04,auc05,
                auc06,auc07,auc08,auc09,auc10,
                auc11,auc12,auc13,auc14,auc15]}
df = pd.DataFrame(data=data)

df.to_csv("../../data/outputPerformance/NetTCR/Pretrained/outputPerformance.csv", index=False)

prob01 = test01_pred["prediction"]
prob02 = test02_pred["prediction"]
prob03 = test03_pred["prediction"]
prob04 = test04_pred["prediction"]
prob05 = test05_pred["prediction"]
prob06 = test06_pred["prediction"]
prob07 = test07_pred["prediction"]
prob08 = test08_pred["prediction"]
prob09 = test09_pred["prediction"]
prob10 = test10_pred["prediction"]
prob11 = test11_pred["prediction"]
prob12 = test12_pred["prediction"]
prob13 = test13_pred["prediction"]
prob14 = test14_pred["prediction"]
prob15 = test15_pred["prediction"]

y_test01 = test01_pred["binder"].to_numpy()
y_test02 = test02_pred["binder"].to_numpy()
y_test03 = test03_pred["binder"].to_numpy()
y_test04 = test04_pred["binder"].to_numpy()
y_test05 = test05_pred["binder"].to_numpy()
y_test06 = test06_pred["binder"].to_numpy()
y_test07 = test07_pred["binder"].to_numpy()
y_test08 = test08_pred["binder"].to_numpy()
y_test09 = test09_pred["binder"].to_numpy()
y_test10 = test10_pred["binder"].to_numpy()
y_test11 = test11_pred["binder"].to_numpy()
y_test12 = test12_pred["binder"].to_numpy()
y_test13 = test13_pred["binder"].to_numpy()
y_test14 = test14_pred["binder"].to_numpy()
y_test15 = test15_pred["binder"].to_numpy()


fpr01, tpr01, thresholds = roc_curve(y_test01, prob01, drop_intermediate=False)
fpr02, tpr02, thresholds = roc_curve(y_test02, prob02, drop_intermediate=False)
fpr03, tpr03, thresholds = roc_curve(y_test03, prob03, drop_intermediate=False)
fpr04, tpr04, thresholds = roc_curve(y_test04, prob04, drop_intermediate=False)
fpr05, tpr05, thresholds = roc_curve(y_test05, prob05, drop_intermediate=False)
fpr06, tpr06, thresholds = roc_curve(y_test06, prob06, drop_intermediate=False)
fpr07, tpr07, thresholds = roc_curve(y_test07, prob07, drop_intermediate=False)
fpr08, tpr08, thresholds = roc_curve(y_test08, prob08, drop_intermediate=False)
fpr09, tpr09, thresholds = roc_curve(y_test09, prob09, drop_intermediate=False)
fpr10, tpr10, thresholds = roc_curve(y_test10, prob10, drop_intermediate=False)
fpr11, tpr11, thresholds = roc_curve(y_test11, prob11, drop_intermediate=False)
fpr12, tpr12, thresholds = roc_curve(y_test12, prob12, drop_intermediate=False)
fpr13, tpr13, thresholds = roc_curve(y_test13, prob13, drop_intermediate=False)
fpr14, tpr14, thresholds = roc_curve(y_test14, prob14, drop_intermediate=False)
fpr15, tpr15, thresholds = roc_curve(y_test15, prob15, drop_intermediate=False)

auc_score01 = auc(fpr01, tpr01)
auc_score02 = auc(fpr02, tpr02)
auc_score03 = auc(fpr03, tpr03)
auc_score04 = auc(fpr04, tpr04)
auc_score05 = auc(fpr05, tpr05)
auc_score06 = auc(fpr06, tpr06)
auc_score07 = auc(fpr07, tpr07)
auc_score08 = auc(fpr08, tpr08)
auc_score09 = auc(fpr09, tpr09)
auc_score10 = auc(fpr10, tpr10)
auc_score11 = auc(fpr11, tpr11)
auc_score12 = auc(fpr12, tpr12)
auc_score13 = auc(fpr13, tpr13)
auc_score14 = auc(fpr14, tpr14)
auc_score15 = auc(fpr15, tpr15)

aucs = ([auc_score01, auc_score02, auc_score03, auc_score04, auc_score05, 
         auc_score06, auc_score07, auc_score08, auc_score09, auc_score10, 
         auc_score11, auc_score12, auc_score13, auc_score14, auc_score15 ])

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
def draw_ROC_Curve(y_trues, y_preds, cutoffs):
    fprs_in_different_sets, tprs_in_different_sets = cal_fprs_and_tprs_in_different_sets(y_trues, y_preds, cutoffs)
    fpr_mean= np.mean(fprs_in_different_sets, axis=0)
    tpr_mean= np.mean(tprs_in_different_sets, axis=0)
    tpr_mean[-1] = 0.0
    tpr_std= np.std(tprs_in_different_sets, axis=0)
    tpr_upper= np.minimum(tpr_mean + tpr_std, 1)
    tpr_lower= np.maximum(tpr_mean - tpr_std, 0)
    fig, ax = plt.subplots(figsize=(12, 11))
    mean_auc = auc(fpr_mean, tpr_mean)
    std_auc = np.std(aucs)
    
    ax.plot(fpr01, tpr01, label = 'NetTCR - test01 (AUC = {0})'.format(round(auc01,3)),linewidth=2)
    ax.plot(fpr02, tpr02, label = 'NetTCR - test02 (AUC = {0})'.format(round(auc02,3)),linewidth=2)
    ax.plot(fpr03, tpr03, label = 'NetTCR - test03 (AUC = {0})'.format(round(auc03,3)),linewidth=2)
    ax.plot(fpr04, tpr04, label = 'NetTCR - test04 (AUC = {0})'.format(round(auc04,3)),linewidth=2)
    ax.plot(fpr05, tpr05, label = 'NetTCR - test05 (AUC = {0})'.format(round(auc05,3)),linewidth=2)
    ax.plot(fpr06, tpr06, label = 'NetTCR - test06 (AUC = {0})'.format(round(auc06,3)),linewidth=2)
    ax.plot(fpr07, tpr07, label = 'NetTCR - test07 (AUC = {0})'.format(round(auc07,3)),linewidth=2)
    ax.plot(fpr08, tpr08, label = 'NetTCR - test08 (AUC = {0})'.format(round(auc08,3)),linewidth=2)
    ax.plot(fpr09, tpr09, label = 'NetTCR - test09 (AUC = {0})'.format(round(auc09,3)),linewidth=2)
    ax.plot(fpr10, tpr10, label = 'NetTCR - test10 (AUC = {0})'.format(round(auc10,3)),linewidth=2)
    ax.plot(fpr11, tpr11, label = 'NetTCR - test11 (AUC = {0})'.format(round(auc11,3)),linewidth=2)
    ax.plot(fpr12, tpr12, label = 'NetTCR - test12 (AUC = {0})'.format(round(auc12,3)),linewidth=2)
    ax.plot(fpr13, tpr13, label = 'NetTCR - test13 (AUC = {0})'.format(round(auc13,3)),linewidth=2)
    ax.plot(fpr14, tpr14, label = 'NetTCR - test14 (AUC = {0})'.format(round(auc14,3)),linewidth=2)
    ax.plot(fpr15, tpr15, label = 'NetTCR - test15 (AUC = {0})'.format(round(auc15,3)),linewidth=2)    
    
    ax.plot([0, 1], [0, 1], linestyle="dashed", lw=1, color="k", label="Random guess", alpha=0.8)
    ax.plot(
        fpr_mean,
        tpr_mean,
        color="b",
        lw=0.5,
        label=r"Mean ROC (AUC = %0.3f $\pm$ %0.3f)" % (mean_auc, std_auc),
        alpha=0.8,
    )
    ax.fill_between(
        fpr_mean,
        tpr_lower,
        tpr_upper,
        color="grey",
        alpha=0.4,
        label=r"$\pm$ 1 std. dev.",
    )
    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
    )
    ax.legend(loc="lower right")
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.savefig("../../analysis/figures/benchmarkNetTCRPretrained.png", dpi=600)
    plt.savefig("../../analysis/figures/benchmarkNetTCRPretrained.pdf", dpi=600)
    plt.rcParams.update({'font.size': 11})
    plt.show()
    
draw_ROC_Curve([y_test01,y_test02,y_test03,y_test04,y_test05,
                y_test06,y_test07,y_test08,y_test09,y_test10,
                y_test11,y_test12,y_test13,y_test14,y_test15],
               [prob01,prob02,prob03,prob04,prob05,
                prob06,prob07,prob08,prob09,prob10,
                prob11,prob12,prob13,prob14,prob15],np.linspace(0.0, 1.0, num=100))