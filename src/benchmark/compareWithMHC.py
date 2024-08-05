import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, plot_roc_curve, roc_auc_score

# Data
mlib_acc    = [0.89,0.88,0.89,0.89,0.88,0.89,0.88,0.89,0.88]
mlib_sens   = [0.94,0.95,0.94,0.94,0.94,0.94,0.95,0.94,0.94]
mlib_spec   = [0.88,0.88,0.88,0.88,0.89,0.88,0.88,0.88,0.87]
mlib_auc    = [0.97,0.97,0.97,0.97,0.97,0.97,0.97,0.97,0.97]

pmtnet_acc  = [0.86,0.86,0.86,0.86,0.86,0.86,0.86,0.86,0.86]
pmtnet_sens = [0.08,0.08,0.08,0.09,0.08,0.07,0.08,0.07,0.07]
pmtnet_spec = [0.94,0.94,0.94,0.94,0.94,0.94,0.94,0.94,0.94]
pmtnet_auc  = [0.51,0.52,0.52,0.52,0.52,0.52,0.52,0.51,0.51]

algo = ["Testset 01","Testset 02","Testset 03","Testset 04","Testset 05",
        "Testset 06","Testset 07","Testset 08","Testset 09"]

# DataFrames
def create_df(metric, mlib_values, pmtnet_values):
    df = pd.DataFrame({"algo": algo, "epiTCR": mlib_values, "pMTnet": pmtnet_values})
    return pd.melt(df, id_vars="algo")

df_acc = create_df("Accuracy", mlib_acc, pmtnet_acc)
df_sens = create_df("Sensitivity", mlib_sens, pmtnet_sens)
df_spec = create_df("Specificity", mlib_spec, pmtnet_spec)
df_auc = create_df("AUC", mlib_auc, pmtnet_auc)

# Plotting function
def plot_metrics(df, metric, ylim, save_name):
    plt.rcParams["figure.figsize"] = (16, 8)
    fig, ax = plt.subplots(1)
    splot = sns.barplot(data=df, x="algo", y="value", hue="variable")
    plt.legend(loc='best')

    for p in splot.patches:
        splot.annotate(format(p.get_height(), '.2f'), 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha='center', va='center', 
                       xytext=(0, 10), size=15,
                       textcoords='offset points')
    plt.ylim(ylim)
    ax.set_xticklabels([])
    plt.xlabel('Test sets', fontsize=20)
    plt.ylabel(metric, fontsize=20)
    plt.setp(ax.get_legend().get_texts(), fontsize='16')
    plt.setp(ax.get_legend().get_title(), fontsize='16')
    plt.savefig(save_name)
    plt.show()

# Generate plots
plot_metrics(df_acc, 'Accuracy', [0.0, 1.10], "../../analysis/figures/comparisonWithmhcAcc005.png")
plot_metrics(df_sens, 'Sensitivity', [0.0, 1.15], "../../analysis/figures/comparisonWithmhcSens005.png")
plot_metrics(df_spec, 'Specificity', [0.0, 1.19], "../../analysis/figures/comparisonWithmhcSpec005.png")
plot_metrics(df_auc, 'AUC', [0.0, 1.19], "../../analysis/figures/comparisonWithmhcAUC005.png")

# Repeat with another set of data for further comparison
mlib_acc_002 = [0.89,0.88,0.89,0.89,0.88,0.89,0.88,0.89,0.88]
mlib_sens_002 = [0.94,0.95,0.94,0.94,0.94,0.94,0.95,0.94,0.94]
mlib_spec_002 = [0.88,0.88,0.88,0.88,0.89,0.88,0.88,0.88,0.87]
mlib_auc_002 = [0.97,0.97,0.97,0.97,0.97,0.97,0.97,0.97,0.97]

pmtnet_acc_002 = [0.89,0.89,0.89,0.89,0.89,0.88,0.89,0.89,0.89]
pmtnet_sens_002 = [0.03,0.03,0.04,0.04,0.04,0.03,0.03,0.03,0.03]
pmtnet_spec_002 = [0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98]
pmtnet_auc_002 = [0.51,0.52,0.52,0.52,0.52,0.52,0.52,0.51,0.51]

# DataFrames for new data
df_acc_002 = create_df("Accuracy", mlib_acc_002, pmtnet_acc_002)
df_sens_002 = create_df("Sensitivity", mlib_sens_002, pmtnet_sens_002)
df_spec_002 = create_df("Specificity", mlib_spec_002, pmtnet_spec_002)
df_auc_002 = create_df("AUC", mlib_auc_002, pmtnet_auc_002)

# Generate plots for new data
plot_metrics(df_acc_002, 'Accuracy', [0.0, 1.10], "../../analysis/figures/comparisonWithmhcAcc002.png")
plot_metrics(df_sens_002, 'Sensitivity', [0.0, 1.15], "../../analysis/figures/comparisonWithmhcSens002.png")
plot_metrics(df_spec_002, 'Specificity', [0.0, 1.19], "../../analysis/figures/comparisonWithmhcSpec002.png")
plot_metrics(df_auc_002, 'AUC', [0.0, 1.19], "../../analysis/figures/comparisonWithmhcAUC002.png")
