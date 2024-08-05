import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, roc_auc_score
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# Load datasets
ATMTCR_old = pd.read_csv("../../data/setDataPredict10Subs/ATMTCR_predict.csv", delimiter='\t')
ATMTCR_old.columns = ['epitope', 'CDR3b', 'binder', 'binder_pred', 'prediction']
RF_old = pd.read_csv("../../data/setDataPredict10Subs/epiTCR_predict.csv")
NetTCR_old = pd.read_csv("../../data/setDataPredict10Subs/NetTCR_predict.csv")
NetTCR_old = NetTCR_old.rename(columns={'peptide': 'epitope'})

print(NetTCR_old.shape)
print(ATMTCR_old.shape)
print(RF_old.shape)

# Split datasets into positive and negative samples
def split_pos_neg(data, binder_col='binder'):
    pos = data[data[binder_col] == 1]
    neg = data[data[binder_col] == 0]
    return pos, neg

NetTCR_pos, NetTCR_neg = split_pos_neg(NetTCR_old)
ATMTCR_pos, ATMTCR_neg = split_pos_neg(ATMTCR_old)
RF_pos, RF_neg = split_pos_neg(RF_old)

# Create stratified folds
def create_folds(pos, neg, num_folds=10, pos_sample_size=10000, neg_sample_size=100000):
    pos_folds = [pos.sample(n=pos_sample_size, random_state=i) for i in range(1, num_folds + 1)]
    neg_folds = [neg.sample(n=neg_sample_size, random_state=1110 + i) for i in range(1, num_folds + 1)]
    folds = [pd.concat([pos_fold, neg_fold]) for pos_fold, neg_fold in zip(pos_folds, neg_folds)]
    return folds

RF_folds = create_folds(RF_pos, RF_neg)

# Merge folds with original datasets
def merge_folds_with_data(folds, data, on_cols=['CDR3b', 'epitope', 'binder'], prediction_col='binder_pred_y'):
    merged_folds = [fold.merge(data, how='inner', on=on_cols) for fold in folds]
    for fold in merged_folds:
        fold.rename(columns={prediction_col: 'binder_pred'}, inplace=True)
    return merged_folds

ATMTCR_folds_v2 = merge_folds_with_data(RF_folds, ATMTCR_old)
NetTCR_folds_v2 = merge_folds_with_data(RF_folds, NetTCR_old, prediction_col='prediction')

# Save merged folds
def save_folds(folds, path_template):
    for i, fold in enumerate(folds):
        fold.to_csv(path_template.format(i + 1), index=False)

save_folds(ATMTCR_folds_v2, "../../data/get10Subsampling/ATMTCR/ATMTCR_fold{}.csv")
save_folds(NetTCR_folds_v2, "../../data/get10Subsampling/NetTCR/NetTCR_fold{}.csv")
save_folds(RF_folds, "../../data/get10Subsampling/epiTCR/RF_fold{}.csv")

# ROC and AUC calculations
def calculate_roc_auc(folds, binder_col='binder', prediction_col='prediction'):
    fprs, tprs, aucs = [], [], []
    for fold in folds:
        fpr, tpr, _ = roc_curve(fold[binder_col], fold[prediction_col], drop_intermediate=False)
        auc_score = auc(fpr, tpr)
        fprs.append(fpr)
        tprs.append(tpr)
        aucs.append(auc_score)
    return fprs, tprs, aucs

fprs_rf, tprs_rf, aucs_rf = calculate_roc_auc(RF_folds, prediction_col='predict_proba')
fprs_nettcr, tprs_nettcr, aucs_nettcr = calculate_roc_auc(NetTCR_folds_v2)
fprs_atmtcr, tprs_atmtcr, aucs_atmtcr = calculate_roc_auc(ATMTCR_folds_v2)

# Plot ROC curves
def plot_roc_curves(fprs, tprs, aucs, label, color, ax):
    for fpr, tpr in zip(fprs, tprs):
        ax.plot(fpr, tpr, linewidth=1, color=color)
    mean_auc = np.mean(aucs)
    ax.plot(fprs[0], tprs[0], label=f'{label} - Mean ROC (AUC = {mean_auc:.2f})', linewidth=2, color=color)

fig, ax = plt.subplots(figsize=(7, 7))
plot_roc_curves(fprs_rf, tprs_rf, aucs_rf, 'epiTCR', 'r', ax)
plot_roc_curves(fprs_nettcr, tprs_nettcr, aucs_nettcr, 'NetTCR', 'purple', ax)
plot_roc_curves(fprs_atmtcr, tprs_atmtcr, aucs_atmtcr, 'ATMTCR', 'c', ax)

plt.legend(loc="best")
plt.xlabel("1 - Specificity")
plt.ylabel("Sensitivity")
plt.savefig("../../analysis/figures/benchmarkTools10Subs.png", dpi=600)
plt.savefig("../../analysis/figures/benchmarkTools10Subs.pdf", dpi=600)
plt.show()

# Calculate accuracy, sensitivity, specificity, and AUC for each fold
def calculate_metrics(folds, binder_col='binder', prediction_col='prediction'):
    metrics_list = []
    for fold in folds:
        y_true = fold[binder_col].to_numpy()
        y_pred = (fold[prediction_col] >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        accuracy = accuracy_score(y_true, y_pred)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        auc_score = roc_auc_score(y_true, fold[prediction_col])
        metrics_list.append([accuracy, sensitivity, specificity, auc_score])
    return metrics_list

metrics_rf = calculate_metrics(RF_folds, prediction_col='predict_proba')
metrics_nettcr = calculate_metrics(NetTCR_folds_v2)
metrics_atmtcr = calculate_metrics(ATMTCR_folds_v2)

# Prepare data for plotting
def prepare_plot_data(metrics, tool_name):
    data = {
        'time_samplings': [f'{i + 1}th sampling' for i in range(10)],
        f'{tool_name}_accuracy': [m[0] for m in metrics],
        f'{tool_name}_sensitivity': [m[1] for m in metrics],
        f'{tool_name}_specificity': [m[2] for m in metrics],
        f'{tool_name}_auc': [m[3] for m in metrics],
    }
    return pd.DataFrame(data)

df_rf = prepare_plot_data(metrics_rf, 'epiTCR')
df_nettcr = prepare_plot_data(metrics_nettcr, 'NetTCR')
df_atmtcr = prepare_plot_data(metrics_atmtcr, 'ATMTCR')

# Merge data for plotting
def merge_plot_data(dfs):
    df_merged = pd.concat(dfs, axis=1)
    df_merged = df_merged.loc[:, ~df_merged.columns.duplicated()]
    return df_merged

df_merged = merge_plot_data([df_rf, df_nettcr, df_atmtcr])
df_melted = pd.melt(df_merged, id_vars="time_samplings")

# Plotting function
def plot_metrics(df, metric, ylim, save_name):
    plt.rcParams["figure.figsize"] = (16, 8)
    fig, ax = plt.subplots(1)
    splot = sns.barplot(data=df[df['variable'].str.contains(metric)], x="time_samplings", y="value", hue="variable")
    plt.legend(loc='best', prop={'size': 6}, fontsize=11)

    for p in splot.patches:
        splot.annotate(format(p.get_height(), '.2f'),
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center',
                       xytext=(0, 10), size=10,
                       textcoords='offset points')
    plt.ylim(ylim)
    ax.set_xticklabels([])
    plt.xlabel('10-time samplings', fontsize=14)
    plt.ylabel(metric, fontsize=14)
    plt.setp(ax.get_legend().get_texts(), fontsize='12')
    plt.setp(ax.get_legend().get_title(), fontsize='12')
    plt.savefig(save_name)
    plt.show()

plot_metrics(df_melted, 'auc', [0.0, 0.95], '../../analysis/figures/epiTCRNetTCRATMTCRon10Subsamling_auc.png')
plot_metrics(df_melted, 'sensitivity', [0.0, 1.0], '../../analysis/figures/epiTCRNetTCRATMTCRon10Subsamling_sensitivity.png')
plot_metrics(df_melted, 'specificity', [0.0, 0.82], '../../analysis/figures/epiTCRNetTCRATMTCRon10Subsamling_specificity.png')
plot_metrics(df_melted, 'accuracy', [0.0, 0.80], '../../analysis/figures/epiTCRNetTCRATMTCRon10Subsamling_accuracy.png')
