import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Function to read prediction data
def read_predictions(file_paths):
    return [pd.read_csv(file) for file in file_paths]

# Function to calculate ROC and AUC
def calculate_roc_auc(predictions, true_label, pred_proba):
    fpr_list, tpr_list, auc_list = [], [], []
    for df in predictions:
        y_true = df[true_label].to_numpy()
        y_proba = df[pred_proba]
        fpr, tpr, _ = roc_curve(y_true, y_proba, drop_intermediate=False)
        auc_score = auc(fpr, tpr)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        auc_list.append(auc_score)
    return fpr_list, tpr_list, auc_list

# Function to plot ROC curves
def plot_roc_curves(fpr_list, tpr_list, label, color, linestyle="-", linewidth=1):
    for fpr, tpr in zip(fpr_list, tpr_list):
        plt.plot(fpr, tpr, label=label, linewidth=linewidth, color=color, linestyle=linestyle)
        label = "_nolegend_"

# File paths for each tool's prediction data
file_paths_epitcr = [f"../../data/predepTCRData/withoutMHC/test{i:02d}_predict_proba.csv" for i in range(1, 16)]
file_paths_imrex = [f"../../data/predToolsData/Imrex/test{i:02d}_pred.csv" for i in range(1, 16)]
file_paths_nettcr = [f"../../data/predToolsData/nettcr/Pretrained/test{i:02d}_pred.csv" for i in range(1, 16)]
file_paths_nettcr2 = [f"../../data/predToolsData/nettcr/Retraining/test{i:02d}_pred.csv" for i in range(1, 16)]
file_paths_atmtcr = [f"../../data/predToolsData/ATMTCR/Pretrained/test{i:02d}_pred.csv" for i in range(1, 16)]
file_paths_atmtcr2 = [f"../../data/predToolsData/ATMTCR/Retraining/test{i:02d}_pred.csv" for i in range(1, 16)]

# Read prediction data
predictions_epitcr = read_predictions(file_paths_epitcr)
predictions_imrex = read_predictions(file_paths_imrex)
predictions_nettcr = read_predictions(file_paths_nettcr)
predictions_nettcr2 = read_predictions(file_paths_nettcr2)
predictions_atmtcr = read_predictions(file_paths_atmtcr)
predictions_atmtcr2 = read_predictions(file_paths_atmtcr2)

# Calculate ROC and AUC for each tool
fpr_epitcr, tpr_epitcr, auc_epitcr = calculate_roc_auc(predictions_epitcr, "binder", "predict_proba")
fpr_imrex, tpr_imrex, auc_imrex = calculate_roc_auc(predictions_imrex, "binder", "prediction_score")
fpr_nettcr, tpr_nettcr, auc_nettcr = calculate_roc_auc(predictions_nettcr, "binder", "binder_pred")
fpr_nettcr2, tpr_nettcr2, auc_nettcr2 = calculate_roc_auc(predictions_nettcr2, "binder", "prediction")
fpr_atmtcr, tpr_atmtcr, auc_atmtcr = calculate_roc_auc(predictions_atmtcr, "binder", "predict_proba")
fpr_atmtcr2, tpr_atmtcr2, auc_atmtcr2 = calculate_roc_auc(predictions_atmtcr2, "binder", "prediction")

# Plotting
plt.figure(figsize=(7, 7))
plot_roc_curves(fpr_epitcr, tpr_epitcr, 'epiTCR - Mean ROC (AUC = 0.980)', color="r")
plot_roc_curves(fpr_imrex, tpr_imrex, 'Imrex - Mean ROC (AUC = 0.551)', color="g")
plot_roc_curves(fpr_nettcr, tpr_nettcr, 'NetTCR - Mean ROC (AUC = 0.518)', color="purple")
plot_roc_curves(fpr_nettcr2, tpr_nettcr2, 'NetTCR* - Mean ROC (AUC = 0.931)', color="orange")
plot_roc_curves(fpr_atmtcr, tpr_atmtcr, 'ATM-TCR - Mean ROC (AUC = 0.494)', color="c")
plot_roc_curves(fpr_atmtcr2, tpr_atmtcr2, 'ATM-TCR* - Mean ROC (AUC = 0.494)', color="salmon")

plt.plot([0, 1], [0, 1], linestyle="dashed", lw=1, color="k", label="Random guess", alpha=0.8)
plt.legend(loc="best")
plt.xlabel("1 - Specificity", fontsize=11)
plt.ylabel("Sensitivity", fontsize=11)
plt.savefig("../../analysis/figures/benchmarkToolsWithoutMHC.png", dpi=600)
plt.savefig("../../analysis/figures/benchmarkToolsWithoutMHC.pdf", dpi=600)
plt.show()
