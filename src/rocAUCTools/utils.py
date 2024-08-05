import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, roc_auc_score, confusion_matrix, classification_report

# Function to display the confusion matrix with seaborn heatmap
def confusionMatrix(y_true, y_pred):
    target_names = ['Non-binder', 'Binder']
    print(classification_report(y_true, y_pred, target_names=target_names))
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred), index=target_names, columns=target_names)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='g')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.show()

# Function to plot ROC curve and calculate AUC using matplotlib
def rocAuc(y_true, y_score):
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    auc_score = metrics.roc_auc_score(y_true, y_score)
    plt.plot(fpr, tpr, label="AUC="+str(round(auc_score, 3)))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random guess')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.grid()
    plt.show()
    print("AUC: ", auc_score)

# Function to calculate FPR and TPR for a given cutoff
def cal_fpr_and_tpr_with_one_cutoff(y_true, y_prred, cutoff):
    tp = np.where(np.logical_and(y_prred >= cutoff, y_true == 1))[0]
    fp = np.where(np.logical_and(y_prred >= cutoff, y_true == 0))[0]
    tpr = len(tp) / len(y_true[y_true == 1])
    fpr = len(fp) / len(y_true[y_true == 0])
    return fpr, tpr

# Function to calculate FPR and TPR across different cutoffs
def cal_fpr_and_tpr_in_different_cutoffs(y_true, y_prred, cutoffs):
    fpr_in_different_cutoffs = []
    tpr_in_different_cutoffs = []
    for cutoff in cutoffs:
        fpr, tpr = cal_fpr_and_tpr_with_one_cutoff(y_true, y_prred, cutoff)
        fpr_in_different_cutoffs.append(fpr)
        tpr_in_different_cutoffs.append(tpr)
    return np.array(fpr_in_different_cutoffs), np.array(tpr_in_different_cutoffs)

# Function to calculate FPRs and TPRs across different datasets
def cal_fprs_and_tprs_in_different_sets(y_trues, y_preds, cutoffs):
    fprs_in_different_sets = []
    tprs_in_different_sets = []
    for y_true, y_pred in zip(y_trues, y_preds):
        fprs_in_a_set, tprs_in_a_set = cal_fpr_and_tpr_in_different_cutoffs(y_true, y_pred, cutoffs)
        fprs_in_different_sets.append(fprs_in_a_set)
        tprs_in_different_sets.append(tprs_in_a_set)
    fpr_mean = np.mean(fprs_in_different_sets, axis=0)
    tpr_mean = np.mean(tprs_in_different_sets, axis=0)
    tpr_std = np.std(tprs_in_different_sets, axis=0)
    return fprs_in_different_sets, tprs_in_different_sets, fpr_mean, tpr_mean, tpr_std

# Example usage of the functions
if __name__ == "__main__":
    # Assuming test predictions are stored in CSV files
    test_files = [f"../../data/predToolsData/NetTCR/Retraining/test{i:02d}_pred.csv" for i in range(1, 16)]
    test_data = [pd.read_csv(file) for file in test_files]

    # Initialize lists for storing results
    accuracies, sensitivities, specificities, aucs = [], [], [], []

    # Calculate metrics
    for data in test_data:
        y_test = data["binder"].to_numpy()
        y_pred = data["binder_pred"].to_numpy()
        y_prob = data["prediction"].to_numpy()
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        accuracy = accuracy_score(y_test, y_pred)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        auc_score = metrics.roc_auc_score(y_test, y_prob)
        
        accuracies.append(accuracy)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        aucs.append(auc_score)

        # Plot confusion matrix for each test set
        confusionMatrix(y_test, y_pred)
        # Plot ROC curve for each test set
        rocAuc(y_test, y_prob)

    # Create DataFrame for performance metrics
    data = {
        'acc': accuracies,
        'sens': sensitivities,
        'spec': specificities,
        'auc': aucs
    }
    df = pd.DataFrame(data=data)
    df.to_csv("../../data/outputPerformance/NetTCR/Retraining/outputPerformance.csv", index=False)

    # Calculate ROC curves
    roc_curves = [roc_curve(data["binder"], data["prediction"]) for data in test_data]

    # Plot ROC curves
    fig, ax = plt.subplots(figsize=(12, 11))
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    for i, (fpr, tpr, _) in enumerate(roc_curves):
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        ax.plot(fpr, tpr, lw=1, alpha=0.3, label=f'NetTCR* - test{i+1:02d} (AUC = {aucs[i]:.3f})')

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    ax.plot(mean_fpr, mean_tpr, color='b', lw=2, alpha=0.8,
            label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
    ax.fill_between(mean_fpr, np.maximum(mean_tpr - np.std(tprs, axis=0), 0), 
                    np.minimum(mean_tpr + np.std(tprs, axis=0), 1), color='grey', alpha=0.2, label='± 1 std. dev.')

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Random guess', alpha=0.8)

    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.legend(loc='lower right', fontsize=12)
    plt.savefig("../../analysis/figures/benchmarkNetTCRRetraining.png", dpi=600)
    plt.savefig("../../analysis/figures/benchmarkNetTCRRetraining.pdf", dpi=600)
    plt.show()
