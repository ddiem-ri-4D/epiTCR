import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score

import utils as Utils

# Load predictions
test_files = [f"../../data/predToolsData/NetTCR/Pretrained/test{i:02d}_pred.csv" for i in range(1, 16)]
test_data = [pd.read_csv(file) for file in test_files]

# Initialize lists for storing results
accuracies, sensitivities, specificities, aucs = [], [], [], []

# Calculate metrics
for data in test_data:
    y_test = data["binder"].to_numpy()
    y_pred = data["binder_pred"].to_numpy()
    y_prob = data["prediction"].to_numpy()
    
    tn, fp, fn, tp = Utils.confusion_matrix(y_test, y_pred).ravel()
    
    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    auc_score = metrics.roc_auc_score(y_test, y_prob)
    
    accuracies.append(accuracy)
    sensitivities.append(sensitivity)
    specificities.append(specificity)
    aucs.append(auc_score)

# Create DataFrame for performance metrics
data = {
    'acc': accuracies,
    'sens': sensitivities,
    'spec': specificities,
    'auc': aucs
}
df = pd.DataFrame(data=data)
df.to_csv("../../data/outputPerformance/NetTCR/Pretrained/outputPerformance.csv", index=False)

# Calculate ROC curves
roc_curves = [roc_curve(data["binder"], data["prediction"]) for data in test_data]

# Plot ROC curves
fig, ax = plt.subplots(figsize=(12, 11))
mean_fpr = np.linspace(0, 1, 100)
tprs = []

for i, (fpr, tpr, _) in enumerate(roc_curves):
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    ax.plot(fpr, tpr, lw=1, alpha=0.3, label=f'NetTCR - test{i+1:02d} (AUC = {aucs[i]:.3f})')

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
plt.savefig("../../analysis/figures/benchmarkNetTCRPretrained.png", dpi=600)
plt.savefig("../../analysis/figures/benchmarkNetTCRPretrained.pdf", dpi=600)
plt.show()
