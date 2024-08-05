import numpy as np
import pandas as pd
import time
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, classification_report, accuracy_score
import warnings
import modules.processor as Processor
import modules.model as Model

warnings.simplefilter(action="ignore", category=FutureWarning)

# Data loading
df = pd.read_csv("../../data/finalData/finalWithoutHLA.csv")
train = pd.read_csv("../../data/splitData/withoutMHC/train/train.csv")

# Extracting specific epitopes
values = ["GLCTLVAML", "NLVPMVATV", "GILGFVFTL", "TPRVTGGGAM", "ELAGIGILTV", "AVFDRKSDAK", "KLGGALQAK"]

data_final = train[train["epitope"].isin(values)].reset_index(drop=True)

# Splitting data by epitope
epitope_data = {value: df[df["epitope"] == value].reset_index(drop=True) for value in values}

# Preparing datasets
datasets = {}
for epitope, data in epitope_data.items():
    X = data.iloc[:, [0, 1]]
    y = data.iloc[:, [2]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pX_res, py_res = Processor.dataRepresentationDownsamplingWithoutMHCb(pd.concat([X_train, y_train], axis=1))
    pX_test = Processor.dataRepresentationBlosum62WithoutMHCb(X_test)
    datasets[epitope] = (pX_res, py_res, pX_test, y_test)

# Processing final dataset
X_final = data_final.iloc[:, [0, 1]]
y_final = data_final.iloc[:, [2]]
X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(X_final, y_final, test_size=0.2, random_state=42)
pX_res_final, py_res_final = Processor.dataRepresentationDownsamplingWithoutMHCb(pd.concat([X_train_final, y_train_final], axis=1))
pX_test_final = Processor.dataRepresentationBlosum62WithoutMHCb(X_test_final)

# Training models
model_rf = RandomForestClassifier(bootstrap=False, max_features=15, n_estimators=300, n_jobs=-1, random_state=42)

models = {}
for epitope, (pX_res, py_res, pX_test, y_test) in datasets.items():
    model = model_rf.fit(pX_res, np.ravel(py_res))
    models[epitope] = model

# Training final model
model_final = model_rf.fit(pX_res_final, np.ravel(py_res_final))

# Predictions and evaluations
results = {}
for epitope, (pX_res, py_res, pX_test, y_test) in datasets.items():
    auc_score, acc_score, sens, spec = Model.predicMLModel(models[epitope], X_test.reset_index(drop=True), pX_test.reset_index(drop=True), y_test.reset_index(drop=True), f'../../data/pred7DominantPeptide/predict{epitope}.csv')
    accuracy, classify_metrics, fpr, tpr, auc_score_rf, f1_score_rf = Model.modelRun(models[epitope], pX_res, py_res, pX_test, y_test)
    results[epitope] = (auc_score, acc_score, sens, spec, accuracy, classify_metrics, fpr, tpr, auc_score_rf, f1_score_rf)

# Final model evaluation
auc_score_final, acc_score_final, sens_final, spec_final = Model.predicMLModel(model_final, X_test_final.reset_index(drop=True), pX_test_final.reset_index(drop=True), y_test_final.reset_index(drop=True), '../../data/pred7DominantPeptide/predictfinal.csv')
accuracy_rf_final, classify_metrics_rf_final, fpr_rf_final, tpr_rf_final, auc_score_rf_final, f1_rf_final = Model.modelRun(model_final, pX_res_final, py_res_final, pX_test_final, y_test_final)

# Plotting ROC curves
fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(111)

for epitope, result in results.items():
    fpr, tpr, auc_score_rf = result[6], result[7], result[8]
    ax.plot(fpr, tpr, label=f'{epitope} (AUC = {round(auc_score_rf, 3)})', linewidth=2)

ax.plot(fpr_rf_final, tpr_rf_final, label=f'MLibTCR for 7 peptides (AUC = {round(auc_score_rf_final, 3)})', linewidth=2)

plt.legend(loc="best")
plt.xlabel("1 - Specificity", fontsize=12)
plt.ylabel("Sensitivity", fontsize=12)
plt.savefig("../../analysis/figures/ROCAUC7HighlyFPpeptides.png")
plt.savefig("../../analysis/figures/ROCAUC7HighlyFPpeptides.pdf")
plt.rcParams.update({'font.size': 12})
plt.show()
