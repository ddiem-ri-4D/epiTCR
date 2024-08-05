import pandas as pd
import numpy as np
import pickle
import time

from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, roc_curve, auc, f1_score, plot_roc_curve
)
from sklearn.model_selection import cross_validate, GridSearchCV

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def train(lst_models, X, y, cv):
    res_table = []
    for mdl_name, model in tqdm(lst_models):
        tic = time.time()
        cv_res = cross_validate(
            model, X, y, cv=cv, return_train_score=True,
            scoring=['accuracy', 'roc_auc']
        )
        res_table.append([
            mdl_name, 
            cv_res['train_accuracy'].mean(), 
            cv_res['test_accuracy'].mean(), 
            np.abs(cv_res['train_accuracy'].mean() - cv_res['test_accuracy'].mean()),
            cv_res['train_accuracy'].std(),
            cv_res['test_accuracy'].std(),
            cv_res['train_roc_auc'].mean(),
            cv_res['test_roc_auc'].mean(),
            np.abs(cv_res['train_roc_auc'].mean() - cv_res['test_roc_auc'].mean()),
            cv_res['train_roc_auc'].std(),
            cv_res['test_roc_auc'].std(),
            cv_res['fit_time'].mean()
        ])
        toc = time.time()
        print(f'\tModel {mdl_name} has been trained in {toc - tic:.2f} seconds')
    
    res_table = pd.DataFrame(res_table, columns=[
        'model', 'train_acc', 'test_acc', 'diff_acc', 'train_acc_std', 'test_acc_std',
        'train_roc_auc', 'test_roc_auc', 'diff_roc_auc', 'train_roc_auc_std', 
        'test_roc_auc_std', 'fit_time'
    ])
    res_table.sort_values(by=['test_acc', 'test_roc_auc'], ascending=False, inplace=True)
    return res_table.reset_index(drop=True)

def confusionMatrix(y_true, y_pred):
    target_names = ['Non-binder', 'Binder']
    print(classification_report(y_true, y_pred, target_names=target_names))
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred), index=target_names, columns=target_names)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='g')
    plt.title('Confusion matrix')
    plt.xlabel('Predicted values')
    plt.ylabel('Actual values')
    plt.show()

def rocAuc(model, X, y_true):
    plot_roc_curve(model, X, y_true)
    plt.show()

def _rocAuc(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_value = roc_auc_score(y_true, y_score)
    plt.plot(fpr, tpr, label=f"AUC = {auc_value:.3f}")
    plt.legend(loc=4)
    plt.show()

def predicMLModel(model, data, X_test, y_test, path):
    y_rf_test_proba = model.predict_proba(X_test)[:, 1]
    df_test_rf = pd.DataFrame({'predict_proba': y_rf_test_proba})
    df_prob_test_rf = pd.concat([data, df_test_rf], axis=1)
    df_prob_test_rf['binder_pred'] = (df_prob_test_rf['predict_proba'] >= 0.5).astype(int)

    y_true = y_test["binder"].to_numpy()
    y_pred = df_prob_test_rf["binder_pred"].to_numpy()

    confusionMatrix(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    auc_value = roc_auc_score(y_true, df_prob_test_rf["predict_proba"])

    print(f"AUC: {auc_value:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Sensitivity (TPR): {sensitivity:.3f}")
    print(f"Specificity (TNR): {specificity:.3f}")

    df_prob_test_rf.to_csv(path, index=False)

    return round(auc_value, 3), round(accuracy, 3), round(sensitivity, 3), round(specificity, 3)

def trainTunningModel(lst_models, X, y, cv):
    models_final = []
    for model_name, model, params in tqdm(lst_models):
        tic = time.time()
        search = GridSearchCV(estimator=model, param_grid=params, cv=cv, scoring='roc_auc', n_jobs=-1)
        search.fit(X, y)
        model_tunned = model.set_params(**search.best_params_)
        models_final.append((model_name, model_tunned))
        toc = time.time()
        print(f'Model {model_name} has been tuned in {toc - tic:.2f} seconds')

    return models_final

def modelRun(algo, pX_res, py_res, pX_test, py_test):
    algo.fit(pX_res, np.ravel(py_res))
    y_pred = algo.predict(pX_test)
    y_pred_proba = algo.predict_proba(pX_test)[:, 1]

    accuracy = accuracy_score(py_test, y_pred)
    f1 = f1_score(py_test, y_pred)
    fpr, tpr, _ = roc_curve(py_test, y_pred_proba)
    auc_value = auc(fpr, tpr)
    tn, fp, fn, tp = confusion_matrix(py_test, y_pred).ravel()

    confusionMatrix(py_test, y_pred)

    print(algo)
    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1 score: {f1:.3f}")
    print(f"AUC: {auc_value:.3f}")
    print(f"Sensitivity (TPR): {tp / (tp + fn):.3f}")
    print(f"Specificity (TNR): {tn / (tn + fp):.3f}")

    rocAuc(algo, pX_test, py_test)
    return accuracy, f1, fpr, tpr, auc_value

def predictModel(path, path_out):
    data_predict = pd.read_csv(path)
    data_predict['binder'] = data_predict['binder'].astype(int)
    data_predict['binder_pred'] = (data_predict['prediction'] >= 0.5).astype(int)

    y_true = data_predict["binder"].to_numpy()
    y_pred = data_predict["binder_pred"].to_numpy()

    confusionMatrix(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    auc_value = roc_auc_score(y_true, data_predict["prediction"])

    print(f"AUC: {auc_value:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Sensitivity (TPR): {sensitivity:.3f}")
    print(f"Specificity (TNR): {specificity:.3f}")

    _rocAuc(y_true, data_predict["prediction"])
    data_predict.to_csv(path_out, index=False)

    return round(auc_value, 3), round(accuracy, 3), round(sensitivity, 3), round(specificity, 3)

def saveByPickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"{obj} has been saved at {path}.")

def evaluation(tunning_models, X_train, y_train, X_test, y_test):
    res = []
    for name, model in tqdm(tunning_models):
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        train_roc_auc = roc_auc_score(y_train, y_train_pred)
        test_roc_auc = roc_auc_score(y_test, y_test_pred)
        res.append([name, train_acc, test_acc, train_roc_auc, test_roc_auc])

    res = pd.DataFrame(res, columns=['model', 'train_acc', 'test_acc', 'train_roc_auc', 'test_roc_auc'])
    res.sort_values(by=['test_acc', 'test_roc_auc'], ascending=False, inplace=True)

    return res.reset_index(drop=True)
