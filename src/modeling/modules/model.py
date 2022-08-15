import numpy as np
import pandas as pd
import pickle 
import time

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.model_selection import train_test_split


def dataSplitSaved(pdata: pd.DataFrame, ptest_size: float, ppath: str):
    X_train, X_test, y_train, y_test = train_test_split(pdata.iloc[:, :-1], pdata.iloc[:, -1], test_size=ptest_size, random_state=42)
    
    X_train.to_csv(f"{ppath}/train/X.csv", index=False)
    y_train.to_csv(f"{ppath}/train/y.csv", index=False)

    X_test.to_csv(f"{ppath}/test/X.csv", index=False)
    y_test.to_csv(f"{ppath}/test/y.csv", index=False)

def loadData(ppath: str):
    X = pd.read_csv(f"{ppath}/X.csv")
    y = pd.read_csv(f"{ppath}/y.csv")
    
    return X, y

def saveByPickle(object, path):
    pickle.dump(object, open(path, "wb"))
    print(f"{object} has been saved at {path}.")

def train(lst_models, X, y, cv):
    res_table = []
    for mdl_name, model in lst_models:
        tic = time.time()
        cv_res = cross_validate(model, X, y, cv=cv, return_train_score=True, scoring=['accuracy', 'roc_auc'])
        res_table.append([mdl_name, 
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
        print('\tModel {} has been trained in {:,.2f} seconds'.format(mdl_name, (toc - tic)))
    
    res_table = pd.DataFrame(res_table, columns=['model', 'train_acc', 'test_acc', 'diff_acc',
                                                 'train_acc_std', 'test_acc_std', 'train_roc_auc', 'test_roc_auc',
                                                 'diff_roc_auc', 'train_roc_auc_std', 'test_roc_auc_std', 'fit_time'])
    res_table.sort_values(by=['test_acc', 'test_roc_auc'], ascending=False, inplace=True)

    return res_table.reset_index(drop=True)

