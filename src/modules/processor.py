import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from IPython.display import display


codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        
def one_hot_encode(seq):
    o = list(set(codes) - set(seq))
    s = pd.DataFrame(list(seq))    
    x = pd.DataFrame(np.zeros((len(seq),len(o)),dtype=int),columns=o)    
    a = s[0].str.get_dummies(sep=',')
    a = a.join(x)
    a = a.sort_index(axis=1)
    e = a.values.flatten()
    return e


def data_representation(X_data):
    seq_cdr3_lst = X_data['CDR3b'].to_list()
    seq_epitope_lst = X_data['epitope'].to_list()
    
    lst1 = []
    for seq in seq_cdr3_lst:
        e=one_hot_encode(seq)
        m_cdr3 = np.reshape(e, (-1, 20))
        padding = 19 - len(m_cdr3)
        c = np.concatenate((m_cdr3, np.zeros(padding * 20).reshape(padding, 20)), axis=0)
        lst1.append(c)

    lst2 = []
    for seq in seq_epitope_lst:
        e=one_hot_encode(seq)
        m_epitope = np.reshape(e, (-1, 20))
        padding = 11 - len(m_epitope)
        c = np.concatenate((m_epitope, np.zeros(padding * 20).reshape(padding, 20)), axis=0)
        lst2.append(c)
        
    res1 = list(map(lambda x: lst1[x].flatten(), range(len(lst1)))) 
    res2 = list(map(lambda x: lst2[x].flatten(), range(len(lst2))))
    
    df_res1 = pd.DataFrame(res1)
    df_res2 = pd.DataFrame(res2)
    result = pd.concat([df_res1, df_res2], axis=1)
    
    return result

def data_representation_mhc(X_data):
    seq_cdr3_lst = X_data['CDR3b'].to_list()
    seq_epitope_lst = X_data['epitope'].to_list()
    seq_mhc_lst = X_data['MHC'].to_list()
    
    lst1 = []
    for seq in seq_cdr3_lst:
        e=one_hot_encode(seq)
        m_cdr3 = np.reshape(e, (-1, 20))
        padding = 19 - len(m_cdr3)
        c = np.concatenate((m_cdr3, np.zeros(padding * 20).reshape(padding, 20)), axis=0)
        lst1.append(c)

    lst2 = []
    for seq in seq_epitope_lst:
        e=one_hot_encode(seq)
        m_epitope = np.reshape(e, (-1, 20))
        padding = 11 - len(m_epitope)
        c = np.concatenate((m_epitope, np.zeros(padding * 20).reshape(padding, 20)), axis=0)
        lst2.append(c)
    
    lst3 = []
    for seq in seq_mhc_lst:
        e=one_hot_encode(seq)
        m_mhc = np.reshape(e, (-1, 20))
        padding = 366 - len(m_mhc)
        c = np.concatenate((m_mhc, np.zeros(padding * 20).reshape(padding, 20)), axis=0)
        lst3.append(c)

    res1 = list(map(lambda x: lst1[x].flatten(), range(len(lst1)))) 
    res2 = list(map(lambda x: lst2[x].flatten(), range(len(lst2))))
    res3 = list(map(lambda x: lst3[x].flatten(), range(len(lst3))))
    
    df_res1 = pd.DataFrame(res1)
    df_res2 = pd.DataFrame(res2)
    df_res3 = pd.DataFrame(res3)
    result = pd.concat([df_res1, df_res2, df_res3], axis=1)
    
    return result

def split_dataframe_by_position(df, splits):
    """
    Takes a dataframe and an integer of the number of splits to create.
    Returns a list of dataframes.
    """
    dataframes = []
    index_to_split = len(df) // splits
    start = 0
    end = index_to_split
    for split in range(splits):
        temporary_df = df.iloc[start:end, :]
        dataframes.append(temporary_df)
        start += index_to_split
        end += index_to_split
    return dataframes


class epitcrModel:
    def __init__(self, pmodel, pX, py):
        self.model = pmodel
        self.model.fit(pX, py)
    
    def predict(self, pnew_data):
        yhat_class = self.model.predict(pnew_data)
        return yhat_class 

    
    def info(self):
        print(self.model)
    
    def rocAuc(self, X, y_true):
        plot_roc_curve(self.model, X, y_true)
        plt.show()
    
    def predict_proba(self, pnew_data):
        yhat_class = self.model.predict_proba(pnew_data)
        return yhat_class 