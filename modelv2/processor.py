import pandas as pd
import numpy as np
import utils as Utils

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
    seq_cdr3_lst = X_data['CDR3'].to_list()
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
    seq_cdr3_lst = X_data['CDR3'].to_list()
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

def data_representation_cnn(X_data):
    seq_cdr3_lst = X_data['CDR3'].to_list()
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

    return lst1, lst2


codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

def one_hot_encode(seq):
    o = list(set(codes) - set(seq))
    s = pd.DataFrame(list(seq))    
    x = pd.DataFrame(np.zeros((len(seq),len(o)),dtype=int),columns=o)    
    a = s[0].str.get_dummies(sep=',')
    a = a.join(x)
    a = a.sort_index(axis=1)
    # show_matrix(a)
    e = a.values.flatten()
    return e

def encoder_sequence(df):
    seq_cdr3_lst = df['CDR3'].to_list()
    seq_epitope_lst = df['epitope'].to_list()
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

    return lst1, lst2

def encoder_sequence_mhc(df):
    seq_cdr3_lst = df['CDR3'].to_list()
    seq_epitope_lst = df['epitope'].to_list()
    seq_hla_lst = df['MHC'].to_list()

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
    for seq in seq_hla_lst:
        e=one_hot_encode(seq)
        m_hla = np.reshape(e, (-1, 20))
        padding = 34 - len(m_hla)
        c = np.concatenate((m_hla, np.zeros(padding * 20).reshape(padding, 20)), axis=0)
        lst3.append(c)

    return lst1, lst2, lst3

def data_representation_blosum62(X_data):
    encoding = Utils.blosum62_20aa
    
    m_cdr3 = Utils.enc_list_bl_max_len(X_data.CDR3b, encoding, 19)
    m_cdr3 = m_cdr3.reshape(len(m_cdr3), 380)
    
    m_epitope = Utils.enc_list_bl_max_len(X_data.epitope, encoding, 11)
    m_epitope = m_epitope.reshape(len(m_epitope), 220)
    
    df_res1 = pd.DataFrame(m_cdr3)
    df_res2 = pd.DataFrame(m_epitope)
    
    res = pd.concat([df_res1, df_res2], axis=1)
    
    return res

def data_representation_blosum62_cnn(X_data):
    encoding = Utils.blosum62_20aa

    m_epitope = Utils.enc_list_bl_max_len(X_data.epitope, encoding, 11)
    m_cdr3 = Utils.enc_list_bl_max_len(X_data.CDR3b, encoding, 19)
    
    res = [m_cdr3, m_epitope]
    return res 