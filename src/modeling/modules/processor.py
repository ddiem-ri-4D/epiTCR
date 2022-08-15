import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#from gensim.models import Word2Vec
#from gensim.models import Phrases
#from gensim.models.phrases import Phraser
from sklearn.decomposition import PCA
from IPython.display import display


codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        
vector_size = 20
window_size = 2
min_count = 2
workers = 4
sg = 1
         
def enc_list_bl_max_len1(aa_seqs, blosum, max_seq_len):
    '''
    blosum encoding of a list of amino acid sequences with padding 
    to a max length

    parameters:
        - aa_seqs : list with AA sequences
        - blosum : dictionnary: key= AA, value= blosum encoding
        - max_seq_len: common length for padding
    returns:
        - enc_aa_seq : list of np.ndarrays containing padded, encoded amino acid sequences
    '''

    # encode sequences:
    sequences=[]
    for seq in aa_seqs:
        e_seq=np.zeros((len(seq),len(blosum["A"])))
        count=0
        for aa in seq:
            if aa in blosum:
                e_seq[count]=blosum[aa]
                count+=1
            else:
                sys.stderr.write("Unknown amino acid in peptides: "+ aa +", encoding aborted!\n")
                sys.exit(2)
                
        sequences.append(e_seq)

    # pad sequences:
    #max_seq_len = max([len(x) for x in aa_seqs])
    n_seqs = len(aa_seqs)
    n_features = sequences[0].shape[1]

    enc_aa_seq = np.zeros((n_seqs, max_seq_len, n_features))
    for i in range(0,n_seqs):
        enc_aa_seq[i, :sequences[i].shape[0], :n_features] = sequences[i]
    return enc_aa_seq

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

def show_matrix(m):
    #display a matrix
    cm = sns.light_palette("seagreen", as_cmap=True)
    display(m.style.background_gradient(cmap=cm))

def encoding(lst_seq, mlen_seq):
    lst = []
    for seq in lst_seq:
        e= one_hot_encode(seq)
        m_seq = np.reshape(e, (-1, 20))
        padding = mlen_seq - len(m_seq)
        c = np.concatenate((m_seq, np.zeros(padding * 20).reshape(padding, 20)), axis=0)
        lst.append(c)
    return lst

def cv_seq(lst):
    res = []
    for x in lst:
        tmp = " ".join(x)
        res.append(tmp)
    return res

def lst_corpus(lst):
    corpus = []
    tmp = [line.split(".") for line in cv_seq(lst)]
    for line in cv_seq(lst):
        words = [x for x in line.split()]
        corpus.append(words)
    return corpus

def Word2Vec_model(corpus):
    res_model = []
    for x in corpus:
        model = Word2Vec(x, min_count=1, vector_size=vector_size, window=window_size)
        model.save('model/word2vec.model')
        new_model = Word2Vec.load('model/word2vec.model')
        res_model.append(new_model)
    return res_model

def padding1(seq, lst_seq):
    res = []
    for x in seq:
        max_seq = max(lst_seq, key = len)
        ml_seq = len(max_seq)
        m_seq = len(x)
        padding = ml_seq - m_seq
        c1 = [x, np.zeros(padding * 20).reshape(padding, 20)]
        res.append(c1)
    return res

def lst_mt(corpus):
    res_mt = []
    tmp = Word2Vec_model(corpus)
    for x in tmp:
        X = x.wv[x.wv.index_to_key]
        res_mt.append(X)
    return res_mt