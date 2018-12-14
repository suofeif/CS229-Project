#refer: https://github.com/brightmart/sentiment_analysis_fine_grain/blob/master/preprocess_word.ipynb

import random
random.seed = 16
import pandas as pd
import tensorflow as tf
#from gensim.models.word2vec import Word2Vec
import jieba
from collections import Counter
import numpy as np
import os
import pickle
import config
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)

def train_tencent_model():
    wv_from_text = KeyedVectors.load_word2vec_format("/scratch/users/qingyin/Tencent_AILab_ChineseEmbedding.txt", binary = False)
    vocab = wv_from_text.wv.vocab

    print("length of vocab")
    logger.info("length of vocab"+str(len(vocab)))

    print(len(vocab))
    return vocab, wv_from_text



def load_data_from_csv(file_name, header=0, encoding="utf-8"):

    data_df = pd.read_csv(file_name, header=header, encoding=encoding)

    return data_df


# text tokenization
def seg_words(contents):
    contents_segs = list()
    for content in contents:
        segs = jieba.lcut(content, cut_all=False)
        contents_segs.append(" ".join(segs))

    return contents_segs

def train_vec(sentences):
    model = Word2Vec(sentences,size=300, window = 5, min_count=3, iter = 100)

    vocab = model.wv.vocab
    print("length of vocab")
    print(len(vocab))
    return vocab, model

def convert_to_onehot(labels, class_num):
    one_hot_mat = []
    for j in range(len(labels)):
        label = labels[j]
        new_labels = [0 for i in range(class_num)]
        label_types = {
        -2: 0,
        -1: 1,
        0: 2,
        1: 3
        }
        new_labels[label_types[label]] = 1
        one_hot_mat.append(new_labels) # list concatenation
    return one_hot_mat

def wordToIndex(vocab):
    wordtoindex = {}
    wordtoindex["PAD"]=0
    wordtoindex["UNK"]=1
    i=2
    for k in vocab:
        wordtoindex[k] = i
        i = i+1
    return wordtoindex

def sentence_to_indice(lists, word2index, max_len, vocab):
    X = np.array(lists)
    m = X.shape[0]
    X_indices = np.zeros((m, max_len))
    for i in range(m):
        sentence = lists[i]
        for j in range(len(sentence)):
            if j == max_len:
                break
            word = sentence[j]
            k=1 #unk index
            if word in vocab:
                k = word2index[word]
            X_indices[i, j] = k

    return X_indices

def embedding_data(vocab_len, emb_dim, word2index, embedding_model, vocab):
    vocab_len = vocab_len
    emb_dim = emb_dim
    logger.info("embedding_matrix size: ")
    emb_matrix = np.zeros((vocab_len, emb_dim))
    emb_matrix[1, :] = np.ones((1, emb_dim))
    for word, index in word2index.items():
        if word in vocab:
            emb_matrix[index, :] = embedding_model[word]

    return emb_matrix


def get_uniq_index(train_indices, val_indices, test_indices):

    uniq_train = np.unique(train_indices)
    uniq_val=np.unique(val_indices)
    uniq_test=np.unique(test_indices)
    result = np.concatenate((uniq_train, uniq_val, uniq_test), axis = None,)
    result = np.unique(result)
    result = result.astype(int)

    return result

def convert_oh_to_cls_name(y_oh):
    class_index2name = {
    0: -2,
    1: -1,
    2: 0,
    3: 1
    }

    #y_list_temp = tf.Session().run(y_oh)
    #y_list = y_list_temp.tolist()
    y_list = y_oh.tolist()
    result = []
    for i in range(len(y_list)):
        single_one_hot = y_list[i]
        #print("y_true:")
        #print(single_one_hot)
        index = single_one_hot.index(1)
        #print("index: "+str(index))
        class_name = class_index2name.get(index)
        result.append(class_name)
    return result

def convert_prob_to_cls_name(y_prob):
    class_index2name = {
    0: -2,
    1: -1,
    2: 0,
    3: 1
    }

    result = []
    y_prob = np.matrix(y_prob)
    for i in range(y_prob.shape[0]):
        single_prob = np.array(y_prob[i, :])
        index = np.argmax(single_prob)
        class_name = class_index2name.get(index)
        result.append(class_name)
    return result

def new_indice(train_indices, ind_list):
    m = train_indices.shape[0]
    n = train_indices.shape[1]
    train = np.zeros((m, n))
    for i in range(m):
        sentence = train_indices[i, :]
        for j in range(n):
            k = ind_list.index(sentence[j])
            train[i, j] = k

    return train
