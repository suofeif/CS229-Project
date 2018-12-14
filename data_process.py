#!/user/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
import jieba
from gensim.models import Word2Vec
from gensim.models.word2vec import KeyedVectors


# load data from csv data files
def load_data_from_csv(file_name, header=0, encoding="utf-8"):

    data_df = pd.read_csv(file_name, header=header, encoding=encoding)

    return data_df


# text tokenization
def seg_words(contents):
    contents_segs = list()
    for content in contents:
        segs = jieba.lcut(content)
        contents_segs.append(" ".join(segs))

    return contents_segs

def load_word2vec_model():
    #wv_from_text = KeyedVectors.load_word2vec_format("../tencentVec/Tencent_AILab_ChineseEmbedding.txt", binary = False)

    return wv_from_text
