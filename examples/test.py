#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
Author:
    Congqing He,hecongqing@hotmail.com
"""

from deeptext.models.textcnn import TextCNN
from deeptext.models.textrnn import TextRNN

import pandas as pd
import numpy as np
from keras.preprocessing import text
from keras.preprocessing import sequence
train = pd.read_csv("./data/train.csv")
dev= pd.read_csv("./data/dev.csv")
print(train.label.value_counts())

# 词向量

tokenizer = text.Tokenizer(num_words=None, lower=True)
tokenizer.fit_on_texts(list(train["texts"].values) + list(dev["texts"].values))

train_ = sequence.pad_sequences(tokenizer.texts_to_sequences(train["texts"].values), maxlen=200)
test_ = sequence.pad_sequences(tokenizer.texts_to_sequences(dev["texts"].values), maxlen=200)

word_index = tokenizer.word_index


embeddings_index = {}
with open('./data/glove.840B.300d.txt','r',encoding='utf-8') as f:
    for i in f:
        values = i.split(' ')
        if len(values) == 2: continue
        word = str(values[0])
        embedding = np.asarray(values[1:],dtype='float')
        embeddings_index[word] = embedding




k = 0
MAX_NB_WORDS = 300000
EMBEDDING_DIM = 300
nb_words = min(MAX_NB_WORDS, len(word_index))

embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NB_WORDS: continue
    embedding_vector = embeddings_index.get(str(word))
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else:
        unk_vec = np.random.random(EMBEDDING_DIM) * 0.5
        unk_vec = unk_vec - unk_vec.mean()
        embedding_matrix[i] = unk_vec
        k += 1

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
lb = LabelEncoder()
train_label = lb.fit_transform(train['label'].values)
train_label = to_categorical(train_label)

# model =TextCNN(seq_length=200,embedding_weights=embedding_matrix,kernel_sizes=[1,2,3,4],
#          kernel_filters=[64,64,64,64],activation='relu',pool_size=50,dropout_rate=0.1,label_size=4,
#          optimizer='adam').build()

model =TextRNN(seq_length=200,embedding_weights=embedding_matrix,rnn_type='GRU',bidirectional=True,hidden_size=100,
               pool_type="attpool",activation='relu',dropout_rate=0.1,label_size=4, optimizer='adam').build()


model.summary()
model.fit(train_,train_label)