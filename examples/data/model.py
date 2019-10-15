
# coding: utf-8

# In[1]:


# coding: UTF-8
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.model_selection import train_test_split, StratifiedKFold
import keras
import tensorflow as tf
import codecs
import gc
from multiprocessing import Pool
import contextlib
import pickle
import os
from sklearn.metrics import roc_auc_score

from keras.utils.training_utils import multi_gpu_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import *
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from keras.activations import *
from keras.optimizers import *
import warnings

import os
import gc
import random

from gensim.models import Word2Vec


# In[18]:



train_set = pd.read_csv('../input/train_set.csv')
#train_set0 = train_set[train_set.label==0].sample(frac=0.3,replace=False)
#train_set1 = train_set[train_set.label==1]
#train_set = pd.concat([train_set0,train_set1])
valid_set = pd.read_csv('../input/valid_set.csv')
train_des = pd.read_csv("../input/train_release.csv")
train_des = train_des.drop_duplicates(subset=['description_id'])
train_des = train_des[['description_id','description_text']]


candidate_paper = pd.read_csv("../input/candidate_paper.csv")

candidate_paper['paper_text'] = candidate_paper['title'] +" "+ candidate_paper['abstract'] +" "+ candidate_paper['journal'] +" "+  candidate_paper['keywords']

valid_release = pd.read_csv("../input/validation_release.csv")

train = pd.merge(train_set, train_des, on='description_id', how='left')
train = pd.merge(train, candidate_paper, on='paper_id', how='left')

test = pd.merge(valid_set, valid_release, on='description_id', how='left')
test = pd.merge(test, candidate_paper, on='paper_id', how='left')

labels = train['label'].astype(int).values

train['description_text'] =train['description_text'].map(lambda x: str(x))
test['description_text'] =test['description_text'].map(lambda x: str(x))
train['paper_text'] =train['paper_text'].map(lambda x: str(x))
test['paper_text'] =test['paper_text'].map(lambda x: str(x))


# In[19]:



MAX_NB_WORDS = 300000

tokenizer = Tokenizer(num_words=MAX_NB_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(train['description_text'].tolist()+train['paper_text'].tolist())
train_q1_word_seq = tokenizer.texts_to_sequences(train['description_text'])
train_q2_word_seq = tokenizer.texts_to_sequences(train['paper_text'])
test_q1_word_seq = tokenizer.texts_to_sequences(test['description_text'])
test_q2_word_seq = tokenizer.texts_to_sequences(test['paper_text'])

L_MAX_WORD_SEQUENCE_LENGTH=400
R_MAX_WORD_SEQUENCE_LENGTH=300


train_q1_word_seq  = pad_sequences(train_q1_word_seq, maxlen=L_MAX_WORD_SEQUENCE_LENGTH, truncating='post', value=0)
train_q2_word_seq  = pad_sequences(train_q2_word_seq, maxlen=R_MAX_WORD_SEQUENCE_LENGTH, truncating='post', value=0)

test_q1_word_seq   = pad_sequences(test_q1_word_seq, maxlen=L_MAX_WORD_SEQUENCE_LENGTH, truncating='post', value=0)
test_q2_word_seq   = pad_sequences(test_q2_word_seq, maxlen=R_MAX_WORD_SEQUENCE_LENGTH, truncating='post', value=0)

print(train_q1_word_seq)
# In[24]:


word_index = tokenizer.word_index


# In[26]:


len(word_index)


# In[29]:



embeddings_index = {}
with open('../input/glove.840B.300d.txt','r') as f:
    for i in f:
        values = i.split(' ')
        if len(values) == 2: continue
        word = str(values[0])
        embedding = np.asarray(values[1:],dtype='float')
        embeddings_index[word] = embedding
print('char embedding',len(embeddings_index))




# In[30]:


k=0
EMBEDDING_DIM = 300
nb_words = min(MAX_NB_WORDS, len(word_index))

char_embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NB_WORDS: continue
    embedding_vector = embeddings_index.get(str(word))
    if embedding_vector is not None:
        char_embedding_matrix[i] = embedding_vector
    else:
        unk_vec = np.random.random(EMBEDDING_DIM) * 0.5
        unk_vec = unk_vec - unk_vec.mean()
        char_embedding_matrix[i] = unk_vec     
        k +=1
        


# In[31]:


k


# In[32]:



def lstm_cross_char_add_feature():
    char_embedding_layer = Embedding(name="char_embedding",
                                     input_dim=char_embedding_matrix.shape[0],
                                     weights=[char_embedding_matrix],
                                     output_dim=char_embedding_matrix.shape[1],
                                     trainable=False)
    q1_char = Input(shape=(L_MAX_WORD_SEQUENCE_LENGTH,), dtype="int32")
    q1_char_embed = SpatialDropout1D(0.3)(char_embedding_layer(q1_char))

    q2_char = Input(shape=(R_MAX_WORD_SEQUENCE_LENGTH,), dtype="int32")
    q2_char_embed = SpatialDropout1D(0.3)(char_embedding_layer(q2_char))

    char_bilstm = Bidirectional(CuDNNLSTM(200, return_sequences=True))

    q1_char_encoded = Dropout(0.3)(char_bilstm(q1_char_embed))
    q1_char_encoded = Concatenate(axis=-1)([q1_char_encoded, q1_char_embed])
    q1_char_encoded = GlobalAveragePooling1D()(q1_char_encoded)

    q2_char_encoded = Dropout(0.3)(char_bilstm(q2_char_embed))
    q2_char_encoded = Concatenate(axis=-1)([q2_char_encoded, q2_char_embed])
    q2_char_encoded = GlobalAveragePooling1D()(q2_char_encoded)

    char_diff = Lambda(lambda x: K.abs(x[0] - x[1]))([q1_char_encoded, q2_char_encoded])
    char_angle = Lambda(lambda x: x[0] * x[1])([q1_char_encoded, q2_char_encoded])

    # Classifier
    merged = concatenate([char_diff, char_angle])
    merged = Dropout(0.5)(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(200, activation="relu")(merged)
    merged = Dropout(0.5)(merged)
    merged = BatchNormalization()(merged)
    out = Dense(1, activation="sigmoid")(merged)

    model = Model(inputs=[q1_char, q2_char], outputs=out)
    model.compile(loss='binary_crossentropy', optimizer='nadam',metrics=['acc'])
    return model


# In[35]:


model_count=0

for train_idx, val_idx in StratifiedKFold(n_splits=10, shuffle=True, random_state=42).split(labels,labels):
    val_q1_word_seq, val_q2_word_seq = train_q1_word_seq[val_idx], train_q2_word_seq[val_idx]
    train_q1_word_seq_, train_q2_word_seq_ = train_q1_word_seq[train_idx], train_q2_word_seq[train_idx]
    train_y, val_y = labels[train_idx], labels[val_idx]
    model = lstm_cross_char_add_feature()
    model.summary()
    early_stopping = EarlyStopping(monitor="val_loss", patience=8)
    plateau = ReduceLROnPlateau(monitor="val_loss", verbose=1, mode='min', factor=0.5, patience=3)
    best_model_path = "../models/" + "lstm_cross_char_add_feature_augmentation_best_model" + str(model_count) + ".h5"
    model_checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, save_weights_only=True)

    if not os.path.exists(best_model_path):
        hist = model.fit([train_q1_word_seq_, train_q2_word_seq_],
                         train_y,
                         validation_data=([val_q1_word_seq, val_q2_word_seq], val_y),
                         epochs=50,
                         batch_size=128,
                         shuffle=True,
                         callbacks=[early_stopping, model_checkpoint, plateau],
                         verbose=2)

    model.load_weights(best_model_path)
    print(model.evaluate([val_q1_word_seq, val_q2_word_seq], val_y, batch_size=128, verbose=1))
    res = test[['description_id','paper_id']]
    res['prob'] = model.predict([test_q1_word_seq, test_q2_word_seq], batch_size=2 ** 9)
    res.to_csv("res{0}.csv".format(model_count),index=False)
    model_count +=1

