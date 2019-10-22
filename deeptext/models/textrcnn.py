#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
Author:
    Congqing He,hecongqing@hotmail.com
"""



import tensorflow as tf
from tensorflow.keras.layers import (Input,Embedding,Conv1D, concatenate,Flatten,
                                     MaxPooling1D,Dropout,Activation,BatchNormalization,Dense)
from tensorflow.keras.models import Model
from ..layers.rnn import RNN



class TextRCNN:
    def __init__(self, seq_length, embedding_weights, rnn_type, bidirectional, hidden_size,
                 kernel_sizes,kernel_filters, pool_size, activation, dropout_rate, label_size, optimizer):
        """
        :param seq_length: sequence length (int)
        :param embedding_weights:  embedding weights matrix (array)
        :param rnn_type: RNN type (optional="LSTM","GRU")
        :param bidirectional: bidirectional (bool =True or False)
        :param hidden_size:rnn hidden size (int)
        :param kernel_sizes: kernel sizes (list)
        :param kernel_filters: kernel channels/filters (list)
        :param pool_size: pooling size (int)
        :param activation: activation （string）
        :param dropout_rate: dropout rate (float)
        :param label_size: label size (int)
        :param optimizer: optimizer （string）
        """
        self.seq_length = seq_length
        self.embedding_weights = embedding_weights
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.kernel_sizes = kernel_sizes
        self.kernel_filters = kernel_filters
        self.pool_size = pool_size
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.label_size = label_size
        self.optimizer = optimizer


    def build(self):
        inp = Input(shape=(self.seq_length,), dtype='int32')
        embedding = Embedding(
            name="embedding",
            input_dim=self.embedding_weights.shape[0],
            weights=[self.embedding_weights],
            output_dim=self.embedding_weights.shape[1],
            trainable=False
        )
        embed = embedding(inp)
        rnn = RNN(self.rnn_type, self.bidirectional, self.hidden_size)(embed)
        cnn_feat=[]
        for s, f in zip(self.kernel_sizes, self.kernel_filters):
            cnn = Conv1D(filters=f,
                         kernel_size=s,
                         strides=1,
                         padding='same',
                         activation=self.activation
                         )(rnn)
            cnn = MaxPooling1D(pool_size=self.pool_size)(cnn)
            cnn_feat.append(cnn)
        if len(cnn_feat) >1:
            fc = concatenate(cnn_feat, axis=-1)
        else:
            fc = cnn_feat[0]
        fc = Flatten()(fc)
        fc = Dropout(rate=self.dropout_rate)(fc)

        fc = Dense(256)(fc)
        fc = BatchNormalization()(fc)
        fc = Activation(activation=self.activation)(fc)
        fc = Dropout(rate=self.dropout_rate)(fc)

        out = Dense(self.label_size, activation="softmax")(fc)
        model = Model(inputs=inp, outputs=out)
        model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        return model
