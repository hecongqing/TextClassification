#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
Author:
    Congqing He,hecongqing@hotmail.com
"""
import tensorflow as tf
from tensorflow.keras.layers import (Input,Embedding,GlobalMaxPooling1D, GlobalAveragePooling1D,
                                     Dropout,Activation,BatchNormalization,Dense)
from tensorflow.keras.models import Model
from ..layers.rnn import RNN
from ..layers.rnn import RNNType
from ..layers.attentionpool import Attention


class TextRNN:
    def __init__(self, seq_length, embedding_weights, rnn_type, bidirectional, hidden_size, pool_type, activation,
                 dropout_rate, label_size, optimizer):
        """
        :param seq_length: sequence length (int)
        :param embedding_weights:  embedding weights matrix (array)
        :param rnn_type: RNN type (optional="LSTM","GRU")
        :param bidirectional: bidirectional (bool =True or False)
        :param hidden_size:rnn hidden size (int)
        :param pool_type: pooling type (optional='maxpool','avgpool','attpool')
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
        self.pool_type = pool_type
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
        pooling = None
        if self.pool_type == 'maxpool':
            pooling = GlobalMaxPooling1D()(rnn)
        elif self.pool_type == 'avgpool':
            pooling = GlobalAveragePooling1D()(rnn)
        elif self.pool_type == 'attpool':
            pooling = Attention()(rnn)

        fc = Dense(256)(pooling)
        fc = BatchNormalization()(fc)
        fc = Activation(activation=self.activation)(fc)
        fc = Dropout(rate=self.dropout_rate)(fc)

        out = Dense(self.label_size, activation="softmax")(fc)
        model = Model(inputs=inp, outputs=out)
        model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        return model

