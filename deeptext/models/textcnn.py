#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
Author:
    Congqing He,hecongqing@hotmail.com
Reference:
    Yoon  Kim. Convolutional neural networks for sentence classification. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1746–1751, Doha,Qatar, October 2014. Association for Computational Linguistics.
"""
import tensorflow as tf
from tensorflow.keras.layers import (Input, Embedding, Conv1D, concatenate,
                                     MaxPooling1D, Flatten, Dropout, Activation,
                                     BatchNormalization,
                                     Dense)
from tensorflow.keras.models import Model


class Text_CNN:
    def __init__(self, seq_length, embedding_weights, kernel_sizes, kernel_filters,
                 activation, pool_size, dropout_rate, label_size, optimizer):
        """
        :param seq_length: sequence length (int)
        :param embedding_weights: embedding weights matrix (array)
        :param kernel_sizes: kernel sizes (list)
        :param kernel_filters: kernel channels/filters (list)
        :param activation: activation （string）
        :param pool_size: pooling size (int)
        :param dropout_rate:  dropout rate (float)
        :param label_size: label size (int)
        :param optimizer: optimizer （string）
        """
        self.seq_length = seq_length
        self.embedding_weights = embedding_weights
        self.kernel_sizes = kernel_sizes
        self.kernel_filters = kernel_filters
        self.activation = activation
        self.pool_size = pool_size
        self.dropout_rate = dropout_rate
        self.label_size = label_size
        self.optimizer = optimizer

    def build(self):
        inp = Input(shape=(self.seq_length,), dtype='int32')
        embedding = Embedding(
            name="word_embedding",
            input_dim=self.embedding_weights.shape[0],
            weights=[self.embedding_weights],
            output_dim=self.embedding_weights.shape[1],
            trainable=False
        )
        embed = embedding(inp)
        cnn_feat = []
        for s, f in zip(self.kernel_sizes, self.kernel_filters):
            cnn = Conv1D(filters=f,
                         kernel_size=s,
                         strides=1,
                         padding='same',
                         activation=self.activation
                         )(embed)
            cnn = MaxPooling1D(pool_size=self.pool_size)(cnn)
            cnn_feat.append(cnn)
        fc = concatenate(cnn_feat, axis=-1)
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
