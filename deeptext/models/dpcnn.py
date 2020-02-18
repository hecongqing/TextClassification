#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
Author:
    Congqing He,hecongqing@hotmail.com
"""
import tensorflow as tf
from tensorflow.keras.layers import (Input, Embedding, Conv1D, concatenate,
                                     MaxPooling1D, Flatten, Dropout, Activation,
                                     BatchNormalization,
                                     Dense)
from tensorflow.keras.models import Model





class DPCNN:
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
            name="embedding",
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






def get_text_dpcnn(sent_length, embeddings_weight,class_num):
    print("get_text_dpcnn")
    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0],
        weights=[embeddings_weight],
        output_dim=embeddings_weight.shape[1],
        trainable=False)

    embed = SpatialDropout1D(0.2)(embedding(content))

    block1 = Conv1D(128, kernel_size=3, padding='same', activation='linear')(embed)
    block1 = BatchNormalization()(block1)
    block1 = PReLU()(block1)
    block1 = Conv1D(128, kernel_size=3, padding='same', activation='linear')(block1)
    block1 = BatchNormalization()(block1)
    block1 = PReLU()(block1)

    resize_emb = Conv1D(128, kernel_size=3, padding='same', activation='linear')(embed)
    resize_emb = PReLU()(resize_emb)

    block1_output = add([block1, resize_emb])
    block1_output = MaxPooling1D(pool_size=10)(block1_output)

    block2 = Conv1D(128, kernel_size=4, padding='same', activation='linear')(block1_output)
    block2 = BatchNormalization()(block2)
    block2 = PReLU()(block2)
    block2 = Conv1D(128, kernel_size=4, padding='same', activation='linear')(block2)
    block2 = BatchNormalization()(block2)
    block2 = PReLU()(block2)

    block2_output = add([block2, block1_output])
    block2_output = MaxPooling1D(pool_size=10)(block2_output)

    block3 = Conv1D(128, kernel_size=5, padding='same', activation='linear')(block2_output)
    block3 = BatchNormalization()(block3)
    block3 = PReLU()(block3)
    block3 = Conv1D(128, kernel_size=5, padding='same', activation='linear')(block3)
    block3 = BatchNormalization()(block3)
    block3 = PReLU()(block3)

    output = add([block3, block2_output])
    maxpool = GlobalMaxPooling1D()(output)
    average = GlobalAveragePooling1D()(output)

    x = concatenate([maxpool, average])

    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(x))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output = Dense(class_num, activation="softmax")(x)

    model = Model(inputs=content, outputs=output)
    #model = multi_gpu_model(model, 2)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model





