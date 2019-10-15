#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
Author:
    Congqing He,hecongqing@hotmail.com
"""
import tensorflow as tf
from tensorflow.keras.layers import (Input,Embedding,GlobalMaxPooling1D, GlobalAveragePooling1D)
from ..layers.rnn import RNN


class TextRNN:
    def __init__(self, seq_length, embedding_weights, rnn_type, bidirectional, hidden_size, pool_type, dropout_rate, label_size, optimizer ):
        self.seq_length = seq_length
        self.embedding_weights = embedding_weights
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.pool_type = pool_type
        self.rnn =


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
        if self.pool_type == 'maxpool':
            max_pool = GlobalMaxPooling1D()(rnn)
        elif self.pool_type == 'avgpool':
            avg_pool = GlobalAveragePooling1D()(rnn)
        elif self.pool_type == 'attpool':
            att_pool =
