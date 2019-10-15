#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
Author:
    Congqing He,hecongqing@hotmail.com
"""

from tensorflow.keras.layers import (LSTM, GRU, Bidirectional)


class RNNType:
    RNN = 'RNN'
    LSTM = 'LSTM'
    GRU = 'GRU'

    def str(cls):
        return ",".join([cls.RNN, cls.LSTM, cls.GRU])


class RNN:
    def __init__(self, rnn_type, bidirectional, hidden_size):
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        if bidirectional:
            if rnn_type == RNNType.GRU:
                self.rnn = GRU(hidden_size, activation='tanh', dropout=0.0, return_sequences=True)
            elif rnn_type == RNNType.LSTM:
                self.rnn = LSTM(hidden_size, activation='tanh', dropout=0.0, return_sequences=True)
            else:
                raise TypeError(
                    "Unsupported rnn init type: %s. Supported rnn type is: %s" % (
                        rnn_type, RNNType.str()))
        else:
            if rnn_type == RNNType.GRU:
                self.rnn = Bidirectional(GRU(hidden_size, activation='tanh', dropout=0.0, return_sequences=True))
            elif rnn_type == RNNType.LSTM:
                self.rnn = Bidirectional(LSTM(hidden_size, activation='tanh', dropout=0.0, return_sequences=True))
            else:
                raise TypeError(
                    "Unsupported rnn init type: %s. Supported rnn type is: %s" % (
                        rnn_type, RNNType.str()))

