# Standard Libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dropout, GRU, Bidirectional, SeparableConv2D, ConvLSTM2D, BatchNormalization, Add, ReLU, Layer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, TimeDistributed, LSTM, Flatten, Dense, Reshape, Input, MultiHeadAttention

# Keras backend
from keras import backend as K
from keras.callbacks import ModelCheckpoint

import numpy as np
import tensorflow as tf

class MultiHeadSelfAttention(Layer):
    def __init__(self, num_heads, num_filters, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.num_filters = num_filters  # Total number of filters

    def build(self, input_shape):
        assert len(input_shape) == 4, 'The input tensor  4D'
        assert input_shape[-1] == self.num_filters,

        self.depth = self.num_filters // self.num_heads
        assert self.depth * self.num_heads == self.num_filters, 'The number of filters must be divisible by the number of heads'

        # Dense layers for Q, K, V, and final output
        self.Wq = Dense(self.num_filters, use_bias=False)
        self.Wk = Dense(self.num_filters, use_bias=False)
        self.Wv = Dense(self.num_filters, use_bias=False)
        self.linear = Dense(self.num_filters)

        super(MultiHeadSelfAttention, self).build(input_shape)

    def split_heads(self, x):
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        q = self.Wq(inputs)
        k = self.Wk(inputs)
        v = self.Wv(inputs)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        scaled_attention_logits = tf.matmul(q, k, transpose_b=True)
        scaled_attention_logits /= tf.math.sqrt(tf.cast(self.depth, tf.float32))
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        output = tf.matmul(attention_weights, v)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, tf.shape(inputs))

        output = self.linear(output)

        return output

    def get_config(self):
        config = super(MultiHeadSelfAttention, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'num_filters': self.num_filters,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    class ConvLSTMCell:
        def __init__(self, input_channels, hidden_channels, kernel_size):
            self.input_channels = input_channels
            self.hidden_channels = hidden_channels
            self.kernel_size = kernel_size
            self.padding = kernel_size // 2

            self.Wxi = tf.Variable(tf.random.normal([kernel_size, kernel_size, input_channels, hidden_channels]))
            self.Whi = tf.Variable(tf.random.normal([kernel_size, kernel_size, hidden_channels, hidden_channels]))
            self.Wxf = tf.Variable(tf.random.normal([kernel_size, kernel_size, input_channels, hidden_channels]))
            self.Whf = tf.Variable(tf.random.normal([kernel_size, kernel_size, hidden_channels, hidden_channels]))
            self.Wxc = tf.Variable(tf.random.normal([kernel_size, kernel_size, input_channels, hidden_channels]))
            self.Whc = tf.Variable(tf.random.normal([kernel_size, kernel_size, hidden_channels, hidden_channels]))
            self.Wxo = tf.Variable(tf.random.normal([kernel_size, kernel_size, input_channels, hidden_channels]))
            self.Who = tf.Variable(tf.random.normal([kernel_size, kernel_size, hidden_channels, hidden_channels]))

            self.bi = tf.Variable(tf.zeros([hidden_channels]))
            self.bf = tf.Variable(tf.zeros([hidden_channels]))
            self.bc = tf.Variable(tf.zeros([hidden_channels]))
            self.bo = tf.Variable(tf.zeros([hidden_channels]))

        def forward(self, input, hidden_state):
            hidden, cell_state = hidden_state

            input_gate = tf.sigmoid(
                tf.nn.conv2d(input, self.Wxi, strides=[1, 1, 1, 1], padding='SAME') +
                tf.nn.conv2d(hidden, self.Whi, strides=[1, 1, 1, 1], padding='SAME') +
                self.bi
            )

            forget_gate = tf.sigmoid(
                tf.nn.conv2d(input, self.Wxf, strides=[1, 1, 1, 1], padding='SAME') +
                tf.nn.conv2d(hidden, self.Whf, strides=[1, 1, 1, 1], padding='SAME') +
                self.bf
            )

            cell_gate = tf.tanh(
                tf.nn.conv2d(input, self.Wxc, strides=[1, 1, 1, 1], padding='SAME') +
                tf.nn.conv2d(hidden, self.Whc, strides=[1, 1, 1, 1], padding='SAME') +
                self.bc
            )

            output_gate = tf.sigmoid(
                tf.nn.conv2d(input, self.Wxo, strides=[1, 1, 1, 1], padding='SAME') +
                tf.nn.conv2d(hidden, self.Who, strides=[1, 1, 1, 1], padding='SAME') +
                self.bo
            )

            cell_state = forget_gate * cell_state + input_gate * cell_gate
            hidden = output_gate * tf.tanh(cell_state)

            return hidden, cell_state

    class ConvLSTM:
        def __init__(self, input_channels, hidden_channels, kernel_size):
            self.input_channels = input_channels
            self.hidden_channels = hidden_channels
            self.kernel_size = kernel_size

            self.cell = ConvLSTMCell(input_channels, hidden_channels, kernel_size)

        def forward(self, inputs):
            batch_size, time_steps, height, width, _ = inputs.shape

            hidden_state = (
                tf.zeros([batch_size, height, width, self.hidden_channels]),
                tf.zeros([batch_size, height, width, self.hidden_channels])
            )

            outputs = []
            for t in range(time_steps):
                hidden_state = self.cell.forward(inputs[:, t, :, :, :], hidden_state)
                outputs.append(hidden_state[0])

            outputs = tf.stack(outputs, axis=1)
            return outputs