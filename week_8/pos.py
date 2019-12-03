"""
Viktor Valastin
[NS] Week 8, POSTagger
"""

from functools import reduce
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional, Activation, Masking
import os
import sys
import tensorflow.keras as keras
import week_8.backstage.plots as plots
import week_8.backstage.data as data
import numpy as np
from week_8.backstage.data import *

train = data.load_pos_data('data/train')
test = data.load_pos_data('data/test')
vocab = data.load_vocabulary()
pos_vocab = data.pos_vocabulary

train_x = []
train_y = []
test_x = []
test_y = []

for sample in train:
    train_x.append([vocab[word] for word in sample.text])
    train_y.append([pos_vocab[tag] for tag in sample.labels])

for sample in test:
    test_x.append([vocab[word] for word in sample.text])
    test_y.append([pos_vocab[tag] for tag in sample.labels])

train_x = keras.preprocessing.sequence.pad_sequences(train_x, padding='post')
train_y = keras.preprocessing.sequence.pad_sequences(train_y, padding='post')
test_x = keras.preprocessing.sequence.pad_sequences(test_x, padding='post')
test_y = keras.preprocessing.sequence.pad_sequences(test_y, padding='post')


class POSTagger(keras.Model):

    def __init__(self):
        super(POSTagger, self).__init__()
#         Pre-trained Embedding
#         self.embedding = Embedding(input_dim=embedding_matrix.shape[0],
#                       output_dim=embedding_matrix.shape[1],
#                       input_length=train_x.shape[0],
#                       weights=[embedding_matrix],
#                       trainable=False,
#                       mask_zero=True)
#         Own Embedding
        self.model = [
            Embedding(input_dim=len(vocab.keys()) + 1,
                      output_dim=train_x.shape[1],
                      mask_zero=True),
            LSTM(train_x.shape[1], return_sequences=True),
            Dense(len(pos_vocab.keys()), activation='softmax')
        ]

    def call(self, x):
        return reduce(lambda previous, layer: layer(previous), self.model, x)


model = POSTagger()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

callbacks = [
    keras.callbacks.TensorBoard(
        log_dir=os.path.join("logs", timestamp()),
        histogram_freq=1,
        profile_batch=0)
]

model.fit(
    x=train_x,
    y=train_y,
    batch_size=10,
    epochs=30,
    callbacks=callbacks,
    validation_data=(test_x, test_y))
