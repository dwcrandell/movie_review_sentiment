#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 15:02:12 2017

@author: douglascrandell
"""

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb

#IMDB Dataset loading
train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000, valid_portion=0.1)

#Split training and testing sets into reviews and labels
trainX, trainY = train
testX, testY = test

#Data preprocessing
#Sequence padding
#Vectorize inputs
#value must be float
trainX = pad_sequences(trainX, maxlen=100, value=0.)
testX = pad_sequences(testX, maxlen=100, value=0.)

#Convert labels to binary vectors
#1-positive, 0-negative
trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)

#Network building
#input layer
#specify input shape
#Batchsize = None
#Length = 100 (maxlen)
net = tflearn.input_data([None, 100])
#Embedding layer
#input_dim = n_words
#output_dim = 
net = tflearn.embedding(net, input_dim=10000, output_dim=128)
#lstm layer--allows net to remember data from the beginning of the sequences
#dropout randomly turns on and off pathways in network to prevent overfitting
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 2, activation='softmax')
#adam does gradient descent
#loss helps determine diference between expected output and predicted output
net = tflearn.regression(net, optimizer='adam', learning_rate=0.0001,
                         loss='categorical_crossentropy')

#Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
          batch_size=32)