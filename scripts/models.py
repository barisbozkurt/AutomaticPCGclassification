#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keras model definitions

@author: Baris Bozkurt
"""
import keras
from keras.models import Sequential
from keras import regularizers
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

def loadModel(modelName, input_shape, num_classes):
    '''Loading one of the Keras models defined below(within this function)

    To specify a new model, please consider adding a new
        elif modelName=="..." block and place the specification within that block

    Args:
        name (str): Name of the model. Ex: 'uocSeq0','uocSeq1', etc.
        input_shape (tuple): Shape of the input vector as expected in definition
            of the input layer of a Keras model
            For spectrogram-like features the following shape info is expected:
            (timeDimension, frequencyDimension, 1)
        num_classes (int): Number of classes
        The last two arguments are used to define the sizes of input and output
            layers of Keras models
    '''
    model=None
    if modelName=="uocSeq0":#Model with single convolutional layer
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(32,kernel_regularizer=regularizers.l1(0.0005), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                      metrics=['accuracy'])#learning rate default: 0.001

    elif modelName=="uocSeq1":#Model with two convolutional layers
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='valid',activation='relu', input_shape=input_shape))
        model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(16,kernel_regularizer=regularizers.l1(0.0002)))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                      metrics=['accuracy'])#learning rate default: 0.001

    elif modelName=="uocSeq2":
        model = Sequential()#Model with 4 convolutional layers
        model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu', input_shape=input_shape))
        model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
        model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(32,kernel_regularizer=regularizers.l1(0.0001), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(16,kernel_regularizer=regularizers.l1(0.0001), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=sgd)

    return model
