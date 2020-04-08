# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 17:42:13 2020

@author: Belfor
"""
from keras.models import Model
from keras.layers import Dense,Flatten,Conv2D,Input
from keras import backend as tf

class DQN:
    

    def __init__(self, input_shape , output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
    
    def createModel(self):
        inputs = Input(self.input_shape)
        x = Conv2D(32,kernel_size = (8,8),strides = (4,4),padding = 'valid',activation='relu')(inputs)
        x = Conv2D(64,kernel_size = (4,4),strides = (2,2),activation='relu')(x)
        x = Conv2D(128,kernel_size = (4,4),strides = (2,2),activation='relu')(x)
        x = Flatten()(x)
        x = Dense(512,activation='relu')(x)
        x = Dense(self.output_shape ,activation='linear')(x)
        return Model(inputs=inputs,outputs=x)