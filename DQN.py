# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 17:42:13 2020

@author: Belfor
"""
from keras.models import Model
from keras.layers import Dense,Flatten,Conv2D,Input,Lambda, Add,GaussianDropout
from keras import backend as K
import keras
import math

class DQN:
    

    def __init__(self, input_shape , output_shape, lr, dueling = True):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.dueling = dueling
        self.lr = lr
    
    def createModel(self):
        inputs = Input(self.input_shape)
        x = Conv2D(32,kernel_size = (8,8),strides = (4,4),padding = 'valid',activation='relu')(inputs)
        x = Conv2D(64,kernel_size = (4,4),strides = (2,2),activation='relu')(x)
        x = Conv2D(128,kernel_size = (4,4),strides = (2,2),activation='relu')(x)
        x = Flatten()(x)
        if self.dueling:
            advantage = Dense(512,activation='relu')(x)
            advantage = Dense(self.output_shape)(advantage)
            advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(self.output_shape,))(advantage)
             
            value = Dense(512,activation='relu')(x)
            value = Dense(1 ,activation='linear')(value)
            value = Lambda(lambda s: K.expand_dims(s[:, 0], axis=-1), output_shape=(self.output_shape,))(value)
            
            x = Add()([value, advantage])
        else: 
            x = Dense(self.output_shape ,activation='linear')(x)
            
        model = Model(inputs=inputs,outputs=x)
        model.summary()
        model.compile(loss=keras.losses.mean_squared_error,optimizer=keras.optimizers.Adam(lr=self.lr),metrics=["accuracy"])
        
        return model
    
    def rainbowDqn(self):
        
        inputs = Input(self.input_shape)
        x = Conv2D(32,kernel_size = (8,8),strides = (4,4),padding = 'valid',activation='relu')(inputs)
        x = Conv2D(64,kernel_size = (4,4),strides = (2,2),activation='relu')(x)
        x = Conv2D(128,kernel_size = (4,4),strides = (2,2),activation='relu')(x)
        x = Flatten()(x)
     
        advantage = Dense(512,activation='relu')(x)
        advantage = Dense(self.output_shape)(advantage)
        advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(self.output_shape,))(advantage)
             
        value = Dense(512,activation='relu')(x)
        value = Dense(1 ,activation='linear')(value)
        value = Lambda(lambda s: K.expand_dims(s[:, 0], axis=-1), output_shape=(self.output_shape,))(value)
            
        x = Add()([value, advantage])
        
        noise_weights = Dense(self.output_shape, activation='linear')(x)
        noise = GaussianDropout(1 / math.sqrt(self.output_shape))(noise_weights)

        noisy_action = Add()([x, noise])

        # Each action has a distribution with 51 distinct values
        action_distributions = []
        for _ in range(self.output_shape):
            # 51 node output wiht Softmax activation to yield probabiltiies
            action_distributions.append(Dense(51, activation='softmax')(noisy_action))

        model = Model(inputs=inputs,outputs=action_distributions)
        model.summary()
        model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(lr=self.lr))

        return model