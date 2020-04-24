# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 17:42:13 2020

@author: Belfor
"""
from keras.models import Model
from keras.layers import Dense,Flatten,Conv2D,Input,Lambda, Add,MaxPooling2D,BatchNormalization,Activation
from keras import backend as K
import keras

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
    

    def dueling_dqn(self):
        inputs = Input(self.input_shape)
        x = Conv2D(32,kernel_size = (6,6),strides = (1,1),activation='relu')(inputs)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = Conv2D(64,kernel_size = (5,5),strides = (1,1),activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = Conv2D(64,kernel_size = (3,3),strides = (1,1),activation='relu')(x)
        x = Flatten()(x)

        advantage = Dense(256,activation='relu')(x)
        advantage = Dense(self.output_shape)(advantage)
        advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(self.output_shape,))(advantage)
             
        value = Dense(256,activation='relu')(x)
        value = Dense(1 ,activation='linear')(value)
        value = Lambda(lambda s: K.expand_dims(s[:, 0], axis=-1), output_shape=(self.output_shape,))(value)
            
        x = Add()([value, advantage])
     
            
        model = Model(inputs=inputs,outputs=x)
        model.summary()
        model.compile(loss=keras.losses.mean_squared_error,optimizer=keras.optimizers.Adam(lr=self.lr))
        
        return model

    def dueling_dqn2(self):
        inputs = Input(self.input_shape)
        x = Conv2D(16,kernel_size = (8,8),strides = (1,1),activation='relu')(inputs)
           
        x = Conv2D(32,kernel_size = (4,4),strides = (1,1),activation='relu')(x)
           
      #  x = Conv2D(64,kernel_size = (3,3),strides = (1,1),activation='relu')(x)
        x = Flatten()(x)

        advantage = Dense(256,activation='relu')(x)
        advantage = Dense(self.output_shape)(advantage)
        advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(self.output_shape,))(advantage)
                
        value = Dense(256,activation='relu')(x)
        value = Dense(1 ,activation='linear')(value)
        value = Lambda(lambda s: K.expand_dims(s[:, 0], axis=-1), output_shape=(self.output_shape,))(value)
                
        x = Add()([value, advantage])
        
                
        model = Model(inputs=inputs,outputs=x)
        model.summary()
        model.compile(loss=keras.losses.mean_squared_error,optimizer=keras.optimizers.Adam(lr=self.lr))
            
        return model


    def dueling_dqn3(self):
        inputs = Input(self.input_shape)
        x = Conv2D(32,kernel_size = (8,8),strides = (4,4),padding = 'valid',activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Activation('elu')(x)
        x = Conv2D(64,kernel_size = (4,4),strides = (2,2),activation='relu')(x)
        x = BatchNormalization()(x)
        x = Activation('elu')(x)
        x = Conv2D(128,kernel_size = (4,4),strides = (2,2),activation='relu')(x)
        x = BatchNormalization()(x)
        x = Activation('elu')(x)
        x = Flatten()(x)
 
        advantage = Dense(512,activation='relu')(x)
        advantage = Dense(self.output_shape)(advantage)
        advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(self.output_shape,))(advantage)
                
        value = Dense(512,activation='relu')(x)
        value = Dense(1 ,activation='linear')(value)
        value = Lambda(lambda s: K.expand_dims(s[:, 0], axis=-1), output_shape=(self.output_shape,))(value)
                
        x = Add()([value, advantage])

                
        model = Model(inputs=inputs,outputs=x)
        model.summary()
        model.compile(loss=keras.losses.mean_squared_error,optimizer=keras.optimizers.Adam(lr=self.lr),metrics=["accuracy"])
            
        return model
        
