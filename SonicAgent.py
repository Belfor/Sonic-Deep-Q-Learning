# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import keras
import random
from collections import deque
from keras.models import Model
from keras.layers import Dense,Flatten,Conv2D,Input
from keras.callbacks import TensorBoard
from LinearDecaySchedule import LinearDecaySchedule

class SonicAgent():
    def __init__(self,env,max_steps,test = None):      
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
        self.actions_shape = env.action_space.n      
        self.observation_shape = env.observation_space.shape
        self.num_step = 0
        self.lr = 0.0001
        self.gamma = 0.99
        self.epsilon_decay = LinearDecaySchedule(1.0, 0.1, max_steps)
        self.test = test
        
        self.batch_size = 64
        self.memory = deque(maxlen = 50000)
        
        self.model = self.getModel(env.observation_space.shape,env.action_space.n)
        self.target_model = self.getModel(env.observation_space.shape,env.action_space.n)
        self.target_model.trainable = False
        self.model.summary()
        self.model.compile(loss=keras.losses.mean_squared_error,optimizer=keras.optimizers.Adam(learning_rate=self.lr),metrics=["accuracy"])
        self.target_model.compile(loss=keras.losses.mean_squared_error,optimizer=keras.optimizers.Adam(learning_rate=self.lr),metrics=["accuracy"])
         
    def getModel(self,input_shape,output_shape):
        inputs = Input(input_shape)
        x = Conv2D(64,kernel_size = (4,4),strides = (2,2),padding = 'valid',activation='relu')(inputs)
        x = Conv2D(32,kernel_size = (4,4),strides = (2,2),activation='relu')(x)
        x = Conv2D(32,kernel_size = (3,3),strides = (1,1),activation='relu')(x)
        x = Flatten()(x)
        x = Dense(512,activation='relu')(x)
        x = Dense(output_shape,activation='linear')(x)
       
        return Model(inputs=inputs,outputs=x)
     
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        
    def policy(self, obs):
        obs = obs[np.newaxis,:]
        self.num_step += 1
        if np.random.random() < self.epsilon_decay(self.num_step) and not self.test:
            action = random.choice([a for a in range(self.actions_shape)])
        else:
            action = np.argmax(self.model.predict(obs))
        return action
    
    def save_memory(self, obs, action, reward, next_obs, done):
        self.memory.append((obs, action, reward, next_obs, done))
       
    def replay_and_learn(self):
        mini_batch = random.sample(self.memory, self.batch_size)
        
        inputs = np.zeros(((self.batch_size,) + self.observation_shape))
        targets = np.zeros((self.batch_size,self.actions_shape))
        
        for i in range (self.batch_size):
            obs, action, reward, next_obs, done = mini_batch[i]
            targets[i,action] = self.target_reward(obs, action, reward, next_obs, done)
            inputs[i] = obs[np.newaxis,:]
        self.model.train_on_batch(inputs,targets)
            
    def target_reward(self, obs, action, reward, next_obs, done):
       
        obs = obs[np.newaxis,:]
        next_obs = next_obs[np.newaxis,:]
       
        if done:
            return reward 
        else:
            return reward + self.gamma * np.argmax(self.target_model.predict(next_obs)[0])
       
        #self.model.fit(obs,y=td_target,epochs=1,verbose=0)
              
    def save_model(self,filename):
        self.model.save_weights(filename)
        
    def load_model(self,filename):
        self.model.load_weights(filename)