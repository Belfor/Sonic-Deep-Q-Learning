# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import keras
import json
import random 
import gym
from collections import deque
from keras.models import Model
from keras.layers import Dense,Flatten,Conv2D,Input
from keras.callbacks import TensorBoard


class SonicAgent():
    def __init__(self,epsilon_decay,training = False):      
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
        
        self.num_step = 0
        parameter = json.load(open("sonic.json", 'r'))
       
        self.n_actions = parameter["agent"]["n_actions"]     
        self.observation_shape = tuple(parameter["enviroment"]["shape"])   
        self.lr =  parameter["agent"]["learning_rate"]  
        self.gamma = parameter["agent"]["gamma"]  
        self.memory = deque(maxlen = parameter["agent"]["max_memory"] )
        self.batch_size = parameter["agent"]["batch_size"]  
        
        self.epsilon_decay = epsilon_decay
          
        self.training = training
        
        self.model = self.getModel(self.observation_shape ,self.n_actions)
        self.target_model = self.getModel(self.observation_shape ,self.n_actions)
        self.target_model.trainable = False
        self.model.summary()
        self.model.compile(loss=keras.losses.mean_squared_error,optimizer=keras.optimizers.Adam(lr=self.lr),metrics=["accuracy"])
        self.target_model.compile(loss=keras.losses.mean_squared_error,optimizer=keras.optimizers.Adam(lr=self.lr),metrics=["accuracy"])
        #self.tensorboard = TensorBoard(log_dir="logs/sonic")
        #self.tensorboard.set_model(self.model)
    def getModel(self,input_shape,output_shape):
        inputs = Input(input_shape)
        x = Conv2D(32,kernel_size = (8,8),strides = (4,4),padding = 'valid',activation='relu')(inputs)
        x = Conv2D(64,kernel_size = (4,4),strides = (2,2),activation='relu')(x)
        x = Conv2D(128,kernel_size = (4,4),strides = (2,2),activation='relu')(x)
        x = Flatten()(x)
        x = Dense(512,activation='relu')(x)
        x = Dense(output_shape,activation='linear')(x)
       
        return Model(inputs=inputs,outputs=x)
     
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        
    def policy(self, obs):
        obs = obs[np.newaxis,:]
        self.num_step += 1
        if np.random.random() < self.epsilon_decay(self.num_step) and self.training:
            action = random.randint(0,self.n_actions - 1)
        else:
            action = np.argmax(self.model.predict(obs))
        return action
    
    def save_memory(self, obs, action, reward, next_obs, done):
        self.memory.append((obs, action, reward, next_obs, done))
       
    def replay_and_learn(self):
        mini_batch = random.sample(self.memory, self.batch_size)
        
        inputs = np.zeros(((self.batch_size,) + self.observation_shape))
        targets = np.zeros((self.batch_size, self.n_actions))
        
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
       
              
    def save_model(self,filename):
        self.model.save_weights(filename)
        
    def load_model(self,filename):
        self.model.load_weights(filename)