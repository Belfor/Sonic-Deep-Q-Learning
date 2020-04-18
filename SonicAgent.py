# -*- coding: utf-8 -*-

#import tensorflow as tf
import numpy as np

import json
import random 
import keras
from pathlib import Path
from DQN import DQN
from collections import deque
from keras import backend as k
import tensorflow as tf


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



class SonicAgent():
    def __init__(self,training = False):      
           
        parameter = json.load(open('sonic.json', 'r'))
       
        self.lr =  parameter["agent"]["learning_rate"]  
        self.gamma = parameter["agent"]["gamma"]  
        self.memory = deque(maxlen = parameter["agent"]["max_memory"] )
        self.batch_size = parameter["agent"]["batch_size"]  
                  
        self.training = training
        self.path_model=parameter["agent"]["models"] 
        
        
    def createModel(self,env,file_name_h5 = None):
        self.action_space = env.action_space    
        self.observation_shape =  env.observation_space.shape   

        dqn = DQN(self.observation_shape, self.action_space.n, self.lr)
           
        self.model = dqn.createModel()
        self.target_model = dqn.createModel()
      
       
        if file_name_h5 != None:
            file = Path(self.path_model + file_name_h5)
            
            if file.is_file():
                self.load_model(self.path_model + file_name_h5)
                
                
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        
    def policy(self, obs, epsilon = 0):
        obs = obs[np.newaxis,:]
      
        if np.random.random() < epsilon and self.training:
            action = random.choice([a for a in range(self.action_space.n)])
        else:
            action = np.argmax(self.model.predict(obs))
            
        return action
    
    def save_memory(self, obs, action, reward, next_obs, done):
        self.memory.append((obs, action, reward, next_obs, done))
       
    def replay_and_learn(self):
       
        mini_batch = random.sample(self.memory, self.batch_size)
        inputs = np.zeros(((self.batch_size,) + self.observation_shape))
        targets = np.zeros((self.batch_size, self.action_space.n))
        
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
            return reward + self.gamma * np.amax(self.target_model.predict(next_obs)[0])
       
              
    def save_model(self,filename):
        self.model.save_weights(filename)

     
        
    def load_model(self,filename):
        self.model.load_weights(filename)
        self.update_target_model()


