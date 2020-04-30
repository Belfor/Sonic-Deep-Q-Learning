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
from PriorityExperienceMemory import PriorityExperienceMemory
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
    def __init__(self,lr = 0.0001, 
                gamma = 0.99, 
                priority_experience = True, 
                max_memory = 50000,
                batch_size = 64,
                n_step = 4,
                training = False):      
           
        self.training = training
       
        self.priority_experience = priority_experience
        self.max_memory = max_memory
        self.lr =  lr 
        self.gamma = gamma

        self.batch_size = batch_size

        if priority_experience:
            self.memory = PriorityExperienceMemory(max_memory)
        else:
            self.memory = deque(maxlen = max_memory )

        self.n_step = n_step
                  
        self.model = None
        self.target_model = None
        
    def load_network(self,env):
        self.action_space = env.action_space    
        self.observation_shape =  env.observation_space.shape   

        dqn = DQN(self.observation_shape, self.action_space.n, self.lr)
           
        self.model = dqn.createModel()
        self.target_model = dqn.createModel()
                
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        
    def get_action(self, obs, epsilon = 0):
        obs = obs[np.newaxis,:]
        if np.random.random() < epsilon and self.training:
            action = random.choice([a for a in range(self.action_space.n)])
        else:
            action = np.argmax(self.model.predict(obs))
        return action
    
    def save_memory(self, obs, action, reward, next_obs, done):
        if self.priority_experience:
            _, _, error = self.compute_Q([(obs, action, reward, next_obs, done)])
            self.memory.add((obs, action, reward, next_obs, done), error[0])
        else:
            self.memory.append((obs, action, reward, next_obs, done))

    def update_memory(self, size, indicies, errors):
        for i in range(size):
            self.memory.update(indicies[i],errors[i])

    def learn(self):
        if self.priority_experience:
            mini_batch, indicies = self.memory.sample(self.batch_size)
            inputs, targets, errors = self.compute_Q(mini_batch)
            self.update_memory(self.batch_size, indicies, errors)
        else:
            mini_batch = random.sample(self.memory, self.batch_size)
            inputs, targets, _ = self.compute_Q(mini_batch)
            
        self.model.train_on_batch(inputs,targets)
                       
    def compute_Q(self,batch):

        batch_size = len(batch)
        inputs = np.zeros(((batch_size,) + self.observation_shape))
        targets = np.zeros((batch_size, self.action_space.n))
      
        errors = []
        for i in range(batch_size):
            obs, action, reward, next_obs, done = batch[i]
            obs = obs[np.newaxis,:]
            next_obs = next_obs[np.newaxis,:]

            inputs[i] = obs   

            q = self.model.predict(obs)[0]

            q_val = self.model.predict(next_obs)[0][action]
            q_val_target = self.target_model.predict(next_obs)[0]

            if done:
                q[action] = reward
            else:           
                q[action] = reward + self.gamma * np.amax(q_val_target)
            
            targets[i,] = q
            errors.append(abs(q_val - q[action]))

        return inputs, targets, errors

                   
    def save_model(self, filename):
        self.model.save_weights(filename)

      
    def load_model(self, filename):
        self.model.load_weights(filename)
        self.update_target_model()


