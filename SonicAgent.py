# -*- coding: utf-8 -*-

#import tensorflow as tf
import numpy as np

import json
import random 
import math
import keras
from pathlib import Path
from DQN import DQN
from collections import deque
from keras import backend as k
import tensorflow as tf
from PriorityExperienceMemory import PriorityExperienceMemory


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
        self.memory = PriorityExperienceMemory(parameter["agent"]["max_memory"])
      
        self.batch_size = parameter["agent"]["batch_size"]  
                  
        self.training = training
        self.path_model=parameter["agent"]["models"] 
        
        # Number of future states used to form memories in n-step dqn 
        self.n_step = parameter["agent"]["n_step"] 

        #The support for the value distribution. Set to 51 for C51
        self.num_atoms = 51
        
        # Break the range of rewards into 51 uniformly spaced values (support)
        self.v_max = 5 * self.n_step # Rewards are clipped to -20, 20
        self.v_min = -5 * self.n_step
        self.delta_z = (self.v_max - self.v_min) / float(self.num_atoms - 1)
        self.z = [self.v_min + i * self.delta_z for i in range(self.num_atoms)]
        
    def createModel(self,env,file_name_h5 = None):
        self.action_space = env.action_space    
        self.observation_shape =  env.observation_space.shape   

        dqn = DQN(self.observation_shape, self.action_space.n, self.lr)
           
        self.model = dqn.rainbowDqn()
        self.target_model = dqn.rainbowDqn()  
       
        if file_name_h5 != None:
            file = Path(self.path_model + file_name_h5)
            
            if file.is_file():
                self.load_model(self.path_model + file_name_h5)
                
                
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        
    def get_action(self, obs):
        obs = obs[np.newaxis,:]
        action_distributions = self.model.predict(obs)

        weighted_dists = np.multiply(np.vstack(action_distributions), np.array(self.z))
        
        # Sum the weighted values from each distribution and find the one with 
        # the largest expected value.
        avg_dist_values = np.sum(weighted_dists, axis=1)
        #print(avg_dist_values);
        action = np.argmax(avg_dist_values)
            
        return action
    
    def save_memory(self, obs, action, reward, next_obs, done):
        _, _, error = self.target_rewards([(obs, action, reward, next_obs, done)])
        self.memory.add((obs, action, reward, next_obs, done), error[0])
       
    def update_memory(self, size, indicies,error):
        for i in range(size):
            idx = indicies[i]
            self.memory.update(idx, error[i])
    
    def replay_and_learn(self):
       
        mini_batch, indicies = self.memory.sample(self.batch_size)
     
        inputs, targets, error = self.target_rewards(mini_batch)
        self.update_memory(self.batch_size,indicies,error)  
        self.model.train_on_batch(inputs,targets)
            
            
    def target_rewards(self, batch):
        batch_size = len(batch)
        inputs =  np.zeros(((batch_size,) + self.observation_shape)) 
        targets = np.zeros(((batch_size,) + self.observation_shape)) 
        actions, rewards, dones = [], [], []
        
        for i in range(batch_size):
            obs, action, reward, next_obs, done = batch[i]
            inputs[i] = obs[np.newaxis,:]
            actions.append(action)
            rewards.append(reward)
            targets[i] = next_obs[np.newaxis,:]
            dones.append(done)

        m_prob = [np.zeros((batch_size, self.num_atoms)) for i in range(self.action_space.n)]

        model_dist = self.model.predict(targets)
        target_dist = self.target_model.predict(targets)

        # Stack all of the distributions from all actions and all inputs 
        # (batch_size * action_size, num_atoms)
        stacked_dist = np.vstack(model_dist)

        # Multiply all distributions by value ranges, then sum each distribution,
        # leaving a (batch_size, num_actions) matrix for each input in the batch.
        q_values = np.sum(np.multiply(stacked_dist, np.array(self.z)), axis = 1)
        q_values = q_values.reshape((len(batch), self.action_space.n), order='F')

        next_act_idx = np.argmax(q_values, axis=1)

        error = []

        for i in range(batch_size):
            prev_q = np.sum(model_dist[actions[i]][i])

            if dones[i]:        
                Tz = min(self.v_max, max(self.v_min, rewards[i]))
                # Determine which segments the reward belongs to
                bj = math.floor((Tz - self.v_min) / self.delta_z) 
                segment_l = math.floor(bj)
                segment_u = math.ceil(bj)
                # Convert the rough segment value to indicies for upper and lower.
                # Add the portion of the range (0-1) which belongs to each segment. 
                m_prob[actions[i]][i][int(segment_l)] += (segment_u - bj)
                m_prob[actions[i]][i][int(segment_u)] += (bj - segment_l)
            else:
                for atom in range(self.num_atoms):
                    # Ensure rewards are clipped.
                    Tz = min(self.v_max, max(self.v_min, rewards[i] + self.gamma * self.z[atom])) 
                    # Determine which segments the reward belongs to
                    bj = math.floor((Tz - self.v_min) / self.delta_z) 
                    segment_l = math.floor(bj)
                    segment_u = math.ceil(bj)
                    lower_prob = target_dist[next_act_idx[i]][i][atom] * (segment_u - bj)
                    upper_prob = target_dist[next_act_idx[i]][i][atom] * (bj - segment_l)
                    m_prob[actions[i]][i][int(segment_l)] += lower_prob
                    m_prob[actions[i]][i][int(segment_u)] += upper_prob

            q_update = np.sum(m_prob[actions[i]][i])            
            error.append(abs(prev_q - q_update))
        
        return inputs, m_prob, error
              
    def save_model(self,filename):
        self.model.save_weights(filename)
        
    def load_model(self,filename):
        self.model.load_weights(filename)
        self.update_target_model()


