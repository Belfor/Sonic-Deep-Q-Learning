#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 21:34:32 2020

@author: belfor
"""


import random
import csv
from SonicAgent import SonicAgent
from utils.utils import make_env
from retro_contest.local import make

TIMESTEPS_PER_EPISODE = 4500
EPISODES = 100

class LoadEnv():
    
    def __init__(self,training_file, validation_file):
    
        self.training = self.readFile(training_file)
        self.validation = self.readFile(validation_file)
    
    def readFile(self,file):
        env = []
        file =  open(file)
        reader = csv.reader(file,delimiter=',')
        for row in reader:
            env.append(row)
        return env
    
    def size_maps_training(self):
        return len(self.training)
    
    def listMap(self, training = True):
        if(training):
            for i in range(1,len(self.training) -1):
                print("{} -> {} - {}".format(i,self.training[i][0],self.training[i][1]))
        else:
            for i in range(1,len(self.validation) -1):
                print("{} -> {} - {}".format(i,self.validation[i][0],self.validation[i][1]))
    def loadMap(self,level,training = True):
        if(training):
            return self.loadEnv(self.training[level][0],self.training[level][1])
        
        return self.loadEnv(self.validation[level][0],self.validation[level][1])
    
    def loadRandomEnv(self, test = None):
        if not test:
            rnd = random.randint(1,len(self.training) - 1)
        else:
            rnd = random.randint(1,len(self.validation) - 1)
        return make(self.training[rnd][0],self.training[rnd][1])
        
    def loadEnv(self,level,state):
        return make(level,state)

def training(env):
    
    total_reward = 0.0
    reward_per_episode = []
    env = make_env(env)

    sonic = SonicAgent(env,TIMESTEPS_PER_EPISODE * EPISODES)
    sonic.load_model('sonic_model_final.h5')
    obs = env.reset()
    for episodes in range(EPISODES):
        done = False
        print("Empieza Episodio #{}".format(episodes + 1))
        while not done:
            action = sonic.policy(obs)
            next_obs, reward, done, info = env.step(action)
            sonic.save_memory(obs,action,reward,next_obs,done)
            if (len(sonic.memory) > sonic.batch_size):
                sonic.replay_and_learn()
            total_reward += reward
            obs = next_obs
            if done:
                break
            
        if episodes % 50 == 0:
            sonic.save_model('models/' + str(episodes) + '_sonic_model.h5')
        reward_per_episode.append(total_reward)
        sonic.update_target_model()
        print("Episodio #{} finalizado con recompensa {}".format(episodes + 1, total_reward))
        obs = env.reset()
        total_reward = 0.0
        
    print(reward_per_episode)
    env.close()
    sonic.save_model('sonic_model_final.h5')
       
def validation(env):
    env = make_env(env)
    sonic = SonicAgent(env,TIMESTEPS_PER_EPISODE* EPISODES, True)
    sonic.load_model('models/650_sonic_model.h5')
    obs = env.reset()
    while True:
        action = sonic.policy(obs)
        #action = random.choice([a for a in range(env.action_space.n)])
        next_obs, reward, done, info = env.step(action)
        print("Para la accion #{} la recompensa es {}".format(action, reward))
        env.render()
        obs = next_obs
        if done:
            obs = env.close()
            
if __name__ == '__main__':
      loadEnv = LoadEnv('training-validation/sonic-train.csv',"training-validation/sonic-train.csv") 
      loadEnv.listMap()
      for i in range(1, loadEnv.size_maps_training() - 1):
          env = loadEnv.loadMap(i)
          training(env)