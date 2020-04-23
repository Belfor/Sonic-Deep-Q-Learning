#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 10:02:19 2020

@author: Belfor
"""
import csv
import json
import numpy as np
from collections import deque
from SonicAgent import SonicAgent
from utils.utils import make_env
from utils.levelManager import LevelManager
from retro_contest.local import make

from utils.LinearDecaySchedule import LinearDecaySchedule
from tensorboardX import SummaryWriter

parameter = json.load(open('sonic.json', 'r'))

agent = parameter["agent"]
enviroment = parameter["enviroment"]

steps_episode = agent["steps_episode"]
writer = SummaryWriter(agent["logs"])
update_target_freq = agent["update_target_freq"]
timestep_per_train = agent["timestep_per_train"]
max_num_episodes =agent["episodes"]

n_step = agent["n_step"]
n_step_rewards = deque(maxlen=n_step) 
n_step_exp = deque(maxlen=n_step)

def training(env,sonic,global_step_num):
    epsilon = LinearDecaySchedule(1,0.1,max_num_episodes * steps_episode)
    total_reward = 0.0
    
    sonic.createModel(env)
     
    for episodes in range(max_num_episodes):
        n_step_rewards.clear()
        n_step_exp.clear()
        obs = env.reset()
        done = False
        total_reward = 0.0
        best_reward = 0.0
        steps = 0
        print("Empieza Episodio #{}".format(episodes + 1))
        while not done:
            action = sonic.get_action(obs, epsilon(global_step_num))      
            next_obs, reward, done, _ = env.step(action)

            n_step_rewards.appendleft(reward)
            n_step_exp.append((next_obs, action, done))

            if (len(n_step_rewards) >= sonic.n_step):
                total_n_reward = 0
                for i in range(len(n_step_rewards)):
                    total_n_reward += ((sonic.gamma ** i) * n_step_rewards[i])

                obs, action, _ = n_step_exp[-1]
                n_state_obs, _, n_state_done = n_step_exp[0]
                sonic.save_memory(obs,action,total_n_reward,n_state_obs,n_state_done)
                        
            obs = next_obs
            total_reward += reward        
           
                 
            if enviroment["render"]:
                env.render()
            
                
            global_step_num += 1
            steps += 1
            if (global_step_num > agent["max_memory"]):
                if ((global_step_num % update_target_freq) == 0):
                    sonic.update_target_model()

                if ((global_step_num % timestep_per_train) == 0):
                    sonic.replay_and_learn()
        
        if (global_step_num > agent["max_memory"]):
            if ((episodes % 50) == 0):
                sonic.update_target_model()

            if ((episodes % 50) == 0):
                sonic.replay_and_learn()
            
        if (total_reward > best_reward):
            best_reward = total_reward
            sonic.save_model('models/best_reward_sonic.h5')
   
       
        print("Episodio #{} finalizado con recompensa {}".format(episodes + 1, total_reward))
        writer.add_scalar('rewards', total_reward, global_step_num)
           
    env.close()
    sonic.save_model('models/sonic_model_final.h5')
    writer.close()
    
       
    
    
if __name__ == '__main__':
    
    global_step_num = 0
    levelManager = LevelManager('training-validation/sonic-train.csv',"training-validation/sonic-train.csv") 
    levelManager.listMap()
    levels = []
    if (enviroment["all_maps"] == False):
        levels = enviroment["selected_maps"]
    else:
        levels = [0..levelManager.size_maps_training() - 1]

    setps_per_episode = agent["steps_episode"]
    
    sonic = SonicAgent(True)
      
    for i in levels:
        print("Mapa #{} Comienza...".format(i))
        level = levelManager.getMap(i)
        env = make(level[0],level[1])
        env = make_env(env,allow_backtracking=True)
        training(env,sonic,global_step_num)

        print("Mapa #{} Finaliza...".format(i))