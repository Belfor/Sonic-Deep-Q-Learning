#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 10:02:19 2020

@author: Belfor
"""

import json
import numpy as np
from SonicAgent import SonicAgent
from utils.utils import make_env
from utils.levelManager import LevelManager
from retro_contest.local import make
from LinearDecaySchedule import LinearDecaySchedule

from tensorboardX import SummaryWriter

parameter = json.load(open('sonic.json', 'r'))

agent = parameter["agent"]
enviroment = parameter["enviroment"]

steps_episode = agent["steps_episode"]
writer = SummaryWriter(agent["logs"])
update_target_freq = agent["update_target_freq"]
timestep_per_train = agent["timestep_per_train"]
max_num_episodes =agent["episodes"]

def training(env,sonic,global_step_num,epsilon_decay):
    
    total_reward = 0.0
    
    sonic.createModel(env)
     
    for episodes in range(max_num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        print("Empieza Episodio #{}".format(episodes + 1))
        while not done:
            action = sonic.policy(obs,epsilon_decay(global_step_num))
            writer.add_scalar("epsilon", epsilon_decay(global_step_num),global_step_num)
            
            next_obs, reward, done, _ = env.step(action)
            
            sonic.save_memory(obs,action,reward,next_obs,done)
                        
            obs = next_obs
            total_reward += reward        
           
                 
            if enviroment["render"]:
                env.render()
                
                
            global_step_num += 1
            steps += 1
            
            if ((global_step_num % update_target_freq) == 0):
                 sonic.update_target_model()

            if ((global_step_num % timestep_per_train) == 0):
                sonic.replay_and_learn()
    
        global_step_num += steps_episode - steps
        

        if ((episodes % 100) == 0):
            sonic.update_target_model()

        if ((episodes % 100) == 0):
            sonic.replay_and_learn()
   
       
        print("Episodio #{} finalizado con recompensa {}".format(episodes + 1, total_reward))
        writer.add_scalar('ep_reward', total_reward, global_step_num)
      
      
        
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
    
    epsilon_initial = agent["epsilon_max"]
    epsilon_final = agent["epsilon_min"]
 
    setps_per_episode = agent["steps_episode"]
    linear_schedule = LinearDecaySchedule( epsilon_initial,
                                           epsilon_final, 
                                           len(levels) * max_num_episodes * setps_per_episode)
    
    sonic = SonicAgent(True)
      
    for i in levels:
        print("Mapa #{} Comienza...".format(i))
        level = levelManager.getMap(i)
        env = make(level[0],level[1])
        env = make_env(env,allow_backtracking=True)
        training(env,sonic,global_step_num,linear_schedule)

        print("Mapa #{} Finaliza...".format(i))