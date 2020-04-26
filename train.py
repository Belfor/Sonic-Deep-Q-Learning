#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 10:02:19 2020

@author: Belfor
"""

import json
import numpy as np
from SonicAgent import SonicAgent
from NStep import NStep
from utils.utils import make_env
from utils.levelManager import LevelManager
from utils.LinearDecaySchedule import LinearDecaySchedule
from retro_contest.local import make
from tensorboardX import SummaryWriter

parameter = json.load(open('sonic.json', 'r'))

agent = parameter["agent"]
enviroment = parameter["enviroment"]

steps_episode = agent["steps_episode"]
writer = SummaryWriter(agent["logs"])
update_target_freq = agent["update_target_freq"]
timestep_per_train = agent["timestep_per_train"]
max_num_episodes =agent["episodes"]


def training(sonic,global_step_num,epsilon_decay,level):
    n_step = NStep(sonic.n_step,sonic.gamma)
  
    env = make(level[0],level[1])
    env = make_env(env,allow_backtracking=False)
    sonic.load_network(env)
     
    for episodes in range(max_num_episodes):
        n_step.clear()
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        print("Empieza Episodio #{}".format(episodes + 1))
        while not done:
            action = sonic.get_action(obs,epsilon_decay(global_step_num))
            writer.add_scalar("epsilon", epsilon_decay(global_step_num),global_step_num)
            
            next_obs, reward, done, _ = env.step(action)
            #if done == True and total_reward < 9000:
            #    reward = -200
            n_step.append(reward, (next_obs,action,done))
            if n_step.is_last_step():
                first_obs,first_action,rewards_n_step,last_obs,last_done = n_step.calculate_rewards_nstep()
                sonic.save_memory(first_obs,first_action,rewards_n_step,last_obs,last_done)
                        
            obs = next_obs
            total_reward += reward        
           
                 
            if enviroment["render"]:
                env.render()
                
                
            global_step_num += 1
            steps += 1
            
            if global_step_num > sonic.max_memory * sonic.n_step * 2:
                if ((global_step_num % update_target_freq) == 0):
                    sonic.update_target_model()
                    

                if ((global_step_num % timestep_per_train) == 0):
                    sonic.learn()
        
        first_obs,first_action,rewards_n_step,last_obs,last_done = n_step.calculate_rewards_nstep()
        sonic.save_memory(first_obs,first_action,rewards_n_step,last_obs,last_done)
        
        print("Episodio #{} finalizado con recompensa {}".format(episodes + 1, total_reward))
        writer.add_scalar('ep_reward', total_reward, global_step_num)
        if (episodes + 1) %  300:
            sonic.save_model('models/sonic_model_'+ level[0] +'_' + level[1] + '.h5')
      
        
    sonic.save_model('models/sonic_model_'+ level[0] +'_' + level[1] + '.h5')
    env.close()
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
    
    sonic = SonicAgent(training=True)
      
    for i in levels:
        print("Mapa #{} Comienza...".format(i))
        level = levelManager.getMap(i)      
        training(sonic,global_step_num,linear_schedule,level)
        print("Mapa #{} Finaliza...".format(i))