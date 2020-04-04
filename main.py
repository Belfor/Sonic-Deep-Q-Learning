#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 21:34:32 2020

@author: belfor
"""

import json
from SonicAgent import SonicAgent
from utils.utils import make_env
from utils.levelManager import LevelManager
from retro_contest.local import make
from pathlib import Path
from gym.wrappers import Monitor
from LinearDecaySchedule import LinearDecaySchedule

parameter = json.load(open("sonic.json", 'r'))
agent = parameter["agent"]
enviroment = parameter["enviroment"]

def training(env, sonic):
    
    total_reward = 0.0
   
    file = Path('models/sonic_model_final.h5')
    if file.is_file():
        sonic.load_model('models/sonic_model_final.h5')

    obs = env.reset()
    for episodes in range(agent["episodes"]):
        done = False
        print("Empieza Episodio #{}".format(episodes + 1))
        while not done:
            action = sonic.policy(obs)
            next_obs, reward, done, info = env.step(action)
            sonic.save_memory(obs,action,reward,next_obs,done)
            if enviroment["render"]:
                env.render()
            if (len(sonic.memory) > agent["batch_replay"]):
                sonic.replay_and_learn()
            
            total_reward += reward           
            obs = next_obs

            
        if episodes % 50 == 0:
            sonic.save_model('models/' + str(episodes) + '_sonic_model.h5')
        sonic.update_target_model()
        print("Episodio #{} finalizado con recompensa {}".format(episodes + 1, total_reward))
        obs = env.reset()
        total_reward = 0.0
        
    env.close()
    sonic.save_model('models/sonic_model_final.h5')
       
def validation(env, sonic):
    env = Monitor(env, './video',force=True)
    sonic.load_model('sonic_model_final.h5')
    obs = env.reset()
    while True:
        action = sonic.policy(obs)
        next_obs, reward, done, info = env.step(action)
        print("Para la accion #{} la recompensa es {}".format(action, reward))
        env.render()
        obs = next_obs
        if done:
            obs = env.close()
            
if __name__ == '__main__':
    
    
    levelManager = LevelManager('training-validation/sonic-train.csv',"training-validation/sonic-train.csv") 
    levelManager.listMap()
    levels = []
    if (enviroment["all_maps"] == False):
        levels = enviroment["selected_maps"]
    else:
        levels = [0..levelManager.size_maps_training() - 1]
    
    max_steps = agent["steps_episode"] * agent["episodes"] * len(levels)
    epsilon_initial = agent["epsilon_max"]
    epsilon_final = agent["epsilon_min"]
    max_num_episodes =agent["episodes"]
    setps_per_episode = agent["steps_episode"]
    linear_schedule = LinearDecaySchedule( epsilon_initial,
                                           epsilon_final, 
                                           len(levels) * max_num_episodes * setps_per_episode)
    sonic = SonicAgent(linear_schedule,enviroment["training"])
      
    for i in levels:
        print("Mapa #{} Comienza...".format(i))
        level = levelManager.getMap(i)
        env = make(level[0],level[1])

        if (enviroment["training"]):
            env = make_env(env)
            print(env.action_space)
            training(env,sonic)
        else:
            env = make_env(env, noop_rest=False)
            validation(env,sonic)
        print("Mapa #{} Finaliza...".format(i))