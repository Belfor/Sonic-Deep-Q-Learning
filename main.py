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

from tensorboardX import SummaryWriter

parameter = json.load(open("sonic.json", 'r'))
agent = parameter["agent"]
enviroment = parameter["enviroment"]

writer = SummaryWriter(agent["logs"])

def training(env, sonic,global_step_num):
    
    total_reward = 0.0
    sonic.createModel(env,'sonic_model_final.h5')
    

    obs = env.reset()
    for episodes in range(agent["episodes"]):
        done = False
        print("Empieza Episodio #{}".format(episodes + 1))
        while not done:
            action = sonic.policy(obs)
            writer.add_scalar("epsilon", sonic.epsilon_decay(sonic.num_step),sonic.num_step)
            next_obs, reward, done, info = env.step(action)
            sonic.save_memory(obs,action,reward,next_obs,done)
            
                        
            obs = next_obs
            total_reward += reward        
            global_step_num += 1
           
            
            if enviroment["render"]:
                env.render()
            if len(sonic.memory) > 2 * agent["batch_replay"]:
                sonic.replay_and_learn()


        sonic.update_target_model()
        if (total_reward > sonic.best_reward):
            sonic.best_reward = total_reward
            sonic.save_model('models/best_reward_sonic.h5')
        # if episodes % 50 == 0:
        #     sonic.save_model('models/' + str(episodes) + '_sonic_model.h5')
       
        print("Episodio #{} finalizado con recompensa {}".format(episodes + 1, total_reward))
        writer.add_scalar('ep_reward', total_reward, global_step_num)
        obs = env.reset()
        total_reward = 0.0
        
    env.close()
    sonic.save_model('models/sonic_model_final.h5')
    writer.close()
    
       
def validation(env, sonic):
    env = Monitor(env, './video',force=True)
    sonic.createModel(env,'best_reward_sonic.h5')
    obs = env.reset()
    done = False
    while not done:
        action = sonic.policy(obs)
        next_obs, reward, done, info = env.step(action)
        print("Para la accion #{} la recompensa es {}".format(action, reward))
        env.render()
        obs = next_obs
    env.close()
    
    
if __name__ == '__main__':
    
    global_step_num = 0
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
            training(env,sonic,global_step_num)
        else:
            env = make_env(env, noop_rest=False)
            validation(env,sonic)
        print("Mapa #{} Finaliza...".format(i))