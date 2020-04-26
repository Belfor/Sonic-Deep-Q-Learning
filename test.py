#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 11:03:35 2020

@author: Belfor
"""

import json
from SonicAgent import SonicAgent
from utils.utils import make_env
from utils.levelManager import LevelManager
from retro_contest.local import make
from gym.wrappers import Monitor



parameter = json.load(open('sonic.json', 'r'))
agent = parameter["agent"]
enviroment = parameter["enviroment"]


def validation( sonic,level):
    env = make(level[0],level[1])
    env = make_env(env)
    env = Monitor(env, './video',force=True)
    sonic.load_network(env)
    sonic.load_model('models/sonic_model_'+ level[0] +'_' + level[1] + '.h5')
    total_reward = 0
    obs = env.reset()
    done = False
    while not done:
        action = sonic.get_action(obs)
        next_obs, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()
        obs = next_obs
    env.close()
    print("La recompensa total es #{}".format(total_reward))
    
    
if __name__ == '__main__':
    
    levelManager = LevelManager('training-validation/sonic-train.csv',"training-validation/sonic-train.csv") 
    levelManager.listMap()
    levels = []
    if (enviroment["all_maps"] == False):
        levels = enviroment["selected_maps"]
    else:
        levels = [0..levelManager.size_maps_training() - 1]
    
    sonic = SonicAgent(training=False)
      
    for i in levels:
        print("Mapa #{} Comienza...".format(i))
        level = levelManager.getMap(i)
     
        validation(sonic,level)
            
        print("Mapa #{} Finaliza...".format(i))