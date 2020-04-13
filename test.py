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
    
    levelManager = LevelManager('training-validation/sonic-train.csv',"training-validation/sonic-train.csv") 
    levelManager.listMap()
    levels = []
    if (enviroment["all_maps"] == False):
        levels = enviroment["selected_maps"]
    else:
        levels = [0..levelManager.size_maps_training() - 1]
    
    sonic = SonicAgent(False)
      
    for i in levels:
        print("Mapa #{} Comienza...".format(i))
        level = levelManager.getMap(i)
        env = make(level[0],level[1])
        env = make_env(env, noop_rest=False)
        validation(env,sonic)
            
        print("Mapa #{} Finaliza...".format(i))