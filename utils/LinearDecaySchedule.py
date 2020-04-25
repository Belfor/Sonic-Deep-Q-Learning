#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 15:53:29 2020

@author: belfor
"""


class LinearDecaySchedule():
    
    def __init__(self, init, final,max_steps):
        self.init = init
        self.final = final
        self.factor = (init - final)/max_steps
    
    def __call__(self, step):
        value = self.init - step * self.factor
        if (value < self.final):
            return self.final
        return value
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import json
    parameter = json.load(open("sonic.json", 'r'))
  
    agent = parameter["agent"]
    enviroment = parameter["enviroment"]
    
    if (enviroment["all_maps"] == False):
        levels = enviroment["selected_maps"]
    else:
        levels = [0..levelManager.size_maps_training() - 1]
    
  
    epsilon_initial = agent["epsilon_max"]
    epsilon_final = agent["epsilon_min"]
    max_num_episodes =agent["episodes"]
    setps_per_episode = agent["steps_episode"]
    
    print(len(levels))
    linear_schedule = LinearDecaySchedule( epsilon_initial,
                                           epsilon_final, 
                                           len(levels) * max_num_episodes * setps_per_episode)
    
    epsilons = [linear_schedule(step) for step in range(len(levels) * max_num_episodes * setps_per_episode)]
    plt.plot(epsilons)
    plt.show()