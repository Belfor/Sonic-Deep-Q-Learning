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
    epsilon_initial = 1.0
    epsilon_final = 0.005
    MAX_NUM_EPISODES = 10000
    STEPS_PER_EPISODE = 300
    linear_schedule = LinearDecaySchedule( epsilon_initial,
                                           epsilon_final, 
                                           0.8* MAX_NUM_EPISODES * STEPS_PER_EPISODE)
    
    epsilons = [linear_schedule(step) for step in range(MAX_NUM_EPISODES * STEPS_PER_EPISODE)]
    plt.plot(epsilons)
    plt.show()