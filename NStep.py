#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import deque
class NStep():

    def __init__(self,n_step,gamma):
        self.step = 0
        self.n_step = n_step
        self.gamma = gamma
        self.rewards = deque(maxlen=n_step)
        self.exps = deque(maxlen=n_step)

    def append(self, reward, exp):
        self.rewards.appendleft(reward)
        self.exps.append(exp)
        self.step += 1

    def calculate_rewards_nstep(self):
        total_rewards = 0
        for i in range(len(self.rewards)):
            total_rewards += ((self.gamma ** i) * self.rewards[i]) 
        obs, action, _ = self.exps[-1]
        next_obs, _, done = self.exps[0]

        self.step = 0
        return obs,action,total_rewards,next_obs,done
    
    def clear(self):
        self.step = 0
        self.rewards.clear()
        self.exps.clear()

    def is_last_step(self):
        if (self.step == self.n_step - 1):
            return True
        return False