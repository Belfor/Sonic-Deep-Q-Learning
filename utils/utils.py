#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Environments and wrappers for Sonic training.
"""

import gym
import numpy as np
import random

from baselines.common.atari_wrappers import WarpFrame, FrameStack, ClipRewardEnv, ScaledFloatFrame


def make_env(env,stack=True, scale_rew=False, noop_rest=False, allow_backtracking = False):
    """
    Create an environment with some standard wrappers.
    """
    env = SonicDiscretizer(env)
    env = WarpFrame(env, 128, 128)
    
    if scale_rew:
        env = RewardScaler(env)
  
    if stack:
        env = FrameStack(env, 4)
    if noop_rest:
        env = NoopResetEnv(env)
    if allow_backtracking:
        env = AllowBacktracking(env)
    return env

class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()

class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.
    This is incredibly important and effects performance
    drastically.
    """
    def reward(self, reward):
        return reward * 0.01

class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """
    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0

    def reset(self, **kwargs): # pylint: disable=E0202
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action): # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, rew, done, info

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max = 30):
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.noop_action = 4
    
        
    def reset(self):
        self.env.reset()
        noops = random.randrange(1, self.noop_max +1)
        assert noops > 0
        observation = None
        for _ in range(noops):
            observation, _, done, _ = self.env.step(self.noop_action)
        return observation
    
    def step(self, action):
        return self.env.step(action)