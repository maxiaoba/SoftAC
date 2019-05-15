"""classic Acrobot task"""
import math
import numpy as np
from scipy.stats import norm
import gym
from gym import logger
from rllab.spaces.box import Box
from gym.utils import seeding
from rllab.envs.base import Env
from rllab.core.serializable import Serializable

from numpy import sin, cos, pi


class DeterministicRLEnv(Env, Serializable):

    def __init__(self,
                states = np.array([[1.],[2.],[3]]),
                mode = 2,
                seed = 0,
                ):
        self.states = states
        self.mode = mode
        self.seed(seed)

        self.state_index = 0
        Serializable.quick_init(self, locals())

    @property
    def observation_space(self):
        high = np.array([np.max(np.abs(self.states))])
        low = -high
        return Box(low=low, high=high)

    @property
    def action_space(self):
        # return spaces.Discrete(3)
        high = np.ones(2)*np.max(self.states)
        low = -high
        return Box(low=low, high=high)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state_index = 0
        return self._get_ob()

    def step(self, action):
        reward = self._get_reward(action)
        self.state_index += 1
        terminal = (self.state_index >= len(self.states))
        return self._get_ob(), reward, terminal, {}

    def _get_ob(self):
        return self.states[min(self.state_index,len(self.states)-1)]

    def _get_reward(self,action):
        reward = 0.0
        state = self._get_ob()[0]
        for a in action:
            if self.mode == 2:
                if a > 0.0:
                    reward += (state-np.abs(a-state))/float(state)
                elif a <= 0.0:
                    reward += (state-np.abs(a-(-state)))/float(state)
            elif self.mode == 1:
                reward += (state-np.abs(a-state))/float(state)
            else:
                raise NotImplementedError
        return reward


