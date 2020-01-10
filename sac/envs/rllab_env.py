from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
from rllab.spaces.discrete import Discrete
from rllab.spaces.product import Product
from rllab.misc import logger
from rllab.envs.env_spec import EnvSpec
from rllab.misc.overrides import overrides

import gym

from cached_property import cached_property

def convert_gym_space(space):
    if isinstance(space, gym.spaces.Box):
        return Box(low=space.low, high=space.high)
    elif isinstance(space, gym.spaces.Discrete):
        return Discrete(n=space.n)
    elif isinstance(space, gym.spaces.Tuple):
        return Product([convert_gym_space(x) for x in space.spaces])
    else:
        raise NotImplementedError

class RLLabEnv(Serializable):
    def __init__(self, env):
        Serializable.quick_init(self, locals())
        self.env = env
        self.action_space = convert_gym_space(self.env.action_space)
        self.observation_space = convert_gym_space(self.env.observation_space)

    @cached_property
    def spec(self):
        """
        Returns an EnvSpec.

        Returns:
            spec (garage.envs.EnvSpec)
        """
        return EnvSpec(
            observation_space=self.observation_space,
            action_space=self.action_space)

    def reset(self):
        return self.env.reset()

    def step(self,action):
        return self.env.step(action)

    def seed(self,seed):
        return self.env.seed(seed)

    def render(self, mode='human', close=False):
        return self.env._render(mode, close)
        # self.env.render()

    def log_diagnostics(self, paths):
        pass

    def terminate(self):
        pass