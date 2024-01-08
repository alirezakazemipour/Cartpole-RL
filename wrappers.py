import gymnasium as gym
from abc import ABC
import numpy as np


class OneHotEnv(gym.ObservationWrapper, ABC):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, obs):
        return one_hot(obs, self.env.observation_space.n)


def one_hot(k, n):
    one_hot_rep = np.zeros(n)
    one_hot_rep[k] = 1
    return one_hot_rep
