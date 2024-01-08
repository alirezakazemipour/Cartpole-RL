import gymnasium as gym
import numpy as np
from wrappers import OneHotEnv


def cantor_pairing(x: int, y: int) -> int:
    return int(0.5 * (x + y) * (x + y + 1) + y)


class Worker:
    def __init__(self, id, env_name):
        self.id = id
        self.env_name = env_name
        self.env = None
        self._state = None
        self.ep = 0

    def __str__(self):
        return str(self.id)

    def render(self):
        self.env.render()

    def reset(self):
        ep_seed = cantor_pairing(self.id, self.ep)
        self._state, _ = self.env.reset(seed=ep_seed)

    def step(self, conn):
        self.env = gym.make(self.env_name, is_slippery=False)
        self.env = OneHotEnv(self.env)
        self._state = None
        self.reset()
        while True:
            conn.send(self._state)
            action = conn.recv()
            next_state, r, d, t, _ = self.env.step(action)
            # self.render()
            self._state = next_state
            if d or t:
                self.reset()
                self.ep += 1
            conn.send((next_state, r, d, _))
