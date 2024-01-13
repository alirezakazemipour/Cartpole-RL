import gymnasium as gym
import numpy as np
from wrappers import OneHotEnv


def cantor_pairing(x: int, y: int) -> int:
    return int(0.5 * (x + y) * (x + y + 1) + y)


def evaluate_policy(env_name, agent, seed, iteration):
    env = gym.make(env_name)
    # env = OneHotEnv(env)
    seed = cantor_pairing(seed, iteration)
    s, _ = env.reset(seed=seed)
    episode_reward = 0
    done, t = False, False
    while not (done or t):
        action, _ = agent.get_acts_and_vals(s)
        next_s, r, done, t, _ = env.step(action[0])
        s = next_s
        episode_reward += r

    return episode_reward
