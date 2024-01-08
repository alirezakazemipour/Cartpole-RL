import gymnasium as gym
import numpy as np


def evaluate_policy(env_name, agent):
    env = gym.make(env_name)
    s, _ = env.reset()
    episode_reward = 0
    done, t = False, False
    while not (done or t):
        action, _ = agent.get_actions_and_values(s)
        next_s, r, done, t, _ = env.step(action.squeeze())
        s = next_s
        episode_reward += r

    return episode_reward
