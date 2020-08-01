import gym
import numpy as np


def evaluate_policy(env_name, agent):
    env = gym.make(env_name)
    s = env.reset()
    episode_reward = 0
    done = False
    while not done:
        action, _ = agent.get_actions_and_values(s)
        next_s, r, done, _ = env.step(action.squeeze())
        s = next_s
        episode_reward += r

    return episode_reward
