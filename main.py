import gym
from agent import Agent
from play import Play
import matplotlib.pyplot as plt
import numpy as np
import time
import os

config = {
    "env_name": "MountainCar-v0",
    "do_intro": False,
    "do_train": True,
    "lr": 0.0001,
    "batch_size": 64,
    "hard_update_period": 500,
    "memory_size": 15000,
    "gamma": 0.99,
    "max_episodes": 2000,
    "print_interval": 10,
    "device": "cpu",
    "n_step": 3,
    "V_min": -200.,
    "V_max": 1.,
    "N_Atoms": 51,
    "alpha": 0.6,
    "beta": 0.4,
    "seed": 123
}


env = gym.make(config["env_name"])
num_states = env.observation_space.shape[0]
num_actions = env.action_space.n

config.update({"n_states": num_states,
               "n_actions": num_actions,
               "max_steps": env.spec.max_episode_steps})


print("Environment is: {}".format(config["env_name"]))
print("Number of states: {}".format(num_states))
print("Number of actions: {}".format(num_actions))


def test_env_working():
    test_env = gym.make(config["env_name"])
    test_env.reset()
    for _ in range(test_env._max_episode_steps):
        test_env.render()
        action = test_env.action_space.sample()
        test_env.step(action)
        time.sleep(0.05)
    test_env.close()


if __name__ == "__main__":
    # os.environ[]

    if config["do_intro"]:
        test_env_working()
        print("Environment works.")
        exit(0)

    agent = Agent(env, **config)
    if config["do_train"]:
        running_reward = agent.run()

        episodes = np.arange(agent.max_episodes)
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(episodes, running_reward)
        plt.savefig("running_reward.png")
    else:
        player = Play(env, agent)
        player.evaluate()

