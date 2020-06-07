import gym
from agent import Agent
# from play import Play
# import matplotlib.pyplot as plt
import numpy as np
import time

config = {
    "env_name": "CartPole-v0",
    "do_intro": False,
    "do_train": True,
    "lr": 0.001,
    "batch_size": 64,
    "hard_update_period": 500,
    "memory_size": 100000,
    "gamma": 0.99,
    "max_episodes": 15000,
    "epsilon_decay_rate": 5e-3,
    "min_epsilon": 0.01,
    "epsilon": 1.0,
    "print_interval": 15,
    "V_min": 0,
    "V_max": 200.0,
    "N_Atoms": 51
}


env = gym.make(config["env_name"])
num_states = env.observation_space.shape[0]
num_actions = env.action_space.n

config.update({"n_states": num_states,
               "n_actions": num_actions,
               "max_steps": env._max_episode_steps})

print("Number of states:{}".format(num_states))
print("Number of actions:{}".format(num_actions))


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

    if config["do_intro"]:
        test_env_working()
        print("Environment works.")
        exit(0)

    agent = Agent(env, **config)
    running_reward = agent.run()
    #
    # episodes = np.arange(agent.max_episodes)
    # plt.style.use("ggplot")
    # plt.figure()
    # plt.plot(episodes, running_reward)
    # plt.savefig("running_reward.png")
    #
    # player = Play(env, agent)
    # player.evaluate()


