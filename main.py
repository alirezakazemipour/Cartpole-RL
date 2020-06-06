import gym
# from train import Agent
# from play import Play
# import matplotlib.pyplot as plt
# import numpy as np
import time

env_name = "CartPole-v0"
Intro = True

env = gym.make(env_name)
num_states = env.observation_space.shape[0]
num_actions = env.action_space.n


print("Number of states:{}".format(num_states))
print("Number of actions:{}".format(num_actions))


def test_env_working():
    test_env = gym.make(env_name)
    test_env.reset()
    for _ in range(test_env._max_episode_steps):
        test_env.render()
        action = test_env.action_space.sample()
        test_env.step(action)
        time.sleep(0.05)
    test_env.close()


if __name__ == "__main__":

    if Intro:
        test_env_working()
        print("Environment works.")
        exit(0)

    # agent = Agent(env, num_actions, num_states, num_features)
    # running_reward = agent.run()
    #
    # episodes = np.arange(agent.max_episodes)
    # plt.style.use("ggplot")
    # plt.figure()
    # plt.plot(episodes, running_reward)
    # plt.savefig("running_reward.png")
    #
    # player = Play(env, agent)
    # player.evaluate()


