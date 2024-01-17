from runner import Worker
from multiprocessing import Process, Pipe, set_start_method
import numpy as np
from brain import Brain
import gymnasium as gym
import time
from test_policy import evaluate_policy
from play import Play
import os
import random
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

env_name = "Taxi-v3"
test_env = gym.make(env_name)
n_states = test_env.observation_space.n
n_actions = test_env.action_space.n
n_workers = 6
device = "cpu"
iterations = 40000
T = 5
lr = 0.0001
gamma = 0.99
lam = 0.99


def run_workers(worker, conn):
    worker.step(conn)


if __name__ == '__main__':
    set_start_method("spawn")
    seed = 1
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)



    brain = Brain(n_states=n_states,
                  n_actions=n_actions,
                  device=device,
                  n_workers=n_workers,
                  lr=lr,
                  gamma=gamma,
                  lam=lam,
                  T=T
                  )
    workers = [Worker(i, env_name, seed) for i in range(n_workers)]
    parents = []
    for worker in workers:
        parent_conn, child_conn = Pipe()
        p = Process(target=run_workers, args=(worker, child_conn,))
        parents.append(parent_conn)
        p.start()

    running_reward = 0
    tot_step = 0
    ret_hist = []
    for iteration in tqdm(range(iterations)):
        start_time = time.time()
        tot_states = np.zeros((n_workers, T))
        tot_actions = np.zeros((n_workers, T))
        tot_rewards = np.zeros((n_workers, T))
        tot_dones = np.zeros((n_workers, T))
        tot_values = np.zeros((n_workers, T))
        next_values = np.zeros(n_workers)
        next_states = np.zeros((n_workers,))

        for t in range(T):
            tot_step += n_workers
            for worker_id, parent in enumerate(parents):
                s = parent.recv()
                tot_states[worker_id, t] = s

            tot_actions[:, t], tot_values[:, t] = brain.get_acts_and_vals(tot_states[:, t], batch=True)
            for p, a in zip(parents, tot_actions[:, t]):
                p.send(int(a))

            for worker_id, p in enumerate(parents):
                s_, r, d, _ = p.recv()
                tot_rewards[worker_id, t] = r
                tot_dones[worker_id, t] = d
                next_states[worker_id] = s_
        _, next_values = brain.get_acts_and_vals(next_states, batch=True)

        tot_states = tot_states.reshape((n_workers * T,))
        tot_actions = tot_actions.reshape(n_workers * T)
        total_loss, c_loss, a_loss, entropy = brain.train(tot_states,
                                                          tot_actions,
                                                          np.clip(tot_rewards, -1, 1),
                                                          tot_dones,
                                                          tot_values,
                                                          next_values
                                                          )
        episode_reward = evaluate_policy(env_name, brain, seed, iteration)

        if iteration == 0:
            running_reward = episode_reward
        else:
            running_reward = 0.99 * running_reward + 0.01 * episode_reward

        ret_hist.append(running_reward)
        if iteration % 1000 == 0:
            print(f"Iter: {iteration}| "
                  f"Ep_reward: {episode_reward:.2f}| "
                  f"Running_reward: {running_reward:.2f}| "
                  f"Total_loss: {total_loss:.2f}| "
                  f"Entropy: {entropy:.2f}| "
                  f"Iter_duration: {time.time() - start_time:.2f}| "
                  )
            # brain.save_weights()

    plt.style.use("ggplot")
    plt.figure()
    # plt.ylim([-20, 11])
    plt.plot(np.arange(iterations), ret_hist)
    plt.savefig("running_reward2.png")

    # play = Play(test_env, brain)
    # play.evaluate()
