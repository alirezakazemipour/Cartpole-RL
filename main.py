from runner import Worker
from multiprocessing import Process, Pipe
import numpy as np
from brain import Brain
import gym
import os
import time
from torch.utils.tensorboard import SummaryWriter
from test_policy import evaluate_policy
from play import Play

env_name = "LunarLander-v2"
test_env = gym.make(env_name)
n_states = test_env.observation_space.shape[0]
n_actions = test_env.action_space.n
n_workers = 8
device = "cuda"
iterations = int(4000)
T = 128
epochs = 3
lr = 2.5e-4
clip_range = 0.1


def run_workers(worker, conn):
    worker.step(conn)


if __name__ == '__main__':
    brain = Brain(n_states=n_states, n_actions=n_actions, device=device, n_workers=n_workers, epochs=epochs,
                  n_iters=iterations, epsilon=clip_range, lr=lr)
    workers = [Worker(i, env_name) for i in range(n_workers)]
    parents = []
    for worker in workers:
        parent_conn, child_conn = Pipe()
        p = Process(target=run_workers, args=(worker, child_conn,))
        parents.append(parent_conn)
        p.start()

    running_reward = 0
    for iteration in range(iterations):
        start_time = time.time()
        total_states = np.zeros((n_workers, T, n_states))
        total_actions = np.zeros((n_workers, T))
        total_rewards = np.zeros((n_workers, T))
        total_dones = np.zeros((n_workers, T))
        total_values = np.zeros((n_workers, T))
        next_values = np.zeros(n_workers)
        next_states = np.zeros((n_workers, n_states))

        for t in range(T):
            for worker_id, parent in enumerate(parents):
                s = parent.recv()
                total_states[worker_id, t] = s

            total_actions[:, t], total_values[:, t] = brain.get_actions_and_values(total_states[:, t], batch=True)
            for parent, a in zip(parents, total_actions[:, t]):
                parent.send(int(a))

            for worker_id, parent in enumerate(parents):
                s_, r, d, _ = parent.recv()
                total_rewards[worker_id, t] = r
                total_dones[worker_id, t] = d
                next_states[worker_id] = s_
        _, next_values = brain.get_actions_and_values(next_states, batch=True)

        total_states = total_states.reshape((n_workers * T, n_states))
        total_actions = total_actions.reshape(n_workers * T)
        total_loss, entropy = brain.train(total_states, total_actions, total_rewards,
                                          total_dones, total_values, next_values)
        brain.equalize_policies()
        brain.schedule_lr()
        brain.schedule_clip_range(iteration)
        episode_reward = evaluate_policy(env_name, brain)

        if iteration == 0:
            running_reward = episode_reward
        else:
            running_reward = 0.99 * running_reward + 0.01 * episode_reward

        if iteration % 50 == 0:
            print(f"Iter: {iteration}| "
                  f"Ep_reward: {episode_reward:.3f}| "
                  f"Running_reward: {running_reward:.3f}| "
                  f"Total_loss: {total_loss:.3f}| "
                  f"Entropy: {entropy:.3f}| "
                  f"Iter_duration: {time.time() - start_time:.3f}| "
                  f"Lr: {brain.scheduler.get_last_lr()}| "
                  f"Clip_range:{brain.epsilon:.3f}")
            brain.save_weights()

        with SummaryWriter(env_name + "/logs") as writer:
            writer.add_scalar("running reward", running_reward, iteration)
            writer.add_scalar("episode reward", episode_reward, iteration)
            writer.add_scalar("loss", total_loss, iteration)
            writer.add_scalar("entropy", entropy, iteration)

    play = Play(test_env, brain)
    play.evaluate()
