from runner import Worker
from concurrent import futures
import cv2
import numpy as np
from brain import Brain
import gym
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter


env_name = "CartPole-v0"
test_env = gym.make(env_name)
n_states = test_env.observation_space.shape[0]
n_actions = test_env.action_space.n
n_workers = 8
device = "cuda"
iterations = int(1e6)
T = 128
epochs = 3
lr = 2.5e-4
clip_range = 0.1


def get_states(worker):
    # print('parent process:', os.getppid())
    # print('process id:', os.getpid())
    return worker.state


def render(state, id):
    print("hello")
    cv2.imshow(f"obs:{id}", state)
    cv2.waitKey(1000)


def apply_actions(worker, action):
    return worker.step(action)


if __name__ == '__main__':
    workers = [Worker(i, env_name) for i in range(n_workers)]
    brain = Brain(n_states=n_states, n_actions=n_actions, device=device, n_workers=n_workers, epochs=epochs,
                  n_iters=iterations, epsilon=clip_range, lr=lr)
    running_reward = 0
    for iteration in range(iterations):
        start_time = time.time()
        total_states = []
        total_actions = []
        total_rewards = []
        total_dones = []
        total_values = []
        for step in range(T):
            states = []
            rewards = []
            dones = []
            next_states = []
            values = []
            with futures.ThreadPoolExecutor(n_workers) as p:
                results = p.map(get_states, workers)
                for result in results:
                    states.append(result)
                states = np.vstack(states)
                actions, values = brain.get_action_and_values(states)

                results = p.map(apply_actions, workers, actions)
                for result in results:
                    next_states.append(result["next_state"])
                    rewards.append(result["reward"])
                    dones.append(result["done"])

            # with futures.ProcessPoolExecutor(n_workers) as p:
            #     p.map(render, states, np.arange(n_workers))

            next_states = np.vstack(next_states)
            rewards = np.vstack(rewards).reshape((n_workers, -1))
            dones = np.vstack(dones).reshape((n_workers, -1))
            total_states.append(states)
            total_actions.append(actions)
            total_dones.append(dones)
            total_rewards.append(rewards)
            total_values.append(values)

        _, next_values = brain.get_action_and_values(next_states)
        total_states = np.vstack(total_states)
        total_actions = np.vstack(total_actions).reshape((n_workers * T,))
        total_values = np.vstack(total_values).reshape((n_workers, -1))
        total_rewards = np.vstack(total_rewards).reshape((n_workers, -1))
        total_dones = np.vstack(total_dones).reshape((n_workers, -1))
        total_loss, entropy = brain.train(total_states, total_actions, total_rewards, total_dones, total_values, next_values)
        brain.equalize_policies()
        # brain.schedule_lr()
        # brain.schedule_clip_range(iteration)

        if iteration % 50 == 0:
            print(f"Iter:{iteration}| "
                  f"Mean_Reward:{workers[0].ep_r:.3f}| "
                  f"Running_reward:{workers[0].running_reward:.3f}| "
                  f"Total_loss:{total_loss:.3f}| "
                  f"Entropy:{entropy:.3f}| "
                  # f"Actor_Loss:{actor_loss:3.3f}| "
                  # f"Critic_Loss:{critic_loss:3.3f}| "
                  f"Iter_duration:{time.time() - start_time:.3f}| "
                  f"lr:{brain.scheduler.get_last_lr()} |"
                  f"clip_range:{brain.epsilon:.3f}")
            brain.save_weights()

        with SummaryWriter(env_name + "/logs") as writer:
            writer.add_scalar("running reward", workers[0].running_reward, iteration)
            writer.add_scalar("mean envs reward", workers[0].ep_r, iteration)
            writer.add_scalar("loss", total_loss, iteration)
            writer.add_scalar("entropy", entropy, iteration)

