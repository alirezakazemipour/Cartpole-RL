import numpy as np
from model import Model
from memory import Memory, Transition
import torch
from torch import device
from torch import from_numpy
from torch.optim import Adam
import time
import datetime
import psutil
from collections import deque


class Agent:
    def __init__(self, env, **config):

        self.config = config
        self.env = env
        self.epsilon = self.config["epsilon"]
        self.min_epsilon = self.config["min_epsilon"]
        self.decay_rate = self.config["epsilon_decay_rate"]
        self.n_actions = self.config["n_actions"]
        self.n_states = self.config["n_states"]
        self.max_steps = self.config["max_steps"]
        self.max_episodes = self.config["max_episodes"]
        self.batch_size = self.config["batch_size"]
        self.gamma = self.config["gamma"]
        self.memory = Memory(self.config["memory_size"], self.config["alpha"])
        self.device = device(self.config["device"])
        self.to_gb = lambda in_bytes: in_bytes / 1024 / 1024 / 1024

        self.n_step_buffer = deque(maxlen=self.config["n_step"])
        self.v_min = self.config["V_min"]
        self.v_max = self.config["V_max"]
        self.n_atoms = self.config["N_Atoms"]
        self.support = torch.linspace(self.v_min, self.v_max, self.n_atoms).to(self.device)
        self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)
        self.offset = torch.linspace(0, (self.batch_size - 1) * self.n_atoms, self.batch_size).long() \
            .unsqueeze(1).expand(self.batch_size, self.n_atoms).to(self.device)

        self.eval_model = Model(self.n_states, self.n_actions, self.n_atoms, self.support).to(self.device)
        self.target_model = Model(self.n_states, self.n_actions, self.n_atoms, self.support).to(self.device)
        self.target_model.load_state_dict(self.eval_model.state_dict())
        self.optimizer = Adam(self.eval_model.parameters(), lr=self.config["lr"])

    def choose_action(self, state):
        exp = np.random.rand()
        if self.epsilon > exp:
            return np.random.randint(self.n_actions)

        else:
            state = np.expand_dims(state, axis=0)
            state = from_numpy(state).float().to(self.device)
            return np.argmax(self.eval_model.get_q_value(state).detach().cpu().numpy())

    def hard_update_target_model(self):
        self.target_model.load_state_dict(self.eval_model.state_dict())
        self.target_model.eval()

    @staticmethod
    def soft_update_of_target_network(local_model, target_model, tau=0.001):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        target_model.eval()

    def train(self, beta):
        if len(self.memory) < self.batch_size:
            return 0  # as no loss
        batch, weights, indices = self.memory.sample(self.batch_size, beta)
        weights = from_numpy(weights).float().to(self.device)
        states, actions, rewards, next_states, dones = self.unpack_batch(batch)

        with torch.no_grad():
            q_eval_next = self.eval_model.get_q_value(next_states)
            next_actions = q_eval_next.argmax(dim=-1)
            q_next = self.target_model(next_states)[range(self.batch_size), next_actions.long()]

            projected_atoms = rewards + (self.config["gamma"] ** self.config["n_step"]) * self.support * (1 - dones)
            projected_atoms = projected_atoms.clamp_(self.v_min, self.v_max)

            b = (projected_atoms - self.v_min) / self.delta_z
            lower_bound = b.floor().long()
            upper_bound = b.ceil().long()

            # projected_dist = torch.zeros((self.batch_size, self.n_atoms)).to(self.device)
            # for i in range(self.batch_size):
            #     for j in range(self.n_atoms):
            #         projected_dist[i, lower_bound[i, j]] += (q_next * (upper_bound - b))[i, j]
            #         projected_dist[i, upper_bound[i, j]] += (q_next * (b - lower_bound))[i, j]

            projected_dist = torch.zeros(q_next.size()).to(self.device)
            projected_dist.view(-1).index_add_(0, (lower_bound + self.offset).view(-1),
                                               (q_next * (upper_bound.float() - b)).view(-1))
            projected_dist.view(-1).index_add_(0, (upper_bound + self.offset).view(-1),
                                               (q_next * (b - lower_bound.float())).view(-1))

        eval_dist = self.eval_model(states)[range(self.batch_size), actions.squeeze().long()]
        dqn_loss = - (projected_dist * torch.log(eval_dist + 1e-6)).sum(-1)
        td_error = dqn_loss.abs()
        self.memory.update_priorities(indices, td_error.abs().detach().cpu().numpy() + 0.01)
        dqn_loss = (dqn_loss * weights).mean()

        self.optimizer.zero_grad()
        dqn_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_model.parameters(), 10.0)
        self.optimizer.step()

        return dqn_loss.detach().cpu().numpy()

    def run(self):

        total_global_running_reward = []
        global_running_reward = 0
        for episode in range(1, 1 + self.max_episodes):
            start_time = time.time()
            state = self.env.reset()
            episode_reward = 0
            for step in range(1, 1 + self.max_steps):
                action = self.choose_action(state)
                next_state, reward, done, _, = self.env.step(action)
                episode_reward += reward
                reward = np.clip(reward, -1, 1)
                beta = min(1.0, self.config["beta"] + episode * (1.0 - self.config["beta"]) / 1000)
                self.store(state, reward, done, action, next_state)
                dqn_loss = self.train(beta)
                if done:
                    break
                state = next_state

                if (episode * step) % self.config["hard_update_period"] == 0:
                    self.hard_update_target_model()
                # self.soft_update_of_target_network(self.eval_model, self.target_model, tau=0.05)
                self.epsilon = self.epsilon - self.decay_rate if self.epsilon > self.min_epsilon + self.decay_rate\
                    else self.min_epsilon

            if episode == 1:
                global_running_reward = episode_reward
            else:
                global_running_reward = 0.99 * global_running_reward + 0.01 * episode_reward

            total_global_running_reward.append(global_running_reward)
            ram = psutil.virtual_memory()
            if episode % self.config["print_interval"] == 0:
                print(f"EP:{episode}| "
                      f"DQN_loss:{dqn_loss:.2f}| "
                      f"EP_reward:{episode_reward}| "
                      f"EP_running_reward:{global_running_reward:.3f}| "
                      f"Epsilon:{self.epsilon:.3f}| "
                      f"Memory size:{len(self.memory)}| "
                      f"EP_Duration:{time.time() - start_time:.3f}| "
                      f"Beta:{beta:.3f}| "
                      f"Step:{step}| "
                      f"{self.to_gb(ram.used):.1f}/{self.to_gb(ram.total):.1f} GB RAM| "
                      f'Time:{datetime.datetime.now().strftime("%H:%M:%S")}')
                self.save_weights()

        return total_global_running_reward

    def store(self, state, reward, done, action, next_state):
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) < self.config["n_step"]:
            return
        reward, next_state, done = self.n_step_returns()
        state, action, _, _, _ = self.n_step_buffer.pop()

        state = from_numpy(state).float().to(self.device)
        reward = torch.Tensor([reward]).to(self.device)
        done = torch.Tensor([done]).to(self.device)
        action = torch.Tensor([action]).to(self.device)
        next_state = from_numpy(next_state).float().to(self.device)
        self.memory.add(state, reward, done, action, next_state)

    def unpack_batch(self, batch):

        batch = Transition(*zip(*batch))

        states = torch.cat(batch.state).to(self.device).view(self.batch_size, self.n_states)
        actions = torch.cat(batch.action).to(self.device)
        rewards = torch.cat(batch.reward).to(self.device).view(-1, 1)
        next_states = torch.cat(batch.next_state).to(self.device).view(self.batch_size, self.n_states)
        dones = torch.cat(batch.done).to(self.device).view(-1, 1)
        actions = actions.view((-1, 1))
        return states, actions, rewards, next_states, dones

    def save_weights(self):
        torch.save(self.eval_model.state_dict(), "weights.pth")

    def load_weights(self):
        self.eval_model.load_state_dict(torch.load("weights.pth", map_location="cpu"))

    def set_to_eval_mode(self):
        self.eval_model.eval()

    def n_step_returns(self):
        reward, next_state, done = self.n_step_buffer[-1][-3:]

        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_s, d = transition[-3:]

            reward = r + self.config["gamma"] * reward * (1 - d)
            next_state, done = (n_s, d) if d else (next_state, done)

        return reward, next_state, done
