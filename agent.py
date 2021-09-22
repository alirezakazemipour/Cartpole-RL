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


class Agent:
    def __init__(self, env, **config):

        self.config = config
        self.env = env
        self.n_actions = self.config["n_actions"]
        self.n_states = self.config["n_states"]
        self.max_steps = self.config["max_steps"]
        self.max_episodes = self.config["max_episodes"]
        self.batch_size = self.config["batch_size"]
        self.gamma = self.config["gamma"]
        self.memory = Memory(self.config["memory_size"])
        self.device = device(self.config["device"])

        self.eval_model = Model(self.n_states, self.n_actions).to(self.device)
        self.target_model = Model(self.n_states, self.n_actions).to(self.device)
        self.target_model.load_state_dict(self.eval_model.state_dict())

        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = Adam(self.eval_model.parameters(), lr=self.config["lr"])

        self.to_gb = lambda in_bytes: in_bytes / 1024 / 1024 / 1024

    def choose_action(self, state):

        state = np.expand_dims(state, axis=0)
        state = from_numpy(state).float().to(self.device)
        return np.argmax(self.eval_model(state).detach().cpu().numpy())

    def update_train_model(self):
        self.target_model.load_state_dict(self.eval_model.state_dict())
        self.target_model.eval()

    def train(self):
        if len(self.memory) < self.batch_size:
            return 0, 0  # as no loss
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = self.unpack_batch(batch)

        q_eval = self.eval_model(states).gather(dim=-1, index=actions.long())
        with torch.no_grad():
            q_next = self.target_model(next_states)
            q_eval_next = self.eval_model(next_states)

            next_actions = q_eval_next.argmax(dim=-1).view(-1, 1)
            q_next = q_next.gather(dim=-1, index=next_actions.long())

            q_target = rewards + self.gamma * q_next * (1 - dones)
        dqn_loss = self.loss_fn(q_eval, q_target)

        self.optimizer.zero_grad()
        dqn_loss.backward()
        g_norm = torch.nn.utils.clip_grad_norm_(self.eval_model.parameters(), 10.)
        self.optimizer.step()

        self.eval_model.reset()
        self.target_model.reset()

        return dqn_loss.detach().cpu().numpy(), g_norm.item()

    def run(self):

        total_global_running_reward = []
        global_running_reward = 0
        for episode in range(1, 1 + self.max_episodes):
            start_time = time.time()
            state = self.env.reset()
            episode_reward = 0
            for step in range(1, 1 + self.max_steps):
                action = self.choose_action(state)
                # self.env.render()
                next_state, reward, done, _, = self.env.step(action)
                episode_reward += reward
                self.store(state, reward, done, action, next_state)
                dqn_loss, g_norm = self.train()
                if done:
                    break
                state = next_state

                if (episode * step) % self.config["hard_update_period"] == 0:
                    self.update_train_model()

            if episode == 1:
                global_running_reward = episode_reward
            else:
                global_running_reward = 0.99 * global_running_reward + 0.01 * episode_reward

            total_global_running_reward.append(global_running_reward)
            ram = psutil.virtual_memory()
            if episode % self.config["print_interval"] == 0:
                print(f"E:{episode}| "
                      f"loss:{dqn_loss:.2f}| "
                      f"g_norm:{g_norm:.2f}| "
                      f"E_reward:{episode_reward}| "
                      f"E_running_reward:{global_running_reward:.1f}| "
                      f"Mem size:{len(self.memory)}| "
                      f"E_Duration:{time.time()-start_time:.3f}| "
                      f"{self.to_gb(ram.used):.1f}/{self.to_gb(ram.total):.1f} GB RAM| "
                      f'Time:{datetime.datetime.now().strftime("%H:%M:%S")}')
                self.save_weights()

        return total_global_running_reward

    def store(self, state, reward, done, action, next_state):
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