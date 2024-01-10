from model import Model
import torch
from torch import from_numpy
import numpy as np
from torch.optim.rmsprop import RMSprop


class Brain:
    def __init__(self, n_states, n_actions, device, n_workers, lr, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.device = torch.device(device)
        self.n_workers = n_workers
        self.lr = lr
        self.gamma = gamma

        self.model = Model(self.n_states, self.n_actions).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.mse_loss = torch.nn.MSELoss()

    def get_actions_and_values(self, state, batch=False):
        if not batch:
            state = np.expand_dims(state, 0)
        state = from_numpy(state).float().to(self.device)
        with torch.no_grad():
            dist, value = self.model(state)
            action = dist.sample().cpu().numpy()
        return action, value.cpu().numpy().squeeze()

    def train(self, states, actions, rewards, dones, values, next_values):
        returns = self.get_returns(rewards, next_values, dones, self.n_workers)
        values = np.hstack(values)
        advs = returns - values

        states = from_numpy(states).float().to(self.device)
        actions = from_numpy(actions).float().to(self.device)
        advs = from_numpy(advs).float().to(self.device)
        values_target = from_numpy(returns).float().to(self.device)

        dist, value = self.model(states)
        entropy = dist.entropy().mean()
        log_prob = dist.log_prob(actions)
        a_loss = -(log_prob * advs).mean()

        c_loss = self.mse_loss(values_target, value.squeeze(-1))

        total_loss = 0.5 * c_loss + a_loss - 0.01 * entropy
        self.optimize(total_loss)
        return total_loss.item(), c_loss.item(), a_loss.item(), entropy.item()

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

    def get_returns(self, rewards: np.ndarray, next_values: np.ndarray, dones: np.ndarray, n: int) -> np.ndarray:
        if n == 1:
            next_values = next_values[None]

        returns = [[] for _ in range(n)]
        for worker in range(n):
            R = next_values[worker]  # noqa
            for step in reversed(range(len(rewards[worker]))):
                R = rewards[worker][step] + self.gamma * R * (1 - dones[worker][step])  # noqa
                returns[worker].insert(0, R)

        return np.hstack(returns)

    def save_weights(self):
        torch.save(self.model.state_dict(), "weights.pth")

    def load_weights(self):
        self.model.load_state_dict(torch.load("weights.pth"))

    def set_to_eval_mode(self):
        self.model.eval()
