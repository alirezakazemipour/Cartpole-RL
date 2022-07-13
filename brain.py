from model import Model
import torch
from torch import from_numpy
import numpy as np
from torch.optim.adam import Adam


class Brain:
    def __init__(self, n_states, n_actions, device, n_workers, epochs, n_iters, epsilon, lr, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.device = device
        self.n_workers = n_workers
        self.epochs = epochs
        self.n_iters = n_iters
        self.lr = lr
        self.gamma = gamma

        self.current_policy = Model(self.n_states, self.n_actions).to(self.device)

        self.optimizer = Adam(self.current_policy.parameters(), lr=self.lr, eps=1e-5)
        self.mse_loss = torch.nn.MSELoss()

    def get_actions_and_values(self, state, batch=False):
        if not batch:
            state = np.expand_dims(state, 0)
        state = from_numpy(state).float().to(self.device)
        with torch.no_grad():
            dist, value = self.current_policy(state)
            action = dist.sample().cpu().numpy()
        return action, value.detach().cpu().numpy().squeeze()

    def train(self, states, actions, rewards, dones, values, next_values):
        returns = self.get_returns(rewards, next_values, dones, self.n_workers)
        values = np.vstack(values).reshape((len(values[0]) * self.n_workers,))
        advs = returns - values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        states = from_numpy(states).float().to(self.device)
        actions = from_numpy(actions).float().to(self.device)
        advs = from_numpy(advs).float().to(self.device)
        values_target = from_numpy(values).float().to(self.device)

        dist, value = self.current_policy(states)
        entropy = dist.entropy().mean()
        log_prob = self.calculate_log_probs(self.current_policy, states, actions)
        actor_loss = -(log_prob * advs).mean()

        c_loss = self.mse_loss(values_target, value.squeeze(-1))

        total_loss = c_loss + actor_loss - 0.01 * entropy
        self.optimize(total_loss)

        return total_loss.item(), c_loss.item(), actor_loss.item(), entropy.item()

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.current_policy.parameters(), 0.5)
        self.optimizer.step()

    @staticmethod
    def calculate_log_probs(model, states, actions):
        policy_distribution, _ = model(states)
        return policy_distribution.log_prob(actions)

    def get_returns(self, rewards: np.ndarray, next_values: np.ndarray, dones: np.ndarray, n: int) -> np.ndarray:
        if next_values.shape == ():
            next_values = next_values[None]

        returns = [[] for _ in range(n)]
        for worker in range(n):
            R = next_values[worker]  # noqa
            for step in reversed(range(len(rewards[worker]))):
                R = rewards[worker][step] + self.gamma * R * (1 - dones[worker][step])  # noqa
                returns[worker].insert(0, R)

        return np.hstack(returns).astype("float32")

    def save_weights(self):
        torch.save(self.current_policy.state_dict(), "weights.pth")

    def load_weights(self):
        self.current_policy.load_state_dict(torch.load("weights.pth"))

    def set_to_eval_mode(self):
        self.current_policy.eval()
