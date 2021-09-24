from abc import ABC
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch


class Model(nn.Module, ABC):
    def __init__(self, n_states, n_actions, n_atoms, support):
        super(Model, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.support = support

        self.fc1 = nn.Linear(self.n_states, 256)
        self.adv_fc = NoisyLayer(256, 256)
        self.value_fc = NoisyLayer(256, 256)
        self.adv = NoisyLayer(256, self.n_actions * self.n_atoms)
        self.value = NoisyLayer(256, self.n_atoms)

        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        self.fc1.bias.data.zero_()

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.fc1(x))
        adv_fc = F.relu(self.adv_fc(x))
        value_fc = F.relu(self.value_fc(x))
        adv = self.adv(adv_fc).view(-1, self.n_actions, self.n_atoms)
        value = self.value(value_fc).view(-1, 1, self.n_atoms)

        mass_probs = value + adv - adv.mean(1, keepdim=True)
        return F.softmax(mass_probs, dim=-1)

    def get_q_value(self, x):
        dist = self(x)
        q_value = (dist * self.support).sum(-1)
        return q_value

    def reset(self):
        self.adv_fc.reset_noise()
        self.value_fc.reset_noise()
        self.adv.reset_noise()
        self.value.reset_noise()


class NoisyLayer(nn.Module, ABC):
    def __init__(self, n_inputs, n_outputs):
        super(NoisyLayer, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.mu_w = nn.Parameter(torch.FloatTensor(self.n_outputs, self.n_inputs))
        self.sigma_w = nn.Parameter(torch.FloatTensor(self.n_outputs, self.n_inputs))
        self.register_buffer('weight_epsilon', torch.FloatTensor(self.n_outputs, self.n_inputs))

        self.mu_b = nn.Parameter(torch.FloatTensor(self.n_outputs))
        self.sigma_b = nn.Parameter(torch.FloatTensor(self.n_outputs))
        self.register_buffer('bias_epsilon', torch.FloatTensor(self.n_outputs))

        self.mu_w.data.uniform_(-1 / np.sqrt(self.n_inputs), 1 / np.sqrt(self.n_inputs))
        self.sigma_w.data.fill_(0.1 / np.sqrt(self.n_inputs))

        self.mu_b.data.uniform_(-1 / np.sqrt(self.n_inputs), 1 / np.sqrt(self.n_inputs))
        self.sigma_b.data.fill_(0.1 / np.sqrt(self.n_outputs))

        self.reset_noise()

    def forward(self, inputs):
        x = inputs
        weights = self.mu_w + self.sigma_w * self.weight_epsilon
        biases = self.mu_b + self.sigma_b * self.bias_epsilon
        x = F.linear(x, weights, biases)
        return x

    @staticmethod
    def f(x):
        return torch.sign(x) * torch.sqrt(torch.abs(x))

    def reset_noise(self):
        epsilon_i = self.f(torch.randn(self.n_inputs))
        epsilon_j = self.f(torch.randn(self.n_outputs))
        self.weight_epsilon.copy_(epsilon_j.ger(epsilon_i))
        self.bias_epsilon.copy_(epsilon_j)
