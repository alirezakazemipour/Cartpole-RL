from torch import nn
import torch.nn.functional as F
import torch
import numpy as np


class Model(nn.Module):
    def __init__(self, n_states, n_actions, ):
        super(Model, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions

        self.fc1 = nn.Linear(self.n_states, 128)
        self.fc2 = NoisyLayer(128, 128)
        self.q_values = NoisyLayer(128, self.n_actions)

        nn.init.kaiming_normal_(self.fc1.weight)
        self.fc1.bias.data.zero_()

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q_values(x.T).T

    def reset(self):
        self.fc2.reset_noise()
        self.q_values.reset_noise()


class NoisyLayer(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(NoisyLayer, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.mu_w = nn.Parameter(torch.FloatTensor(self.n_outputs, self.n_inputs))
        self.sigma_w = nn.Parameter(torch.FloatTensor(self.n_outputs, self.n_inputs))
        self.register_buffer('weight_epsilon', torch.FloatTensor(self.n_outputs, self.n_inputs))

        self.mu_b = nn.Parameter(torch.FloatTensor(self.n_outputs, 1))
        self.sigma_b = nn.Parameter(torch.FloatTensor(self.n_outputs, 1))
        self.register_buffer('bias_epsilon', torch.FloatTensor(self.n_outputs, 1))

        self.mu_w.data.uniform_(-1 / np.sqrt(self.n_inputs), 1 / np.sqrt(self.n_inputs))
        self.sigma_w.data.fill_(0.5 / np.sqrt(self.n_inputs))

        self.mu_b.data.uniform_(-1 / np.sqrt(self.n_inputs), 1 / np.sqrt(self.n_inputs))
        self.sigma_b.data.fill_(0.5 / np.sqrt(self.n_outputs))

        self.epsilon_i = 0
        self.epsilon_j = 0
        self.reset_noise()

    def forward(self, inputs):
        x = inputs
        weights = self.mu_w + self.sigma_w.mul(self.weight_epsilon)
        biases = self.mu_b + self.sigma_b.mul(self.bias_epsilon)
        x = x.mm(weights.T).T + biases
        return x

    def f(self, x):
        return torch.sign(x) * torch.sqrt(torch.abs(x))

    def reset_noise(self):
        self.epsilon_i = self.f(torch.randn(self.n_inputs)).view((-1, 1))
        self.epsilon_j = self.f(torch.randn(self.n_outputs)).view((-1, 1))
        self.weight_epsilon.copy_(self.epsilon_j.mm(self.epsilon_i.T))
        self.bias_epsilon.copy_(self.epsilon_j)
