from torch import nn
import torch.nn.functional as F
import numpy as np
import torch


class Model(nn.Module):
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

        nn.init.kaiming_normal_(self.fc1.weight)
        self.fc1.bias.data.zero_()

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.fc1(x))
        adv_fc = F.relu(self.adv_fc(x))
        value_fc = F.relu(self.value_fc(x))
        adv = self.adv(adv_fc.T).T.view(-1, self.n_actions, self.n_atoms)
        value = self.value(value_fc.T).T.view(-1, 1, self.n_atoms)

        mass_probs = value + adv - adv.mean(1, keepdim=True)
        return F.softmax(mass_probs, dim=-1).clamp(min=1e-3)

    def get_q_value(self, x):
        dist = self(x)
        q_value = (dist * self.support).sum(-1)
        return q_value

    def reset(self):
        self.adv_fc.reset_noise()
        self.value_fc.reset_noise()
        self.adv.reset_noise()
        self.value.reset_noise()


class NoisyLayer(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(NoisyLayer, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.mu_w = nn.Parameter(torch.FloatTensor(self.n_outputs, self.n_inputs))
        self.sigma_w = nn.Parameter(torch.FloatTensor(self.n_outputs, self.n_inputs))

        self.mu_b = nn.Parameter(torch.FloatTensor(self.n_outputs, 1))
        self.sigma_b = nn.Parameter(torch.FloatTensor(self.n_outputs, 1))

        self.mu_w.data.uniform_(-1 / np.sqrt(self.n_inputs), 1 / np.sqrt(self.n_inputs))
        self.sigma_w.data.fill_(0.5 / np.sqrt(self.n_inputs))

        self.mu_b.data.uniform_(-1 / np.sqrt(self.n_inputs), 1 / np.sqrt(self.n_inputs))
        self.sigma_b.data.fill_(0.5 / np.sqrt(self.n_inputs))

        self.reset_noise()

    def forward(self, inputs):
        x = inputs
        weights = self.mu_w + self.sigma_w.mul(self.epsilon_j.mm(self.epsilon_i.T))
        biases = self.mu_b + self.sigma_b.mul(self.epsilon_j)
        x = x.mm(weights.T).T + biases
        return x

    def f(self, x):
        return torch.sign(x) * torch.sqrt(torch.abs(x))

    def reset_noise(self):
        self.epsilon_i = self.f(torch.randn(self.n_inputs)).view(-1, 1)
        self.epsilon_j = self.f(torch.randn(self.n_outputs)).view(-1, 1)
