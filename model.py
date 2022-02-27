from torch import nn
import torch.nn.functional as F
import torch
import math


class Model(nn.Module):
    def __init__(self, n_states, n_actions, n_embedding, k):
        super(Model, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_embedding = n_embedding
        self.k = k

        self.fc1 = nn.Linear(self.n_states, 128)
        self.fc2 = nn.Linear(128, 256)

        self.phi = nn.Linear(n_embedding, 128)
        self.z = nn.Linear(256, self.n_actions)

        nn.init.kaiming_normal_(self.fc1.weight)
        self.fc1.bias.data.zero_()
        nn.init.kaiming_normal_(self.fc2.weight)
        self.fc2.bias.data.data.zero_()
        nn.init.kaiming_normal_(self.phi.weight)
        self.phi.bias.data.data.zero_()
        nn.init.xavier_uniform_(self.z.weight)
        self.z.bias.data.zero_()

    def forward(self, inputs):
        states, taus = inputs
        x = states
        state_feats = F.relu(self.fc1(x))

        i_pi = math.pi * torch.arange(1, 1 + self.n_embedding, device=taus.device).view(1, 1, self.n_embedding)
        taus = torch.unsqueeze(taus, -1)
        x = torch.cos(i_pi * taus).view(-1, self.n_embedding)
        phi = F.relu(self.phi(x))

        x = state_feats.view(state_feats.size(0), 1, -1) * phi.view(states.size(0), taus.size(1), -1)
        x = x.view(-1, phi.size(-1))
        x = F.relu(self.fc2(x))
        z = self.z(x)
        return z.view(states.size(0), taus.size(1), -1)

    def get_qvalues(self, x):
        tau_tildas = torch.rand((x.size(0), self.k), device=x.device)
        z = self.forward((x, tau_tildas))
        q_values = torch.mean(z, dim=1)
        return q_values
