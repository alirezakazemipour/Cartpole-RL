from torch import nn
import torch.nn.functional as F
import torch


class Model(nn.Module):
    def __init__(self, n_states, n_actions, n_atoms, support):
        super(Model, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.support = support

        self.fc1 = nn.Linear(self.n_states, 256)
        self.fc2 = nn.Linear(256, 128)
        self.mass_probs = nn.Linear(128, self.n_actions * self.n_atoms)

        nn.init.kaiming_normal_(self.fc1.weight)
        self.fc1.bias.data.zero_()
        nn.init.kaiming_normal_(self.fc2.weight)
        self.fc2.bias.data.data.zero_()

        nn.init.xavier_uniform_(self.mass_probs.weight)
        self.mass_probs.bias.data.zero_()

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.mass_probs(x).view(-1, self.n_actions, self.n_atoms),
                         dim=-1)  # (Batch size, N_Actions, N_Atoms)

    def get_q_value(self, x):
        dist = self(x)
        q_values = (dist * self.support).sum(dim=-1)  # (Batch size, N_Actions)
        return q_values
