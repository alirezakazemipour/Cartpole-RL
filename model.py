from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, n_states, n_actions,):
        super(Model, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions

        self.fc1 = nn.Linear(self.n_states, 32)
        self.fc2 = nn.Linear(32, 32)
        self.q_values = nn.Linear(32, self.n_actions)

        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.zero_()
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc2.bias.data.data.zero_()

        nn.init.xavier_uniform_(self.q_values.weight)
        self.q_values.bias.data.zero_()

    def forward(self, inputs):
        x = inputs
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        return self.q_values(x)

