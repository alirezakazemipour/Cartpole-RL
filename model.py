from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, n_states, n_actions,):
        super(Model, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions

        self.fc1 = nn.Linear(self.n_states, 128)
        self.fc2 = nn.Linear(128, 256)
        self.q_values = nn.Linear(256, self.n_actions)

        nn.init.kaiming_normal_(self.fc1.weight)
        self.fc1.bias.data.zero_()
        nn.init.kaiming_normal_(self.fc2.weight)
        self.fc2.bias.data.data.zero_()

        nn.init.xavier_uniform_(self.q_values.weight)
        self.q_values.bias.data.zero_()

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q_values(x)

