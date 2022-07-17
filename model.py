from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Model(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Model, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions

        self.fc1 = nn.Linear(self.n_states, 256)
        # self.fc2 = nn.Linear(128, 256)
        self.value = nn.Linear(256, 1)
        self.policy = nn.Linear(256, self.n_actions)

        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        self.fc1.bias.data.zero_()
        # nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")
        # self.fc2.bias.data.data.zero_()

        nn.init.xavier_uniform_(self.value.weight)
        self.value.bias.data.zero_()
        nn.init.xavier_uniform_(self.policy.weight)
        self.policy.bias.data.zero_()

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        value = self.value(x)
        dist = Categorical(F.softmax(self.policy(x), dim=1))
        return dist, value
