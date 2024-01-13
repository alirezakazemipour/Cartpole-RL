from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Model(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Model, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions

        self.embed = nn.Embedding(self.n_states, 128)
        self.fc1 = nn.Linear(128, 256)
        self.value = nn.Linear(256, 1)
        self.policy = nn.Linear(256, self.n_actions)

    def forward(self, inputs):
        x = inputs
        x = self.embed(x)
        x = F.silu(self.fc1(x))
        value = self.value(x)
        dist = Categorical(logits=self.policy(x))
        return dist, value
