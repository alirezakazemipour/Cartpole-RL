from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, n_states, n_actions, n_atoms, support):
        super(Model, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.support = support

        self.fc1 = nn.Linear(self.n_states, 128)
        self.adv_fc = nn.Linear(128, 512)
        self.adv = nn.Linear(512, self.n_actions * self.n_atoms)

        self.value_fc = nn.Linear(128, 512)
        self.value = nn.Linear(512, self.n_atoms)

        nn.init.kaiming_normal_(self.adv_fc.weight, nonlinearity="relu")
        self.adv_fc.bias.data.zero_()
        nn.init.xavier_uniform_(self.adv.weight)
        self.adv.bias.data.zero_()

        nn.init.kaiming_normal_(self.value_fc.weight, nonlinearity="relu")
        self.value_fc.bias.data.zero_()
        nn.init.xavier_uniform_(self.value.weight)
        self.value.bias.data.zero_()

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.fc1(x))
        adv_fc = F.relu(self.adv_fc(x))
        adv = self.adv(adv_fc).view(-1, self.n_actions, self.n_atoms)
        value_fc = F.relu(self.value_fc(x))
        value = self.value(value_fc).view(-1, 1, self.n_atoms)

        mass_probs = value + adv - adv.mean(1, keepdim=True)
        return F.softmax(mass_probs, dim=-1)

    def get_q_value(self, x):
        dist = self(x)
        q_value = (dist * self.support).sum(-1)
        return q_value
