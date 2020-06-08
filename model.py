from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, n_states, n_actions,):
        super(Model, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions

        self.fc1 = nn.Linear(self.n_states, 128)
        self.adv_fc = nn.Linear(128, 256)
        self.value_fc = nn.Linear(128, 256)
        self.adv = nn.Linear(256, self.n_actions)
        self.value = nn.Linear(256, 1)

        nn.init.kaiming_normal_(self.fc1.weight)
        self.fc1.bias.data.zero_()
        nn.init.kaiming_normal_(self.adv_fc.weight)
        self.adv_fc.bias.data.data.zero_()
        nn.init.kaiming_normal_(self.value_fc.weight)
        self.value_fc.bias.data.data.zero_()

        nn.init.xavier_uniform_(self.adv.weight)
        self.adv.bias.data.zero_()
        nn.init.xavier_uniform_(self.value.weight)
        self.value.bias.data.zero_()

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.fc1(x))
        adv_fc = F.relu(self.adv_fc(x))
        value_fc = F.relu(self.value_fc(x))
        adv = self.adv(adv_fc)
        value = self.value(value_fc)

        q_value = value + adv - adv.mean(1, keepdim=True)
        return q_value

