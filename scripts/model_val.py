import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        #128 input features, 64 output features (see sizing flow below)
        self.fc1 = nn.Linear(8*4, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 3)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=0)
    
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        #128 input features, 64 output features (see sizing flow below)
        self.fc1 = nn.Linear(8*4, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(8*4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            nn.Linear(8*4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
#         dist  = Categorical(probs)
        return probs, value