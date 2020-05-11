import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4, padding=0),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4, padding=0),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=3, padding=0),
        )
            
        
        #128 input features, 64 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(16*2*2, 32)
        self.fc_val = nn.Linear(2, 10)
        self.fc2 = nn.Linear(42, 16)
        self.fc3 = nn.Linear(16, 3)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, state_img, state_val):
        x = self.feature(state_img)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        y = F.relu(self.fc_val(state_val))
        x = torch.cat((x, y), dim=1)
        x = F.relu(self.fc2(x))
        z = self.fc4(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), z