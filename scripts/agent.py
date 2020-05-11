import torch
from torch import optim
import torch.nn.functional as F
from torch.distributions import Categorical
from model_val import ActorCritic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

class Agent(object):
    def __init__(self):
        self.ac = ActorCritic().to(device)

    def act(self, state):
#         state = torch.FloatTensor(state).view(-1).to(device)
        state = torch.FloatTensor(state).reshape(-1).to(device)
#         state = Variable(state, requires_grad=True)
        probs, value = self.ac.forward(state)
#         print(value)
        m = Categorical(probs)
        _, greedy_action = torch.max(probs, 0)
        action = m.sample()
        return greedy_action.item(), action.item(), m.log_prob(action), value, m.entropy().mean()

# class Agent(object):
#     def __init__(self):
#         self.actor = Actor().to(device)
#         self.critic = Critic().to(device)

#     def act(self, state):
#         state = torch.FloatTensor(state).view(-1).to(device)
#         probs = self.actor.forward(state)
#         value = self.critic.forward(state)
#         m = Categorical(probs)
#         _, greedy_action = torch.max(probs, 0)
#         action = m.sample()
#         return greedy_action.item(), action.item(), m.log_prob(action), value, m.entropy().mean()