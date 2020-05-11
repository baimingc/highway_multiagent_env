import gym
import sys
sys.path.append("/home/baiming/highway_multiagent_env") 
sys.path.append("C://Users//baiming//Seafile//THUFile//Papers//highway_env_multiagent") 
import highway_env

from agent import Agent

import torch
from torch import optim
import numpy as np 

from tqdm import tnrange
from utils import record_videos,  capture_intermediate_frames

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
agent = Agent()
# optimizer = optim.Adam(agent.ac.parameters(),lr=3e-2)
optimizer = optim.SGD(agent.ac.parameters(),lr=3e-2)

env = gym.make("intersection-multiagent-v0")
obs, done = env.reset(), False
# img = env.render(mode='rgb_array')

from agent import compute_returns
from torch.autograd import Variable
sum_rewards = []
losses = []
obs = env.reset()
for e in range(5000):
    log_probs = []
    values    = []
    rewards   = []
    masks     = []
    entropy = 0
    for _ in range(100):
        action = []
        for i in range(len(obs)):
            if i == 0:
                _, ind_action, log_prob, value, ent = agent.act([obs[i%4], obs[(i+1)%4], obs[(i+2)%4], obs[(i+3)%4]])
#                 print(log_prob, value)
                action.append(ind_action)
                log_probs.append(log_prob)
                values.append(value)
                entropy += ent
            else:
                action.append(0)

        next_obs, reward, done, _ = env.step(action)
        rewards.append(torch.FloatTensor([reward]).to(device))
        masks.append(torch.FloatTensor([1 - done]).to(device))
        obs = next_obs
        if done:
            break
    sum_reward = sum(rewards)
    print(sum_reward)
    sum_rewards.append(sum_reward)
    returns = []
    next_value = 0
    _, _, _, next_value, _ = agent.act([obs[i%4], obs[(i+1)%4], obs[(i+2)%4], obs[(i+3)%4]])

    returns = compute_returns(next_value, rewards, masks)
    returns = torch.cat(returns)
    values = torch.cat(values)

    log_probs = torch.stack(log_probs)
    advantage = returns - values
#     print('v', values)
    
    actor_loss  = -(log_probs * advantage.detach()).mean()
    critic_loss = advantage.pow(2).mean()
#     print(actor_loss, critic_loss, entropy)
    loss = actor_loss + 0.05*critic_loss - 0.1 * entropy
    losses.append(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    obs = env.reset()
    
    if e % 100 == 0:
        torch.save(agent.ac.state_dict(), 'ac{}.pth'.format(e))
env.close()