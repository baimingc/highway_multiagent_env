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
# from utils import record_videos,  capture_intermediate_frames

import argparse
import torch
import time
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
# from utils.make_env import make_env
from utils.buffer import ReplayBuffer
# from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.maddpg import MADDPG


USE_CUDA = True  # torch.cuda.is_available()
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"


def run(config):
    model_dir = Path('./models')/ config.model_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    env = gym.make("intersection-multiagent-v0")

    maddpg = MADDPG.init_from_env(env, agent_alg=config.agent_alg,
                                  adversary_alg=config.adversary_alg,
                                  tau=config.tau,
                                  lr=config.lr,
                                  hidden_dim=config.hidden_dim)
    
    replay_buffer = ReplayBuffer(config.buffer_length, maddpg.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])
    t = 0
    delay_step = config.delay_step
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + config.n_rollout_threads,
                                        config.n_episodes))
        obs = env.reset()
        # obs.shape = (n_rollout_threads, nagent)(nobs), nobs differs per agent so not tensor
        maddpg.prep_rollouts(device='gpu')

        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        maddpg.reset_noise()
        
        agent_obs = []
        for i in range(4):
            agent_obs.append(np.array([obs[i%4], obs[(i+1)%4], obs[(i+2)%4], obs[(i+3)%4]]).flatten())
        obs = np.array([agent_obs])
        zero_agent_actions = [1,1,1,1]
        last_agent_actions = [zero_agent_actions for _ in range(delay_step)]

        for et_i in range(config.episode_length):
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [torch.FloatTensor(np.vstack(obs[:, i])) for i in range(maddpg.nagents)]
            # get actions as torch Variables
#             print(obs)
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # print(agent_actions)
            # rearrange actions to be per environment
            if delay_step == 0:
                actions = [np.argmax(agent_actions[i][0]) for i in range(4)]
            else:
                future_actions = [np.argmax(agent_actions[i][0]) for i in range(4)]
                actions = last_agent_actions[0]
                last_agent_actions = last_agent_actions[1:]
                last_agent_actions.append(future_actions)
            next_obs, rewards, dones, infos = env.step(actions)
#             print(rewards)
            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            if dones[0][0]:
                break

            obs = next_obs
            t += config.n_rollout_threads
            if (len(replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                if USE_CUDA:
                    maddpg.prep_training(device='gpu')
                else:
                    maddpg.prep_training(device='cpu')
                for u_i in range(config.n_rollout_threads):

                    for a_i in range(maddpg.nagents): #do not update the runner
                        sample = replay_buffer.sample(config.batch_size,
                                                      to_gpu=USE_CUDA)
                        maddpg.update(sample, a_i, logger=logger)
                    maddpg.update_all_targets()
                    maddpg.prep_rollouts(device='gpu')
        ep_rews = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads)
        for a_i, a_ep_rew in enumerate(ep_rews):
            # logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)
            logger.add_scalars('agent%i/mean_episode_rewards' % a_i, {'reward': a_ep_rew}, ep_i)

        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            maddpg.save(run_dir / 'model.pt')

    maddpg.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name",
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--seed",
                        default=1, type=int,
                        help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e7), type=int)
    parser.add_argument("--n_episodes", default=20000, type=int)
    parser.add_argument("--episode_length", default=100, type=int) #25
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--delay_step", default=1, type=int)
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=2000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--adversary_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--discrete_action",
                        action='store_true')

    config = parser.parse_args()

    run(config)
