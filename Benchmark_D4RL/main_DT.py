from typing import Any, Dict, List, Optional, Tuple, Union

import os
import sys
import gym
import json
import pickle
import timeit
import argparse
import numpy as np
from copy import deepcopy
from time import strftime
from collections import deque

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from stable_baselines3.common.vec_env import SubprocVecEnv
from Algos.DT import DT
from Utils.utils import *
from tqdm import tqdm
import d4rl

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="Parse argument used when running a Flow simulation.",
    epilog="python simulate.py EXP_CONFIG")

# network and dataset setting
parser.add_argument('--env', type=str)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=0,)  # random seed
parser.add_argument('--dataset', type=str, default=None)  # path to datset
parser.add_argument('--load_model', type=str, default=None,)  # path to load the saved model
parser.add_argument('--logdir', type=str, default='./results/',)  # tensorboardx logs directory

# Offline RL parameter
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--max-ep-len', type=int, default=1000)
parser.add_argument('--num-evaluations', type=int, default=5)

# ML parameter
parser.add_argument('--itr', type=int, default=1000)
parser.add_argument('--batch', type=int, default=256)
parser.add_argument('--discount', type=float, default=1.)

# DT parameter
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--warmup-steps', type=int, default=1)
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999))

parser.add_argument('--residual-dropout', type=float, default=0.1)
parser.add_argument('--attention-dropout', type=float, default=0.1)
parser.add_argument('--embedding-dropout', type=float, default=0.1)

parser.add_argument('--seq-len', type=int, default=20)
parser.add_argument('--num-heads', type=int, default=1)
parser.add_argument('--num-layers', type=int, default=3)
parser.add_argument('--episode-len', type=int, default=1000)
parser.add_argument('--embedding-dim', type=int, default=128)

args = parser.parse_args()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(args.device)

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def main(args, log_dir, loader, dataset, env, vec_env):
    setting = f"{args.dataset}_{args.seed}_{args.env}"
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_ep_len = args.max_ep_len
    max_action = env.action_space.high[0]
    num_envs = vec_env.num_envs

    print(env.action_space)
    print('state size:', state_dim)
    print('action size:', act_dim)

    # Initialize and load policy and Q_net
    policy = DT(args, state_dim, act_dim, max_action, loader)

    done = True

    reward_list = []
    for ep in tqdm(range(args.epochs)):
        train_loss = policy.train(args.itr)
        evaluations = []
        timesteps = []
        for target_rtg in args.target_rtg:
            for _ in range(args.num_evaluations):

                tot_reward = 0.
                states = torch.zeros(
                    1, args.episode_len + 1, state_dim, dtype=torch.float, device=args.device
                )
                actions = torch.zeros(
                    1, args.episode_len, act_dim, dtype=torch.float, device=args.device
                )
                returns = torch.zeros(1, args.episode_len + 1, dtype=torch.float, device=args.device)
                time_steps = torch.arange(args.episode_len, dtype=torch.long, device=args.device)
                time_steps = time_steps.view(1, -1)

                states[:, 0] = (torch.as_tensor(env.reset(), device=args.device) -
                                torch.as_tensor(dataset.state_mean, device=args.device)) / \
                               torch.as_tensor(dataset.state_std, device=args.device)
                returns[:, 0] = torch.as_tensor(target_rtg, device=args.device)

                # cannot step higher than model episode len, as timestep embeddings will crash
                episode_return, episode_len = 0.0, 0.0
                for step in range(args.episode_len):

                    predicted_actions = policy.select_action(
                        states[:, : step + 1][:, -args.seq_len:],
                        actions[:, : step + 1][:, -args.seq_len:],
                        returns[:, : step + 1][:, -args.seq_len:],
                        time_steps[:, : step + 1][:, -args.seq_len:],
                    )
                    next_state, reward, done, info = env.step(predicted_actions)
                    # at step t, we predict a_t, get s_{t + 1}, r_{t + 1}
                    actions[:, step] = torch.as_tensor(predicted_actions, device=args.device)
                    states[:, step + 1] = (torch.as_tensor(next_state, device=args.device) - torch.as_tensor(dataset.state_mean, device=args.device)) / \
                         torch.as_tensor(dataset.state_std, device=args.device)
                    returns[:, step + 1] = torch.as_tensor(returns[:, step] - reward, device=args.device)

                    episode_return += reward
                    episode_len += 1

                    if done:
                        break

                evaluations.append(episode_return)

            eval_reward = np.mean(evaluations)

            print('----------------------------------------------------------------------------------------')
            print('# epoch: {} # avg.reward: {} # loss: {}'.format(ep, eval_reward, train_loss))
            print('----------------------------------------------------------------------------------------')

            summary.add_scalar('loss', train_loss, ep)
            summary.add_scalar(f'{target_rtg}_avg_reward', eval_reward, ep)
            reward_list.append(eval_reward)

            policy.save(f'./{log_dir}/{setting}', ep)

    np.save(f'./{log_dir}/reward_{setting}', np.array(reward_list), allow_pickle=True)
    summary.close()

def save_checkpoint(state, filename):
    torch.save(state, filename)

if __name__=="__main__":
    seed_list = [0, 1, 2, 3, 4]
    env_list = ['hopper', 'walker2d', 'halfcheetah']
    dataset_list = ['random', 'medium', 'expert', 'medium-expert', 'medium-replay', 'full-replay']

    for e in env_list:
        args.env = e
        for d in dataset_list:
            args.dataset = d
            for seed in seed_list:
                args.seed = seed

                print(f'--------------------Env: {args.env}-{args.dataset}--------------------')
                print(f'--------------------Seed: {args.seed}--------------------')

                env_name = f'{args.env}-{args.dataset}-v2'
                dataset = args.dataset

                if args.env == 'hopper':
                    args.max_ep_len = 1000
                    args.target_rtg = [3600, 1800]
                elif args.env == 'halfcheetah':
                    args.max_ep_len = 1000
                    args.target_rtg = [12000, 6000]
                elif args.env == 'walker2d':
                    args.max_ep_len = 1000
                    args.target_rtg = [5000, 2500]

                env = gym.make(f'{args.env}-{args.dataset}-v2')
                eval_envs = SubprocVecEnv(
                    [
                        get_env_builder(i, env_name=env_name, target_goal=None)
                        for i in range(args.num_evaluations)
                    ]
                )

                Dataset = SequenceDataset(env_name, seq_len=args.seq_len, reward_scale=1.)
                trainloader = DataLoader(
                    Dataset,
                    batch_size=args.batch,
                    pin_memory=True,
                    num_workers=4,
                )
                # file directory
                log_dir_name = str(args.dataset) + '_Seed' + str(args.seed) + '_Batch' + str(
                    args.batch) + env_name

                from datetime import datetime
                date = datetime.today().strftime("[%Y|%m|%d|%H:%M:%S]")
                log_dir = args.logdir + date + log_dir_name + '_DT'
                os.mkdir(log_dir)
                # Tensorboard: Easy Visualization Tool in PyTorch
                summary = SummaryWriter(log_dir)
                filename = './' + log_dir + '/arguments.txt'
                print(vars(args))
                f = open(filename, 'w')
                f.write(str(vars(args)))
                f.close()

                print('-----------------------------------------------------')
                main(args, log_dir, trainloader, Dataset, env, eval_envs)
                print('-------------------DONE OFFLINE RL-------------------')