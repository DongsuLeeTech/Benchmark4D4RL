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

from stable_baselines3.common.vec_env import SubprocVecEnv
from Algos.BC import BC
from Utils.utils import *
from tqdm import tqdm
import d4rl

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="Parse argument used when running a Flow simulation.",
    epilog="python simulate.py EXP_CONFIG")

# network and dataset setting
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=0,)  # random seed
parser.add_argument('--dataset', type=str, default=None)  # path to datset
parser.add_argument('--env', type=str, default=None)
parser.add_argument('--load_model', type=str, default=None,)  # path to load the saved model
parser.add_argument('--max-size', type=int, default=1e7)
parser.add_argument('--logdir', type=str, default='./results/',)  # tensorboardx logs directory

# OfflineRL parameter
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch', type=int, default=256)  # batch size to update
parser.add_argument('--tau', type=float, default=0.005)
parser.add_argument('--discount', type=float, default=0.99,)  # discounted factor
parser.add_argument('--num-evaluations', type=int, default=10)
parser.add_argument('--max-ep-len', type=int, default=1000)
parser.add_argument('--itr', type=int, default=10000)

# BC algorithm parameter
parser.add_argument('--l2_rate', type=float, default=1e-3,)
parser.add_argument('--lr', type=float, default=3e-04,)
parser.add_argument('--frac', type=float, default=1.0)

parser.add_argument('--normalize', action='store_true')

args = parser.parse_args()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(args.device)

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def main(args, log_dir, replay_buffer, vec_env):

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
    policy = BC(args, state_dim, act_dim)

    done = True

    reward_list = []
    for ep in tqdm(range(args.epochs)):

        train_loss = policy.train(replay_buffer, args.itr)

        evaluations = []
        timesteps = []
        for _ in range(1):

            env.seed(args.seed)
            tot_reward = 0.
            state, done = vec_env.reset(), False

            unfinished = np.ones(num_envs).astype(bool)
            episode_return = np.zeros((num_envs, 1)).astype(float)
            timesteps = torch.tensor([0] * num_envs, device=args.device, dtype=torch.long).reshape(
                num_envs, -1
            )
            ts = 0
            while ts <= args.max_ep_len:
                # env.render()
                state = (state - args.state_mean) / args.state_std
                action = policy.select_action(state)
                next_state, reward, done, _ = vec_env.step(action)
                tot_reward += reward
                state = next_state

                episode_return[unfinished] += reward[unfinished].reshape(-1, 1)

                timesteps = torch.cat(
                    [
                        timesteps,
                        torch.ones((num_envs, 1), device=args.device, dtype=torch.long).reshape(
                            num_envs, 1
                        )
                        * (ts + 1),
                    ],
                    dim=1,
                )

                ts += 1

                if ts == max_ep_len - 1:
                    done = np.ones(done.shape).astype(bool)

                if np.any(done):
                    ind = np.where(done)[0]
                    unfinished[ind] = False

                if not np.any(unfinished):
                    break

            evaluations.append(np.mean(episode_return))

        eval_reward = np.mean(evaluations)

        print('----------------------------------------------------------------------------------------')
        print('# epoch: {} # avg.reward: {} # train_loss: {}'.format(ep, eval_reward,
                                                                                                 train_loss))
        print('----------------------------------------------------------------------------------------')

        summary.add_scalar('loss', train_loss, ep)
        summary.add_scalar('avg_reward', eval_reward, ep)
        reward_list.append(eval_reward)

        policy.save(f'./{log_dir}/{setting}', ep)

    np.save(f'./{log_dir}/reward_{setting}', np.array(reward_list), allow_pickle=True)
    summary.close()

def save_checkpoint(state, filename):
    torch.save(state, filename)

def keep_best_trajectories(
    dataset: Dict[str, np.ndarray],
    frac: float,
    discount: float,
    max_episode_steps: int = 1000,
):
    ids_by_trajectories = []
    returns = []
    cur_ids = []
    cur_return = 0
    reward_scale = 1.0
    for i, (reward, done) in enumerate(zip(dataset["rewards"], dataset["terminals"])):
        cur_return += reward_scale * reward
        cur_ids.append(i)
        reward_scale *= discount
        if done == 1.0 or len(cur_ids) == max_episode_steps:
            ids_by_trajectories.append(list(cur_ids))
            returns.append(cur_return)
            cur_ids = []
            cur_return = 0
            reward_scale = 1.0

    sort_ord = np.argsort(returns, axis=0)[::-1].reshape(-1)
    top_trajs = sort_ord[: int(frac * len(sort_ord))]

    order = []
    for i in top_trajs:
        order += ids_by_trajectories[i]
    order = np.array(order)

    dataset["observations"] = dataset["observations"][order]
    dataset["actions"] = dataset["actions"][order]
    dataset["next_observations"] = dataset["next_observations"][order]
    dataset["rewards"] = dataset["rewards"][order]
    dataset["terminals"] = dataset["terminals"][order]

def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std

def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std

if __name__=="__main__":
    FLOW = ['highway', 'cutin', 'lanereduction']

    seed_list = [0, 1, 2, 3, 4]
    env_list = ['hopper']
    dataset_list = ['medium-expert', 'medium-replay', 'full-replay']

    for e in env_list:
        args.env = e
        for d in dataset_list:
            args.dataset = d
            for seed in seed_list:
                args.seed = seed

                print(f'--------------------Env: {args.env}--------------------')
                print(f'--------------------Dataset: {args.dataset}--------------------')

                buffer_name = f"{args.dataset}"
                set_seed(args.seed)

                if args.env not in FLOW:
                    env_name = f'{args.env}-{args.dataset}-v2'
                    env = gym.make(env_name)

                    state_dim = env.observation_space.shape[0]
                    act_dim = env.action_space.shape[0]

                    dataset = d4rl.qlearning_dataset(env)
                    keep_best_trajectories(dataset, args.frac, args.discount)

                    if args.normalize:
                        args.state_mean, args.state_std = compute_mean_std(dataset["observations"], eps=1e-3)
                    else:
                        args.state_mean, args.state_std = 0, 1

                    dataset["observations"] = normalize_states(
                        dataset["observations"], args.state_mean, args.state_std
                    )
                    dataset["next_observations"] = normalize_states(
                        dataset["next_observations"], args.state_mean, args.state_std
                    )

                    log_dir_name = str(args.dataset) + '_Seed' + str(args.seed) + '_Batch' + str(
                        args.batch) + 'lr' + str(args.lr) + f'_{args.env}-{args.dataset}'
                    from datetime import datetime

                    date = datetime.today().strftime("[%Y|%m|%d|%H:%M:%S]")
                    log_dir = args.logdir + date + log_dir_name + f'{args.frac}' + '_BC'
                    os.mkdir(log_dir)
                    # Tensorboard: Easy Visualization Tool in PyTorch
                    summary = SummaryWriter(log_dir)
                    filename = './' + log_dir + '/arguments.txt'
                    print(vars(args))
                    f = open(filename, 'w')
                    f.write(str(vars(args)))
                    f.close()

                    replay_buffer = ReplayBuffer_d4rl(state_dim=state_dim,
                        action_dim=act_dim,
                        buffer_size=int(args.max_size),
                        device=args.device)

                    eval_envs = SubprocVecEnv(
                        [
                            get_env_builder(i, env_name=f'{args.env}-{args.dataset}-v2', target_goal=None)
                            for i in range(args.num_evaluations)
                        ]
                    )

                    replay_buffer.load_d4rl_dataset(dataset)

                    print('-----------------------------------------------------')
                    main(args, log_dir, replay_buffer, eval_envs)
                    print('-------------------DONE OFFLINE RL-------------------')
