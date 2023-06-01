import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from Algos.model import *

from torch.distributions import MultivariateNormal

class GaussianBC(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(GaussianBC, self).__init__()

        self.l1 = nn.Linear(state_dim, 512)
        self.l2 = nn.Linear(512, 512)
        self.mean = nn.Linear(512, action_dim)

        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))

        self.max_action = max_action

        self.logstd_min = -10.
        self.logstd_max = 2.

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))

        mean = torch.tanh(self.mean(a))
        std = torch.exp(self.log_std.clamp(self.logstd_min, self.logstd_max))
        scale_tril = torch.diag(std)

        return MultivariateNormal(mean, scale_tril=scale_tril)

    def act(self, state, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            dist = self.forward(state)
            action = dist.mean if deterministic else dist.sample()
            action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
            return action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(torch.cat([state, action], 1)))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def q1(self, state, action):
        if state.dim() == 3:
            q1 = F.relu(self.l1(torch.cat([state, action], 2)))
        else:
            q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

class OnestepRL(object):
    def __init__(self, args, state_dim, action_dim, max_action):

        self.device = args.device

        # Initial Policy - Dubbed Beta in OneStepRL Paper
        self.bc= GaussianBC(state_dim, action_dim, max_action).to(self.device)
        self.bc_optimizer = torch.optim.Adam(self.bc.parameters(), lr=args.bc_lr)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.q_lr)

        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = args.discount
        self.tau = args.tau
        self.lmbda = args.lmbda
        self.batch = args.batch
        self.n = args.n_sample

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            state = state.unsqueeze(0).repeat(self.n, 1, 1)
            action = self.bc(state).sample()
            q1 = self.critic.q1(state, action)
            ind = q1.argmax(0)
            action = action.reshape(action.shape[1], self.n, action.shape[2])
            action = action[torch.arange(action.size(0)).unsqueeze(1), ind].squeeze(1)
            action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action.cpu().data.numpy()

    def pretrain_bc(self, replay_buffer, iterations):
        tot_loss = 0

        for it in tqdm(range(iterations)):
            s, a, _, _, _ = replay_buffer.sample(self.batch)
            policy = self.bc.forward(s)
            log_prob = policy.log_prob(a)
            loss = (-log_prob).mean()

            self.bc_optimizer.zero_grad()
            loss.backward()
            self.bc_optimizer.step()

            tot_loss += loss.item()

        tot_loss /= iterations
        return tot_loss

    def q_train(self, replay_buffer, iterations):
        tot_critic_loss = 0

        for it in tqdm(range(iterations)):
            s, a, r, ns, d = replay_buffer.sample(self.batch)

            # Critic Training
            with torch.no_grad():
                # Duplicate next state 10 times
                ns = torch.repeat_interleave(ns, self.n, 0)

                # Compute value of perturbed actions sampled from the VAE
                target_Q1, target_Q2 = self.critic_target(ns, self.bc(ns).rsample())

                # Soft Clipped Double Q-learning
                target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + (1. - self.lmbda) * torch.max(target_Q1,
                                                                                                        target_Q2)
                target_Q = target_Q.reshape(self.batch, -1).max(1)[0].reshape(-1, 1)

                target_Q = r + (1 - d) * self.discount * target_Q

            current_Q1, current_Q2 = self.critic(s, a)

            q1_loss = F.mse_loss(current_Q1, target_Q)
            q2_loss = F.mse_loss(current_Q2, target_Q)
            critic_loss = q1_loss + q2_loss

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            tot_critic_loss += critic_loss

            # Update Target Networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        tot_critic_loss /= iterations

        return tot_critic_loss.detach().cpu().item()

    def save(self, filename, ep):
        torch.save(self.critic.state_dict(), filename + f'{ep}' + "_critic")
        torch.save(self.critic_target.state_dict(), filename + f'{ep}' + "_critic_target")
        torch.save(self.bc.state_dict(), filename + f'{ep}' + "_bc")

    def load(self, filename, ep):
        self.critic.load_state_dict(torch.load(filename + f'{ep}' + "_critic", map_location=self.device))
        self.critic_target.load_state_dict(torch.load(filename + f'{ep}' + "_critic_target", map_location=self.device))

