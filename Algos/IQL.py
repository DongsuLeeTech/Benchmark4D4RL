import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

from torch.distributions import MultivariateNormal
from torch.optim.lr_scheduler import CosineAnnealingLR

def asymmetric_l2_loss(u, alpha):
    return torch.mean(torch.abs(alpha - (u < 0).float()) * u ** 2)

class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(GaussianPolicy, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)

        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))

        self.max_action = max_action

        self.logstd_min = -5.
        self.logstd_max = 2.

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))

        mean = torch.tanh(self.mean(a))
        std = torch.exp(self.log_std.clamp(self.logstd_min, self.logstd_max)) * 10.0
        scale_tril = torch.diag(std)

        return MultivariateNormal(mean, scale_tril=scale_tril)

    def act(self, state, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            dist = self.forward(state)
            action = dist.mean if deterministic else dist.sample()
            action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
            return action

class DeterministicPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DeterministicPolicy, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.mean(a)

    def act(self, state, deterministic=True, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            return self(state)


class TwinQ(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(TwinQ, self).__init__()

        self.q1l1 = nn.Linear(state_dim + action_dim, 256)
        self.q1l2 = nn.Linear(256, 256)
        self.q1l3 = nn.Linear(256, 1)

        self.q2l1 = nn.Linear(state_dim + action_dim, 256)
        self.q2l2 = nn.Linear(256, 256)
        self.q2l3 = nn.Linear(256, 1)

    def both(self, state, action):
        sa = torch.cat([state, action], 1)

        a = F.relu(self.q1l1(sa))
        a = F.relu(self.q1l2(a))

        b = F.relu(self.q1l1(sa))
        b = F.relu(self.q1l2(b))

        return self.q1l3(a), self.q2l3(b)

    def forward(self, state, action):
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim):
        super(ValueFunction, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.l3(a)


class ImplicitQLearning(object):
    def __init__(self, args, state_dim, action_dim, max_action):
        self.device = args.device

        self.q = TwinQ(state_dim, action_dim).to(self.device)
        self.q_target = copy.deepcopy(self.q)
        self.q_optimizer = torch.optim.Adam(self.q.parameters(), lr=args.q_lr)

        self.v = ValueFunction(state_dim).to(self.device)
        self.v_optimizer = torch.optim.Adam(self.v.parameters(), lr=args.v_lr)

        self.actor = DeterministicPolicy(state_dim, action_dim).to(self.device) if args.deterministic \
            else GaussianPolicy(state_dim, action_dim, max_action).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, args.epochs * args.itr)

        self.tau = args.tau
        self.beta = args.beta
        self.batch = args.batch
        self.discount = args.discount
        self.alpha = args.alpha
        self.target_update_interval = args.targ_update_freq

        self.exp_adv_max = 100.

        self.it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        return self.actor.act(state, deterministic=True).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations):
        q_tot_loss, v_tot_loss, actor_tot_loss = 0, 0, 0
        for it in tqdm(range(iterations)):
        # Sample replay buffer / batch
            state, action, reward, next_state, not_done = replay_buffer.sample(self.batch)

            with torch.no_grad():
                qft = self.q_target.forward(state, action)
                v_next = self.v(next_state)

            v = self.v(state)
            adv = qft - v
            v_loss = asymmetric_l2_loss(adv, self.alpha)

            self.v_optimizer.zero_grad(set_to_none=True)
            v_loss.backward()
            self.v_optimizer.step()

            target = reward + (1 - not_done) * self.discount * v_next.detach()
            qt = self.q.both(state, action)
            q_loss = sum(F.mse_loss(q, target) for q in qt) / len(qt)

            self.q_optimizer.zero_grad(set_to_none=True)
            q_loss.backward()
            self.q_optimizer.step()

            exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=self.exp_adv_max)
            policy_out = self.actor.forward(state)
            if isinstance(policy_out, torch.distributions.Distribution):
                loss = -policy_out.log_prob(action)
            elif torch.is_tensor(policy_out):
                assert policy_out.shape == action.shape
                loss = torch.sum((policy_out - action) ** 2, dim=1)
            else:
                raise NotImplementedError

            actor_loss = torch.mean(exp_adv * loss)
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            self.actor_lr_schedule.step()

            self.it += 1.

            if self.it % self.target_update_interval == 0:
                for target_param, param in zip(self.q_target.parameters(), self.q.parameters()):
                    target_param.data.mul_(1. - self.tau).add_(param.data, alpha=self.tau)

            q_tot_loss += q_loss
            v_tot_loss += v_loss
            actor_tot_loss += actor_loss

        q_tot_loss /= iterations
        v_tot_loss /= iterations
        actor_tot_loss /= iterations

        return q_tot_loss.detach().cpu().item(), v_tot_loss.detach().cpu().item(), \
            actor_tot_loss.detach().cpu().item()

    def save(self, filename, ep):
        torch.save(self.actor.state_dict(), filename + f'_{ep}' + '_actor')
        torch.save(self.v.state_dict(), filename + f'_{ep}' + '_v')
        torch.save(self.q.state_dict(), filename + f'_{ep}' + '_q')
        torch.save(self.q_target.state_dict(), filename + f'_{ep}' + '_q_target')

    def load(self, filename, ep):
        self.actor.load_state_dict(torch.load(filename + f'_{ep}' + '_actor', map_location=self.device))
        self.v.load_state_dict(torch.load(filename + f'_{ep}' + '_v', map_location=self.device))
        self.q.load_state_dict(torch.load(filename + f'_{ep}' + '_q', map_location=self.device))
        self.q_target.load_state_dict(torch.load(filename + f'_{ep}' + '_q_target', map_location=self.device))

