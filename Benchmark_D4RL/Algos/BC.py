import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Algos.model import *
from tqdm import tqdm
import math

class BC(object):
    def __init__(self,
                 args,
                 state_dim,
                 action_dim):

        self.device = args.device
        self.batch = args.batch
        self.lr = args.lr

        self.BC_net = MLP(state_dim, action_dim).to(self.device)
        self.optim = torch.optim.Adam(self.BC_net.parameters(), lr=self.lr)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        return self.BC_net.forward(state).detach().cpu().numpy()

    def train(self, replay_buffer, iterations):
        self.BC_net.train()
        train_loss = 0

        for it in tqdm(range(iterations)):
            # Sample replay buffer / batch
            state, action, _, _, _ = replay_buffer.sample(self.batch)
            data = state.to(self.device)
            target = action.to(self.device)
            pred = self.BC_net.forward(data)

            loss = F.mse_loss(pred, target)

            for param in self.BC_net.parameters():
                param.grad = None

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            train_loss += loss.item()

        train_loss /= iterations
        return train_loss

    def save(self, filename, ep):
        torch.save(self.BC_net.state_dict(), filename + f'_{ep}' + '_BC')

    def load(self, filename):
        self.BC_net.load_state_dict(torch.load(filename, map_location=self.device))
        self.BC_net.to(self.device)
