import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Algos.model import *
from tqdm import tqdm
import math


class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.norm1 = nn.LayerNorm(args.embedding_dim)
        self.norm2 = nn.LayerNorm(args.embedding_dim)
        self.drop = nn.Dropout(args.residual_dropout)

        self.attention = nn.MultiheadAttention(
            args.embedding_dim, args.num_heads, args.attention_dropout, batch_first=True
        )
        self.l1 = nn.Linear(args.embedding_dim, 4 * args.embedding_dim)
        self.l2 = nn.Linear(4 * args.embedding_dim, args.embedding_dim)
        self.d1 = nn.Dropout(args.residual_dropout)
        self.gelu = nn.GELU()

        self.register_buffer('causal_mask', ~torch.tril(torch.ones(args.seq_len * 3, args.seq_len * 3)).to(bool))
        self.seq_len = args.seq_len * 3

    def forward(self, x, padding_mask=None):
        causal_mask = self.causal_mask[:x.shape[1], :x.shape[1]]

        norm_x = self.norm1(x)
        attention_out = self.attention(query=norm_x, key=norm_x, value=norm_x,
                                       attn_mask=causal_mask, key_padding_mask=padding_mask,
                                       need_weights=False)[0]
        x = x + self.drop(attention_out)
        out = self.gelu(self.l1(self.norm2(x)))
        out = self.d1(self.l2(out))
        return x + out


class DecisionTransformer(nn.Module):
    def __init__(self, args, state_dim, action_dim, max_action):
        super().__init__()
        self.emb_drop = nn.Dropout(args.embedding_dropout)
        self.emb_norm = nn.LayerNorm(args.embedding_dim)
        self.out_norm = nn.LayerNorm(args.embedding_dim)

        self.timestep_emb = nn.Embedding(args.episode_len + args.seq_len, args.embedding_dim)
        self.state_emb = nn.Linear(state_dim, args.embedding_dim)
        self.action_emb = nn.Linear(action_dim, args.embedding_dim)
        self.return_emb = nn.Linear(1, args.embedding_dim)

        self.blocks = nn.ModuleList([TransformerBlock(args) for _ in range(args.num_layers)])

        self.action_head = nn.Sequential(nn.Linear(args.embedding_dim, action_dim), nn.Tanh())
        self.seq_len = args.seq_len
        self.embedding_dim = args.embedding_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_len = args.episode_len
        self.max_action = max_action

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, state, action, rtg, time_steps, padding_mask=None):
        batch_size, seq_len = state.shape[0], state.shape[1]
        time_emb = self.timestep_emb(time_steps)
        state_emb = self.state_emb(state) + time_emb
        act_emb = self.action_emb(action) + time_emb
        rtg_emb = self.return_emb(rtg.unsqueeze(-1)) + time_emb

        sequence = (
            torch.stack([rtg_emb, state_emb, act_emb], dim=1).permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_len, self.embedding_dim)
        )

        if padding_mask is not None:
            padding_mask = (
                torch.stack([padding_mask, padding_mask, padding_mask], dim=1).permute(0, 2, 1)
                .reshape(batch_size, 3 * seq_len)
            )

        out = self.emb_norm(sequence)
        out = self.emb_drop(out)

        for block in self.blocks:
            out = block(out, padding_mask=padding_mask)

        out = self.out_norm(out)
        out = self.action_head(out[:, 1::3] * self.max_action)
        return out


class DT(object):
    def __init__(self, args, state_dim, action_dim, max_action, dataset):
        self.device = args.device
        self.model = DecisionTransformer(args, state_dim, action_dim, max_action).to(args.device)
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay, betas=args.betas)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim,
                                                      lambda steps: min((steps + 1) / args.warmup_steps, 1))

        self.seq_len = 3 * args.seq_len
        self.dataset = dataset


    def select_action(self, state, action, target_rtg, timestep):
        predicted_act = self.model.forward(state, action, target_rtg, timestep)
        return predicted_act[0, -1].cpu().data.numpy()

    def train(self, iterations):
        tot_loss = 0.
        trainloader_iter = iter(self.dataset)
        for it in tqdm(range(iterations)):
            batch = next(trainloader_iter)
            s, a, rtg, timestep, mask = [b.to(self.device) for b in batch]
            padding_mask = ~mask.to(torch.bool)

            pred = self.model.forward(s, a, rtg, timestep, padding_mask)
            loss = F.mse_loss(pred, a.detach(), reduction='none')
            loss = (loss * mask.unsqueeze(-1)).mean()

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            self.scheduler.step()

            tot_loss += loss.item()
        tot_loss /= iterations

        return tot_loss

    def save(self, filename, ep):
        torch.save(self.model.state_dict(), filename + f'_{ep}' + '_model')

    def load(self, filename, ep):
        self.model.load_state_dict(torch.load(filename + f'_{ep}' + 'model'))

