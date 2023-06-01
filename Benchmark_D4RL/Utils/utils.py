from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union
from collections import defaultdict
import os
import gym
import numpy as np
import torch
import random
from tqdm.auto import tqdm, trange

from torch.utils.data import DataLoader, IterableDataset

from typing import cast
class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, device, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.Gt = np.zeros((max_size, 1))

        self.device = device

    def add(self, state, action, next_state, reward, done, Gt):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.Gt[self.ptr] = Gt

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        self.pointer = 0

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.Gt[ind]).to(self.device)
        )

    def load(self, save_folder, size=-1):
        reward_buffer = np.load(f"{save_folder}/reward.npy", allow_pickle=True)

        # Adjust crt_size if we're using a custom size
        size = min(int(size), self.max_size) if size > 0 else self.max_size
        self.size = min(reward_buffer.shape[0], size)
        print(np.load(f"{save_folder}/action.npy", allow_pickle=True).shape)
        self.state[:self.size] = np.load(f"{save_folder}/state.npy", allow_pickle=True)[:self.size]
        self.action[:self.size] = np.load(f"{save_folder}/action.npy", allow_pickle=True)[:self.size]
        self.next_state[:self.size] = np.load(f"{save_folder}/next_state.npy", allow_pickle=True)[:self.size]
        # self.reward[:self.size] = reward_buffer.reshape(self.size, 1)[:self.size]
        self.not_done[:self.size] = np.load(f"{save_folder}/done.npy", allow_pickle=True)[:self.size]
        self.Gt[:self.size] = np.load(f"{save_folder}/return.npy", allow_pickle=True)[:self.size]

    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self.size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self.max_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self.state[:n_transitions] = self._to_tensor(data["observations"])
        self.action[:n_transitions] = self._to_tensor(data["actions"])
        self.reward[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self.next_state[:n_transitions] = self._to_tensor(data["next_observations"])
        self.not_done[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self.size += n_transitions
        self.pointer = min(self.size, n_transitions)

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self.device)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def env_set_seed(env, seed):
    env.seed(seed)
    env.action_space.seed(seed)

def get_env_builder(seed, env_name, target_goal=None):
    def make_env_fn():
        import d4rl

        env = gym.make(env_name)
        env.seed(seed)
        if hasattr(env.env, "wrapped_env"):
            env.env.wrapped_env.seed(seed)
        elif hasattr(env.env, "seed"):
            env.env.seed(seed)
        else:
            pass
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        if target_goal:
            env.set_target_goal(target_goal)
            print(f"Set the target goal to be {env.target_goal}")
        return env

    return make_env_fn


TensorBatch = List[torch.Tensor]
class ReplayBuffer_d4rl:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return states, actions, rewards, next_states, dones

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError


def pad_along_axis(
    arr: np.ndarray, pad_to: int, axis: int = 0, fill_value: float = 0.0
) -> np.ndarray:
    pad_size = pad_to - arr.shape[axis]
    if pad_size <= 0:
        return arr

    npad = [(0, 0)] * arr.ndim
    npad[axis] = (0, pad_size)
    return np.pad(arr, pad_width=npad, mode="constant", constant_values=fill_value)


def discounted_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    cumsum = np.zeros_like(x)
    cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        cumsum[t] = x[t] + gamma * cumsum[t + 1]
    return cumsum


def load_d4rl_trajectories(
    env_name: str, gamma: float = 1.0
) -> Tuple[List[DefaultDict[str, np.ndarray]], Dict[str, Any]]:
    dataset = gym.make(env_name).get_dataset()
    traj, traj_len = [], []

    data_ = defaultdict(list)
    for i in trange(dataset["rewards"].shape[0], desc="Processing trajectories"):
        data_["observations"].append(dataset["observations"][i])
        data_["actions"].append(dataset["actions"][i])
        data_["rewards"].append(dataset["rewards"][i])

        if dataset["terminals"][i] or dataset["timeouts"][i]:
            episode_data = {k: np.array(v, dtype=np.float32) for k, v in data_.items()}
            # return-to-go if gamma=1.0, just discounted returns else
            episode_data["returns"] = discounted_cumsum(
                episode_data["rewards"], gamma=gamma
            )
            traj.append(episode_data)
            traj_len.append(episode_data["actions"].shape[0])
            # reset trajectory buffer
            data_ = defaultdict(list)

    # needed for normalization, weighted sampling, other stats can be added also
    info = {
        "obs_mean": dataset["observations"].mean(0, keepdims=True),
        "obs_std": dataset["observations"].std(0, keepdims=True) + 1e-6,
        "traj_lens": np.array(traj_len),
    }
    return traj, info

def load_d4rl_dataset(data: Dict[str, np.ndarray], size: int, buffer_size: int):
    states, actions, rewards, next_states, dones = [], [], [], [], []
    if size != 0:
        raise ValueError("Trying to load data into non-empty replay buffer")
    n_transitions = data["observations"].shape[0]
    if n_transitions > buffer_size:
        raise ValueError(
            "Replay buffer is smaller than the dataset you are trying to load!"
        )

    states[:n_transitions] = data["observations"]
    actions[:n_transitions] = data["actions"]
    rewards[:n_transitions] = data["rewards"][..., None]
    next_states[:n_transitions] = data["next_observations"]
    dones[:n_transitions] = data["terminals"][..., None]
    size += n_transitions
    pointer = min(size, n_transitions)

    return (states, actions, rewards, next_states, dones), size, pointer

class D4RLIterDataset(IterableDataset):
    def __init__(self, data: dict, buffer_size: int):
        self.dataset, size, pointer = load_d4rl_dataset(data, 0, buffer_size)
        self.size = size
        self.ptr = pointer

    def __prepare_sample(self, idx):
        states = self.dataset[0][idx]
        actions = self.dataset[1][idx]
        rewards = self.dataset[2][idx]
        next_states = self.dataset[3][idx]
        dones = self.dataset[4][idx] * 1
        return states, actions, rewards, next_states, dones

    def __iter__(self):
        while True:
            idx = random.randint(0, self.size - 1)
            yield self.__prepare_sample(idx)

class SequenceDataset(IterableDataset):
    def __init__(self, env_name: str, seq_len: int = 10, reward_scale: float = 1.0):
        self.dataset, info = load_d4rl_trajectories(env_name, gamma=1.0)
        self.reward_scale = reward_scale
        self.seq_len = seq_len

        self.state_mean = info["obs_mean"]
        self.state_std = info["obs_std"]
        # https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/experiment.py#L116 # noqa
        self.sample_prob = info["traj_lens"] / info["traj_lens"].sum()

    def __prepare_sample(self, traj_idx, start_idx):
        traj = self.dataset[traj_idx]
        # https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/experiment.py#L128 # noqa
        states = traj["observations"][start_idx : start_idx + self.seq_len]
        actions = traj["actions"][start_idx : start_idx + self.seq_len]
        returns = traj["returns"][start_idx : start_idx + self.seq_len]
        time_steps = np.arange(start_idx, start_idx + self.seq_len)

        states = (states - self.state_mean) / self.state_std
        returns = returns * self.reward_scale
        # pad up to seq_len if needed
        mask = np.hstack(
            [np.ones(states.shape[0]), np.zeros(self.seq_len - states.shape[0])]
        )
        if states.shape[0] < self.seq_len:
            states = pad_along_axis(states, pad_to=self.seq_len)
            actions = pad_along_axis(actions, pad_to=self.seq_len)
            returns = pad_along_axis(returns, pad_to=self.seq_len)

        return states, actions, returns, time_steps, mask

    def __iter__(self):
        while True:
            traj_idx = np.random.choice(len(self.dataset), p=self.sample_prob)
            start_idx = random.randint(0, self.dataset[traj_idx]["rewards"].shape[0] - 1)
            yield self.__prepare_sample(traj_idx, start_idx)