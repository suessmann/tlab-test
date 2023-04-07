import torch
import numpy as np

from typing import Any, Dict, List, Optional, Tuple, Union

TensorBatch = List[torch.Tensor]


class TensorDeque:
    def __init__(self, maxlen: int, dim: int, dtype: torch.dtype, device: str):
        self._data = torch.zeros((maxlen, dim), dtype=dtype, device=device)
        self._pointer = 0
        self.maxlen = maxlen
        self._is_full = False

    def add_data(self, data: torch.Tensor):
        assert data.shape[0] > 0, "Trying to add empty data"
        idx_to_add = (
            torch.arange(self._pointer, self._pointer + data.shape[0]) % self.maxlen
        )
        self._data[idx_to_add] = data

        if self._pointer + data.shape[0] >= self.maxlen:
            self._is_full = True

        self._pointer = (self._pointer + data.shape[0]) % self.maxlen

    def size(self):
        size = self.maxlen if self._is_full else self._pointer
        return size

    def __getitem__(self, item):
        return self._data[item]


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        # self._pointer = 0
        self._size = 0

        self._states = TensorDeque(
            buffer_size, state_dim, dtype=torch.float32, device=device
        )
        self._actions = TensorDeque(
            buffer_size, action_dim, dtype=torch.float32, device=device
        )
        self._rewards = TensorDeque(buffer_size, 1, dtype=torch.float32, device=device)
        self._next_states = TensorDeque(
            buffer_size, state_dim, dtype=torch.float32, device=device
        )
        self._dones = TensorDeque(buffer_size, 1, dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]

        self.add_transitions(data)
        self._size = self._states.size()
        # self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        current_size = self._states.size()
        indices = np.random.randint(0, current_size, size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transitions(self, data: Dict[str, np.ndarray]):
        if data["observations"].shape[0] > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )

        self._states.add_data(self._to_tensor(data["observations"]))
        self._actions.add_data(self._to_tensor(data["actions"]))
        self._rewards.add_data(self._to_tensor(data["rewards"][..., None]))
        self._next_states.add_data(self._to_tensor(data["next_observations"]))
        self._dones.add_data(self._to_tensor(data["terminals"][..., None]))

        self._check_consistency()
        self._size = self._states.size()

    def _check_consistency(self):
        assert (
            self._states._pointer
            == self._actions._pointer
            == self._rewards._pointer
            == self._next_states._pointer
            == self._dones._pointer
        ), "Pointers are not consistent!"


if __name__ == "__main__":
    deque = TensorDeque(10, 1, torch.float32, "cpu")
    deque.add_data(torch.ones(9, 1))

    a = 0
