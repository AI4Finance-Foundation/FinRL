from __future__ import annotations

from collections import deque

import numpy as np
from torch.utils.data.dataset import IterableDataset


class PVM:
    def __init__(self, capacity, portfolio_size):
        """Initializes portfolio vector memory.

        Args:
          capacity: Max capacity of memory.
          portfolio_size: Portfolio size.
        """
        # initially, memory will have the same actions
        self.capacity = capacity
        self.portfolio_size = portfolio_size
        self.reset()

    def reset(self):
        self.memory = [np.array([1] + [0] * self.portfolio_size, dtype=np.float32)] * (
            self.capacity + 1
        )
        self.index = 0  # initial index to retrieve data

    def retrieve(self):
        last_action = self.memory[self.index]
        self.index = 0 if self.index == self.capacity else self.index + 1
        return last_action

    def add(self, action):
        self.memory[self.index] = action


class ReplayBuffer:
    def __init__(self, capacity):
        """Initializes replay buffer.

        Args:
          capacity: Max capacity of buffer.
        """
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        """Represents the size of the buffer

        Returns:
          Size of the buffer.
        """
        return len(self.buffer)

    def append(self, experience):
        """Append experience to buffer. When buffer is full, it pops
           an old experience.

        Args:
          experience: experience to be saved.
        """
        self.buffer.append(experience)

    def sample(self):
        """Sample from replay buffer. All data from replay buffer is
        returned and the buffer is cleared.

        Returns:
          Sample of batch_size size.
        """
        buffer = list(self.buffer)
        self.buffer.clear()
        return buffer


class RLDataset(IterableDataset):
    def __init__(self, buffer):
        """Initializes reinforcement learning dataset.

        Args:
            buffer: replay buffer to become iterable dataset.

        Note:
            It's a subclass of pytorch's IterableDataset,
            check https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
        """
        self.buffer = buffer

    def __iter__(self):
        """Iterates over RLDataset.

        Returns:
          Every experience of a sample from replay buffer.
        """
        yield from self.buffer.sample()
