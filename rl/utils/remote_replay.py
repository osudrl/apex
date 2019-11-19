import random
import numpy as np
import ray

# tensorboard
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from colorama import Fore, Style

# more efficient replay memory?
from collections import deque

# Code based on:
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

# Expects tuples of (state, next_state, action, reward, done)

@ray.remote
class ReplayBuffer_remote(object):
    def __init__(self, size, experiment_name, args):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self.storage = deque(maxlen=int(size))
        self.max_size = size

        print("Created replay buffer with size {}".format(self.max_size))
    
    def __len__(self):
        return len(self.storage)

    def storage_size(self):
        return len(self.storage)

    def add(self, data):
        self.storage.append(data)

    def add_bulk(self, data):
        for i in range(len(data)):
            self.storage.append(data[i])

    def print_size(self):
        print("size = {}".format(len(self.storage)))

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        # print("Sampled experience from replay buffer.")
        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)

# Non-ray actor for replay buffer
class ReplayBuffer(object):
    def __init__(self, max_size=1e7):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):
        if len(self.storage) < self.max_size:
            self.storage.append(data)
        self.storage[int(self.ptr)] = data
        self.ptr = (self.ptr + 1) % self.max_size
            

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)

    def get_transitions_from_range(self, start, end):
        ind = np.arange(int(start), int(end))
        x, u = [], []
        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            u.append(np.array(U, copy=False))
        
        return np.array(x), np.array(u)

    def get_all_transitions(self):
        # list of transition tuples
        return self.storage

    def add_parallel(self, data):
        for i in range(len(data)):
            self.add(data[i])