import random
import numpy as np
import ray

# Plot results
from rl.utils import Logger

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
        self.storage = []
        self.max_size = size
        self.ptr = 0

        self.plot_storage = []
        self.logger = Logger(args, env_name=experiment_name, viz=True)

        print("Created replay buffer with size {}".format(self.max_size))
    
    def __len__(self):
        return len(self.storage)

    def storage_size(self):
        return len(self.storage)

    def add(self, data):
        if len(self.storage) < self.max_size:
            self.storage.append(data)
        self.storage[int(self.ptr)] = data
        self.ptr = (self.ptr + 1) % self.max_size
        #print("Added experience to replay buffer.")

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

        #print("Sampled experience from replay buffer.")
        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)

    def get_transitions_from_range(self, start):
        end = self.ptr
        ind = np.arange(int(start), int(end))
        x, u = [], []
        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            u.append(np.array(U, copy=False))
        
        return np.array(x), np.array(u)

    def plot_actor_results(self, actor_id, actor_timesteps, episode_reward):
        self.logger.plot('return', 'Actor timesteps','actor {}'.format(actor_id), 'Actor Episode Return', actor_timesteps, episode_reward)

    def plot_learner_results(self, step_count, avg_reward):
        self.logger.record('Agent Return', avg_reward, step_count, 'Agent Return', x_var_name='Global Timesteps', split_name='eval')
        self.logger.dump()