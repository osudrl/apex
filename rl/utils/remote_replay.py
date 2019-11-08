import random
import numpy as np
import ray

# tensorboard
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from colorama import Fore, Style

# Plot results
from apex import create_logger

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

        self.logger = create_logger(args)

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
        self.logger.add_scalar('Train/Return', episode_reward, actor_timesteps)

    def plot_eval_results(self, step_count, avg_reward, avg_eplen, update_count):
        self.logger.add_scalar("Test/Return", avg_reward, update_count)
        self.logger.add_scalar("Test/Eplen", avg_eplen, update_count)
        self.logger.add_scalar("Misc/Total Timesteps", step_count, update_count)
        self.logger.add_scalar("Misc/Replay Size", len(self.storage), update_count)
        print("Total T: {}\tEval Return: {}\t Eval Eplen: {}".format(step_count, avg_reward, avg_eplen))

    def plot_actor_loss(self, update_count, actor_loss):
        self.logger.add_scalar("Train/pi_loss", actor_loss, update_count)

    def plot_critic_loss(self, update_count, critic_loss, Q1_mean, Q2_mean):
        self.logger.add_scalar("Train/q_loss", critic_loss, update_count)
        self.logger.add_scalar("Train/avg_q1", Q1_mean, update_count)
        self.logger.add_scalar("Train/avg_q2", Q2_mean, update_count)
        # self.logger.add_scalar("Train/critic_Qs_mean", (Q1_mean + Q2_mean) / 2, update_count) # I don't think this is important enough to log

    def plot_policy_hist(self, policy, update_count):
        for name, param in policy.named_parameters():
            self.logger.add_histogram("Model Params/"+name, param.data, update_count)
            # once using distributional critic, plot that distribution

    # # Used to verify that updates are not being bottlenecked (should keep going up straight)
    # def plot_learner_progress(self, update_count, step_count):
    #     self.logger.plot('Step Count', 'Update Count',split_name='train',title_name='Total Updates', x=step_count, y=update_count)

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