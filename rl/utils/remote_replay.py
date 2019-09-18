import random
import numpy as np
import ray

# tensorboard
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from colorama import Fore, Style

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

        self.using_tensorboard = True

        #self.using_tensorboard = True if args.logger_name is "tensorboard" else False
        if self.using_tensorboard:
            now = datetime.now()
            # NOTE: separate by trial name first and time of run after
            log_path = args.logdir + now.strftime("%Y%m%d-%H%M%S")+"/"
            self.logger = SummaryWriter(log_path, flush_secs=0.1)
            print(Fore.GREEN + Style.BRIGHT + "Logging data using TensorBoard to {}".format(log_path + Style.RESET_ALL))
        else:
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
        if not self.using_tensorboard:
            self.logger.plot('Return', 'Actor Timesteps',split_name='actor {}'.format(actor_id),title_name='Actor Episode Return', x=actor_timesteps, y=episode_reward)
        else:
            #self.logger.add_scalar('Data/Return Batch', episode_reward, actor_timesteps)
            pass

    def plot_eval_results(self, step_count, avg_reward, avg_eplen, update_count):
        if not self.using_tensorboard:
            self.logger.record('Update Count', update_count, step_count, title_name='Total Updates', x_var_name='Global Timesteps', split_name='eval')
            self.logger.record('Eval Return', avg_reward, step_count, title_name='Eval Return', x_var_name='Global Timesteps', split_name='eval')
            self.logger.record('Eval Eplen', avg_eplen, step_count, title_name='Eval Eplen', x_var_name='Global Timesteps', split_name='eval')
            self.logger.record('Replay Size', len(self.storage), step_count, title_name='Replay Size', x_var_name='Global Timesteps', split_name='eval')
            self.logger.dump()
        else:
            self.logger.add_scalar("Data/Return Test", avg_reward, update_count)
            self.logger.add_scalar("Data/Mean Eplen", avg_eplen, update_count)
            self.logger.add_scalar("Misc/Total Timesteps", step_count, update_count)
            self.logger.add_scalar("Misc/Replay Size", len(self.storage), update_count)

    def plot_actor_loss(self, update_count, actor_loss):
        if not self.using_tensorboard:
            self.logger.plot('Actor Loss', 'Update Count',split_name='train',title_name='Actor Network Loss', x=update_count, y=actor_loss)
        else:
            self.logger.add_scalar("Losses/actor_network_loss", actor_loss, update_count)

    def plot_critic_loss(self, update_count, critic_loss):
        if not self.using_tensorboard:
            self.logger.plot('Critic Loss', 'Update Count',split_name='train',title_name='Critic Network Loss', x=update_count, y=critic_loss)
        else:
            self.logger.add_scalar("Losses/critic_network_loss", critic_loss, update_count)

    def plot_policy_hist(self, policy, update_count):
        if self.using_tensorboard:
            for name, param in policy.named_parameters():
                self.logger.add_histogram("Model Params/"+name, param.data, update_count)
            # once using distributional critic, plot that distribution

    # # Used to verify that updates are not being bottlenecked (should keep going up straight)
    # def plot_learner_progress(self, update_count, step_count):
    #     self.logger.plot('Step Count', 'Update Count',split_name='train',title_name='Total Updates', x=step_count, y=update_count)