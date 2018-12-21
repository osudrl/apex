"""Proximal Policy Optimization with the clip objective."""
from copy import deepcopy

import torch
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.autograd import Variable

from rl.envs import Vectorize, Normalize

import time

import numpy as np
import os

# TODO: Rollout should be called Episode
class Rollout():
    def __init__(self, num_steps, obs_dim, action_dim, first_state):
        self.states = torch.zeros(num_steps + 1, obs_dim)
        self.states[0] = first_state

        self.actions = torch.zeros(num_steps, action_dim)
        self.rewards = torch.zeros(num_steps, 1)
        self.values = torch.zeros(num_steps + 1, 1)
        self.returns = torch.zeros(num_steps + 1, 1)
        self.masks = torch.ones(num_steps + 1, 1)

        self.initialized = True

    def insert(self, step, state, action, value, reward, mask):
        self.states[step + 1] = state # why?
        self.actions[step] = action
        self.values[step] = value
        self.rewards[step] = reward
        self.masks[step] = mask
    
    def calculate_returns(self, next_value, gamma=0.99, tau=0.95, use_gae=True):
        # "masks" just resets the calculation for each trajectory, based on "done"
        # TODO: make this more easily read
        
        if use_gae:
            self.values[-1] = next_value

            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.values[step + 1] * self.masks[step] - self.values[step]
                gae = delta + gamma * tau * self.masks[step] * gae
                self.returns[step] = gae + self.values[step]
            
        else:
            self.returns[-1] = next_value

            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * gamma * self.masks[step + 1] + self.rewards[step]


# class PPOBuffer:
#     """
#     A buffer for storing trajectories experienced by a PPO agent interacting
#     with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
#     for calculating the advantages of state-action pairs.
#     """

#     def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
#         self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
#         self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
#         self.adv_buf = np.zeros(size, dtype=np.float32)
#         self.rew_buf = np.zeros(size, dtype=np.float32)
#         self.ret_buf = np.zeros(size, dtype=np.float32)
#         self.val_buf = np.zeros(size, dtype=np.float32)
#         self.logp_buf = np.zeros(size, dtype=np.float32)
#         self.gamma, self.lam = gamma, lam
#         self.ptr, self.path_start_idx, self.max_size = 0, 0, size

#     def store(self, obs, act, rew, val, logp):
#         """
#         Append one timestep of agent-environment interaction to the buffer.
#         """
#         assert self.ptr < self.max_size     # buffer has to have room so you can store
#         self.obs_buf[self.ptr] = obs
#         self.act_buf[self.ptr] = act
#         self.rew_buf[self.ptr] = rew
#         self.val_buf[self.ptr] = val
#         self.logp_buf[self.ptr] = logp
#         self.ptr += 1

#     def finish_path(self, last_val=0):
#         """
#         Call this at the end of a trajectory, or when one gets cut off
#         by an epoch ending. This looks back in the buffer to where the
#         trajectory started, and uses rewards and value estimates from
#         the whole trajectory to compute advantage estimates with GAE-Lambda,
#         as well as compute the rewards-to-go for each state, to use as
#         the targets for the value function.
#         The "last_val" argument should be 0 if the trajectory ended
#         because the agent reached a terminal state (died), and otherwise
#         should be V(s_T), the value function estimated for the last state.
#         This allows us to bootstrap the reward-to-go calculation to account
#         for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
#         """

#         path_slice = slice(self.path_start_idx, self.ptr)
#         rews = np.append(self.rew_buf[path_slice], last_val)
#         vals = np.append(self.val_buf[path_slice], last_val)
        
#         # the next two lines implement GAE-Lambda advantage calculation
#         deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
#         self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
#         # the next line computes rewards-to-go, to be targets for the value function
#         self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
#         self.path_start_idx = self.ptr

#     def get(self):
#         """
#         Call this at the end of an epoch to get all of the data from
#         the buffer, with advantages appropriately normalized (shifted to have
#         mean zero and std one). Also, resets some pointers in the buffer.
#         """
#         assert self.ptr == self.max_size    # buffer has to be full before you can get
#         self.ptr, self.path_start_idx = 0, 0
#         # the next two lines implement the advantage normalization trick
#         adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
#         self.adv_buf = (self.adv_buf - adv_mean) / adv_std
#         return [self.obs_buf, self.act_buf, self.adv_buf, 
#                 self.ret_buf, self.logp_buf]



class Statistic:
    def __init__(self):
        self.items = []

    def add(self, item):
        self.items += [item]
    
    def get(self):
        stats = {
            "mean": np.mean(self.items),
            "std": np.std(self.items),
            "max": np.max(self.items),
            "min": np.min(self.items),
            "n": len(self.items)
        }

        return stats

class PPO:
    def __init__(self, 
                 args=None,
                 gamma=None, 
                 tau=None, 
                 lr=None, 
                 eps=None,
                 entropy_coeff=None,
                 clip=None,
                 epochs=None,
                 batch_size=None,
                 num_steps=None):

        self.last_state = None
 
        self.gamma         = gamma         or args.gamma
        self.tau           = tau           or args.tau
        self.lr            = lr            or args.lr
        self.eps           = eps           or args.eps
        self.entropy_coeff = entropy_coeff or args.entropy_coeff
        self.clip          = clip          or args.clip
        self.batch_size    = batch_size    or args.batch_size
        self.epochs        = epochs        or args.epochs
        self.num_steps     = num_steps     or args.num_steps

        self.name = args.name
        self.use_gae = args.use_gae

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--n_itr", type=int, default=1000,
                            help="Number of iterations of the learning algorithm")
        
        parser.add_argument("--lr", type=float, default=3e-4,
                            help="Adam learning rate")

        parser.add_argument("--eps", type=float, default=1e-5,
                            help="Adam epsilon (for numerical stability)")
        
        parser.add_argument("--tau", type=float, default=0.95,
                            help="Generalized advantage estimate discount")

        parser.add_argument("--gamma", type=float, default=0.99,
                            help="MDP discount")
        
        parser.add_argument("--entropy_coeff", type=float, default=0.0,
                            help="Coefficient for entropy regularization")

        parser.add_argument("--clip", type=float, default=0.2,
                            help="Clipping parameter for PPO surrogate loss")

        parser.add_argument("--batch_size", type=int, default=64,
                            help="Batch size for PPO updates")

        parser.add_argument("--epochs", type=int, default=10,
                            help="Number of optimization epochs per PPO update")

        parser.add_argument("--num_steps", type=int, default=5096,
                            help="Number of sampled timesteps per gradient estimate")

        parser.add_argument("--use_gae", type=bool, default=True,
                            help="Whether or not to calculate returns using Generalized Advantage Estimation")
        
    @torch.no_grad()
    def sample_steps(self, env, policy, num_steps, deterministic=False):
        """Collect a set number of frames, as in the original paper."""        
        #if self.last_state is None:
        state = torch.Tensor(env.reset())
        #else:
        #    # BUG: without unsqueeze this drops a dimension due to indexing
        #    state = self.last_state.unsqueeze(0)

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        rollout = Rollout(num_steps, obs_dim, action_dim, state)

        reward_stats = Statistic()

        done = False
        episode_reward = 0
        for step in range(num_steps):
            value, action = policy.act(state, deterministic)

            state, reward, done, _ = env.step(action.data.numpy())

            episode_reward += reward

            if done:
                state = env.reset()
    
                reward_stats.add(episode_reward)
                episode_reward = 0

            reward = torch.Tensor([reward])

            mask = torch.Tensor([0.0 if done else 1.0])

            state = torch.Tensor(state)
            rollout.insert(step, state, action.data, value.data, reward, mask)

        if not done:
            reward_stats.add(episode_reward)

        next_value, _ = policy(state)

        rollout.calculate_returns(next_value.data, self.gamma, self.tau, self.use_gae)

        #self.last_state = rollout.states[-1]
        
        return (rollout.states[:-1], 
               rollout.actions, 
               rollout.returns[:-1], 
               rollout.values[:-1],
               reward_stats.get())
               # TODO: remove that +1^. Divide by 0 only happens when last trajectory gets discarded
               # because it never flags done

    def train(self,
              env_fn,
              policy, 
              n_itr,
              normalized=None,
              logger=None):

        env = Vectorize([env_fn]) # this will be useful for parallelism later
        
        if normalized is not None:
            env = normalized(env)

        old_policy = deepcopy(policy)

        optimizer = optim.Adam(policy.parameters(), lr=self.lr, eps=self.eps)

        for itr in range(n_itr):
            print("********** Iteration %i ************" % itr)
            observations, actions, returns, values, train_stats = self.sample_steps(env, policy, self.num_steps)
            
            advantages = returns - values

            # TODO: make advantage centering an option
            #advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps)

            batch_size = self.batch_size or advantages.numel()

            print("timesteps in batch: %i" % advantages.size(0))

            old_policy.load_state_dict(policy.state_dict())  # WAY faster than deepcopy

            for _ in range(self.epochs):
                losses = []
                sampler = BatchSampler(
                    SubsetRandomSampler(range(self.num_steps)),
                    batch_size,
                    drop_last=False
                )

                for indices in sampler:
                    indices = torch.LongTensor(indices)

                    # TEST: sample WITH replacement
                    # indices = torch.randperm(observations.size(0))[:self.batch_size]

                    obs_batch = observations[indices]
                    action_batch = actions[indices]

                    return_batch = returns[indices]
                    advantage_batch = advantages[indices]

                    values, pdf = policy.evaluate(obs_batch)

                    with torch.no_grad():
                        _, old_pdf = old_policy.evaluate(obs_batch)
                        old_log_probs = old_pdf.log_prob(action_batch) # BUG: log prob should be summed along action axis, aka joint log probs 
                    
                    log_probs = pdf.log_prob(action_batch)
                    
                    ratio = (log_probs - old_log_probs).exp()

                    cpi_loss = ratio * advantage_batch
                    clip_loss = ratio.clamp(1.0 - self.clip, 1.0 + self.clip) * advantage_batch
                    actor_loss = -torch.min(cpi_loss, clip_loss).mean()

                    critic_loss = 0.5 * (return_batch - values).pow(2).mean()

                    entropy_penalty = self.entropy_coeff * pdf.entropy().mean()

                    # https://www.princeton.edu/~yc5/ele538_optimization/
                    # TODO: add ability to optimize critic and actor seperately, with different learning rates

                    optimizer.zero_grad()
                    (actor_loss + critic_loss + entropy_penalty).backward(retain_graph=True)
                    optimizer.step()

                    losses.append([actor_loss.data.clone().numpy(),
                                   pdf.entropy().mean().data.numpy(),
                                   critic_loss.data.numpy(),
                                   ratio.data.mean()])
                # TODO: add verbosity arguments to suppress this
                print(' '.join(["%g"%x for x in np.mean(losses, axis=0)]))

            # TODO: add filtering options for reward graph (e.g., none)
            # add explained variance, high and low rewards, std dev of rewards
            # add options for reward graphs: e.g. explore/no explore

            # total things that should be graphed/logged:
            # number of trajectories
            # average trajectory length (and min/max/std?)
            # reward (return, reward, average,std,max,min, explore/exploit, filtered?)
            # entropy (policy)
            # perplexity
            # explained variance
            # policy average std
            # mean and max KL
            # Loss
            # add "suppress name" option

            # look into making logging a decorator or wrapper?
            _, _, _, _, test_stats = self.sample_steps(env, policy, 800, deterministic=True)
            if logger is not None:
                logger.record("Reward test", test_stats["mean"])
                logger.record("Reward mean", train_stats["mean"])
                logger.record("Reward std", train_stats["std"])
                logger.record("Reward max", train_stats["max"])
                logger.record("Reward min", train_stats["min"])
                logger.dump()

            if itr % 10 == 0:
                # TODO: add option for how often to save model
                save_path = os.path.join("./trained_models", "ppo")
                try:
                    os.makedirs(save_path)
                except OSError:
                    pass

                filetype = ".pt" # pytorch model

                if normalized is not None:
                    # ret_rms is not necessary to run policy, but is necessary to interpret rewards
                    save_model = [policy, (env.ob_rms, env.ret_rms)]
                    
                    filetype = ".ptn" # "normalized" pytorch model
                else:
                    save_model = policy
                
                torch.save(save_model, os.path.join("./trained_models", self.name + filetype))

            
