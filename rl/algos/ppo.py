"""Proximal Policy Optimization with the clip objective."""
from copy import deepcopy

import torch
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.autograd import Variable

#from .base import PolicyGradientAlgorithm

from rl.envs import Vectorize, Normalize

import numpy as np
import os


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
    
    def calculate_returns(self, next_value, gamma=0.99, tau=0.95):
        self.values[-1] = next_value
        gae = 0
        for step in reversed(range(self.rewards.size(0))):
            delta = self.rewards[step] + gamma * self.values[step + 1] * self.masks[step] - self.values[step]
            gae = delta + gamma * tau * self.masks[step] * gae
            self.returns[step] = gae + self.values[step]


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

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--n_itr", type=int, default=1000,
                            help="number of iterations of the learning algorithm")
        
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
        

    def sample_steps(self, env, policy, num_steps):
        """Collect a set number of frames, as in the original paper."""
        rewards = []
        episode_reward = 0

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        if self.last_state is None:
            state = torch.Tensor(env.reset())
        else:
            state = self.last_state
                    
        rollout = Rollout(num_steps, obs_dim, action_dim, state)

        for step in range(num_steps):
            value, action = policy.act(state)

            state, reward, done, _ = env.step(action.data.numpy())

            episode_reward += reward
            if done:
                state = env.reset()
                rewards.append(episode_reward)
                episode_reward = 0

            reward = torch.Tensor([reward])

            mask = torch.Tensor([0.0 if done else 1.0])

            state = torch.Tensor(state)
            rollout.insert(step, state, action.data, value.data, reward, mask)

        next_value, _ = policy(state)

        rollout.calculate_returns(next_value.data, self.gamma, self.tau)

        self.last_state = rollout.states[-1]
        
        return (rollout.states[:-1], 
               rollout.actions, 
               rollout.returns[:-1], 
               rollout.values[:-1],
               sum(rewards)/(len(rewards)+1))

    def train(self,
              env_fn,
              policy, 
              n_itr,
              normalize=True,
              logger=None):

        env = Vectorize([env_fn]) # this will be useful for parallelism later

        if normalize:
            env = Normalize(env, ret=False)

        old_policy = deepcopy(policy)

        optimizer = optim.Adam(policy.parameters(), lr=self.lr, eps=self.eps)
        for itr in range(n_itr):
            print("********** Iteration %i ************" % itr)
            observations, actions, returns, values, epr = self.sample_steps(env, policy, self.num_steps)
            
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps)

            batch_size = self.batch_size or advantages.numel()
            print("timesteps in batch: %i" % advantages.size()[0])

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

                    obs_batch = observations[indices]
                    action_batch = actions[indices]

                    return_batch = Variable(returns[indices])
                    advantage_batch = Variable(advantages[indices])

                    values, pdf = policy.evaluate(Variable(obs_batch))

                    with torch.no_grad():
                        _, old_pdf = old_policy.evaluate(Variable(obs_batch))
                    
                    log_probs = pdf.log_prob(Variable(action_batch))
                    
                    with torch.no_grad():
                        old_log_probs = old_pdf.log_prob(Variable(action_batch))

                    old_log_probs = Variable(old_log_probs.data)
                    
                    ratio = (log_probs - old_log_probs).exp()

                    cpi_loss = ratio * advantage_batch
                    clip_loss = ratio.clamp(1.0 - self.clip, 1.0 + self.clip) * advantage_batch
                    actor_loss = -torch.min(cpi_loss, clip_loss).mean()

                    critic_loss = (return_batch - values).pow(2).mean()

                    #entropy_penalty = 

                    optimizer.zero_grad()
                    (actor_loss + critic_loss).backward()
                    optimizer.step()

                    losses.append([actor_loss.data.clone().numpy(),
                                   #entropy_penalty.data.numpy()[0],
                                   critic_loss.data.numpy(),
                                   ratio.data.mean()])

                print(' '.join(["%g"%x for x in np.mean(losses, axis=0)]))

            if logger is not None:
                logger.record("Reward", epr)
                logger.dump()

            if itr % 10 == 0:
                save_path = os.path.join("./trained_models", "ppo")
                try:
                    os.makedirs(save_path)
                except OSError:
                    pass

                if normalize:
                    save_model = [policy, env.ob_rms]
                else:
                    save_model = policy
                
                torch.save(save_model, os.path.join("./trained_models", "Walker2d-v1" + ".pt"))

            
