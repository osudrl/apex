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
    
    def calculate_returns(self, next_value, gamma=0.99, lam=0.95, use_gae=True):
        # "masks" just resets the calculation for each trajectory, based on "done"
        # TODO: make this more easily read
        
        if use_gae:
            self.values[-1] = next_value

            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.values[step + 1] * self.masks[step] - self.values[step]
                gae = delta + gamma * lam * self.masks[step] * gae
                self.returns[step] = gae + self.values[step]
            
        else:
            self.returns[-1] = next_value

            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * gamma * self.masks[step + 1] + self.rewards[step]

    def get(self):
        return (self.states[:-1], 
               self.actions, 
               self.returns[:-1], 
               self.values[:-1])


class PPOBuffer:
    # Buffer philosophy: the buffer is almost never a bottleneck, so don't
    # worry about preallocating memory or doing cool tricks, but it's a strong
    # source of implementation bugs, so keep it readable
    # Verbose variable names are always better
    def __init__(self, gamma=0.99, lam=0.95, use_gae=False):
        self.observations = []
        self.actions      = []
        self.rewards      = []
        self.values       = []
        self.returns      = []

        self.gamma, self.lam = gamma, lam

        self.ptr, self.path_idx = 0, 0
    
    def store(self, observation, action, reward, value):
        self.observations += [observation.squeeze(0)]
        self.actions      += [action.squeeze(0)]
        self.rewards      += [reward]
        self.values       += [value.squeeze(0)]

        self.ptr += 1
    
    def finish_path(self, last_val=None):
        if last_val is None:
            last_val = np.zeros(shape=(1,))

        path = slice(self.path_idx, self.ptr)
        rewards = self.rewards[path]
        returns = []

        R = last_val.squeeze(0)
        for reward in reversed(rewards):
            R = self.gamma * R + reward
            returns.insert(0, R) # TODO: self.returns.insert(self.path_idx, R) ? 
                                 # also technically O(n^2), may be worth just reversing list
                                 # BUG? This is adding copies of R by reference

        self.returns += returns

        self.path_idx = self.ptr
    
    def get(self):
        return(
            self.observations,
            self.actions,
            self.returns,
            self.values
        )


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
                 lam=None, 
                 lr=None, 
                 eps=None,
                 entropy_coeff=None,
                 clip=None,
                 epochs=None,
                 batch_size=None,
                 num_steps=None):

        self.gamma         = args['gamma']
        self.lam           = args['lam']
        self.lr            = args['lr']
        self.eps           = args['eps']
        self.entropy_coeff = args['entropy_coeff']
        self.clip          = args['clip']
        self.batch_size    = args['batch_size']
        self.epochs        = args['epochs']
        self.num_steps     = args['num_steps']

        self.name = args['name']
        self.use_gae = args['use_gae']


    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--n_itr", type=int, default=10000,
                            help="Number of iterations of the learning algorithm")
        
        parser.add_argument("--lr", type=float, default=3e-4,
                            help="Adam learning rate")

        parser.add_argument("--eps", type=float, default=1e-5,
                            help="Adam epsilon (for numerical stability)")
        
        parser.add_argument("--lam", type=float, default=0.95,
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

    #TODO: make this better
    @torch.no_grad()
    def sample(self, env, policy, min_steps, max_traj_len):
        # SAMPLING PHILOSOPHY:
        # Never truncate trajectories other than because max_traj_len is exceeded
        # I.e. it's better to have a number of steps that's not divisible by the batch size,
        # than to have an arbitrarily small trajectory screwing state-return estimates
        # ALWAYS bootstrap the return estimate of a truncated trajectory using the critic value estimate
        # Fure possibilities: sample independent trajectories NOT steps, as that's what really reduces gradient variance
        # The above may lead to pessimistic performance and slowdowns, so maybe annealing the trajectory sample size
        # is the solution?

        memory = PPOBuffer(self.gamma, self.lam)

        num_steps = 0
        while num_steps < min_steps:
            state = torch.Tensor(env.reset())

            done = False
            value = 0
            traj_len = 0

            while traj_len < max_traj_len:
                value, action = policy.act(state, deterministic=False) # TODO: add determinism?

                state, reward, done, _ = env.step(action.data.numpy())

                memory.store(state, action.data.numpy(), reward, value.data.numpy())

                traj_len += 1
                num_steps += 1

                state = torch.Tensor(state)

                # Need next_state to account for states being offset by one
                # More elegant way to do this?

                if done:
                    break

            memory.finish_path(last_val=(not done) * value.data.numpy())
        
        return memory

        
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

        memory = PPOBuffer(self.gamma)

        reward_stats = Statistic()

        done = False
        episode_reward = 0
        for step in range(num_steps):
            value, action = policy.act(state, deterministic)

            state, reward, done, _ = env.step(action.data.numpy())

            episode_reward += reward

            reward = torch.Tensor([reward])

            mask = torch.Tensor([0.0 if done else 1.0])

            state = torch.Tensor(state)

            rollout.insert(step, state, action.data, value.data, reward, mask)

            if done:
                state = torch.Tensor(env.reset())
    
                reward_stats.add(episode_reward)
                episode_reward = 0

                #memory.finish_path()
            
            #memory.store(state, action.data.numpy(), reward, value.data.numpy())

            # reward = torch.Tensor([reward])

            # mask = torch.Tensor([0.0 if done else 1.0])

            # state = torch.Tensor(state)
            #rollout.insert(step, state, action.data, value.data, reward, mask)

        next_value, _ = policy(state)

        if not done:
            print("truncated traj")
            reward_stats.add(episode_reward)
            memory.finish_path(next_value.data.numpy())

        rollout.calculate_returns(next_value.data, self.gamma, self.lam, self.use_gae)

        #self.last_state = rollout.states[-1]

        # try:
        #     assert np.allclose(rollout.returns[:-1].data.numpy(), memory.returns)
        #     assert np.allclose(rollout.values[:-1].data.numpy(), memory.values)
        #     assert np.allclose(rollout.actions.data.numpy(), memory.actions)
        #     assert np.allclose(rollout.rewards.data.numpy(), memory.rewards)
        #     #assert np.allclose(rollout.states[:-1].data.numpy(), memory.observations)

        # except (ValueError, AssertionError) as e:
        #     print("Error:", e)
        #     print(len(memory.rewards))
        #     print(len(memory.returns))
        #     print(len(memory.values))
        #     np.set_printoptions(threshold=10000)
        #     import pdb
        #     pdb.set_trace()

        return rollout, reward_stats.get()


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

        thresh = 10 #################

        for itr in range(n_itr):
            print("********** Iteration %i ************" % itr)

            optimizer = optim.Adam(policy.parameters(), lr=self.lr, eps=self.eps)

            #observations, actions, returns, values, train_stats = self.sample_steps(env, policy, self.num_steps)

            rollout, _ = self.sample_steps(env, policy, self.num_steps)

            observations, actions, returns, values = map(torch.Tensor, rollout.get())

            # batch = self.sample(env, policy, self.num_steps, 400) #TODO: fix this

            # observations2, actions2, returns2, values2 = map(torch.Tensor, batch.get())

            # if itr % thresh == 0:
            #     import pdb
            #     pdb.set_trace()

            #print(observations.size(), actions.size(), returns.size(), values.size())

            torch.set_printoptions(threshold=50000)

            #print(returns)
            #exit()

            #print(values)

            advantages = returns - values

            # print(values)

            # exit()

            # TODO: make advantage centering an option
            #advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps)

            # batch size should be minibatch

            batch_size = self.batch_size or advantages.numel()

            print("timesteps in batch: %i" % advantages.numel())

            old_policy.load_state_dict(policy.state_dict())  # WAY faster than deepcopy

            for _ in range(self.epochs):
                losses = []
                sampler = BatchSampler(
                    SubsetRandomSampler(range(advantages.numel())),
                    batch_size,
                    drop_last=True
                )

                for indices in sampler:
                    indices = torch.LongTensor(indices)

                    obs_batch = observations[indices]
                    action_batch = actions[indices]

                    return_batch = returns[indices]
                    advantage_batch = advantages[indices]

                    values, pdf = policy.evaluate(obs_batch)

                    with torch.no_grad():
                        _, old_pdf = old_policy.evaluate(obs_batch)
                        old_log_probs = old_pdf.log_prob(action_batch).sum(-1, keepdim=True)
                    
                    log_probs = pdf.log_prob(action_batch).sum(-1, keepdim=True)
                    
                    ratio = (log_probs - old_log_probs).exp()

                    cpi_loss = ratio * advantage_batch
                    clip_loss = ratio.clamp(1.0 - self.clip, 1.0 + self.clip) * advantage_batch
                    actor_loss = -torch.min(cpi_loss, clip_loss).mean()

                    critic_loss = 0.5 * (return_batch - values).pow(2).mean()

                    entropy_penalty = self.entropy_coeff * pdf.entropy().mean()

                    # https://www.princeton.edu/~yc5/ele538_optimization/
                    # TODO: add ability to optimize critic and actor seperately, with different learning rates

                    optimizer.zero_grad()
                    (actor_loss + critic_loss + entropy_penalty).backward()
                    optimizer.step()

                    if actor_loss.item() > thresh:
                        import pdb
                        pdb.set_trace()

                    losses.append([actor_loss.item(),
                                   pdf.entropy().mean().item(),
                                   critic_loss.item(),
                                   ratio.mean().item()])

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
            _, test_stats = self.sample_steps(env, policy, 800, deterministic=True)
            if logger is not None:
                logger.record("Reward test", test_stats["mean"])
                #logger.record("Reward mean", train_stats["mean"])
                #logger.record("Reward std", train_stats["std"])
                #logger.record("Reward max", train_stats["max"])
                #logger.record("Reward min", train_stats["min"])
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

            
