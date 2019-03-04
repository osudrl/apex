"""Proximal Policy Optimization (clip objective)."""
from copy import deepcopy

import torch
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.autograd import Variable
from torch.distributions import kl_divergence

from rl.envs import Vectorize, Normalize

import time

import numpy as np
import os

class PPOBuffer:
    """
    A buffer for storing trajectory data and calculating returns for the policy
    and critic updates.

    This container is intentionally not optimized w.r.t. to memory allocation
    speed because such allocation is almost never a bottleneck for policy 
    gradient. 
    
    On the other hand, experience buffers are a frequent source of
    off-by-one errors and other bugs in policy gradient implementations, so
    this code is optimized for clarity and readability, at the expense of being
    (very) marginally slower than some other implementations. 

    (Premature optimization is the root of all evil).
    """
    def __init__(self, gamma=0.99, lam=0.95, use_gae=False):
        self.states  = []
        self.actions = []
        self.rewards = []
        self.values  = []
        self.returns = []

        self.ep_returns = [] # for logging
        self.ep_lens    = []

        self.gamma, self.lam = gamma, lam

        self.ptr, self.path_idx = 0, 0
    
    def store(self, state, action, reward, value):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        # TODO: make sure these dimensions really make sense
        self.states  += [state.squeeze(0)]
        self.actions += [action.squeeze(0)]
        self.rewards += [reward.squeeze(0)]
        self.values  += [value.squeeze(0)]

        self.ptr += 1
    
    def finish_path(self, last_val=None):
        if last_val is None:
            last_val = np.zeros(shape=(1,))

        path = slice(self.path_idx, self.ptr)
        rewards = self.rewards[path]

        returns = []

        R = last_val.squeeze(0).copy() # Avoid copy?
        for reward in reversed(rewards):
            R = self.gamma * R + reward
            returns.insert(0, R) # TODO: self.returns.insert(self.path_idx, R) ? 
                                 # also technically O(k^2), may be worth just reversing list
                                 # BUG? This is adding copies of R by reference (?)

        self.returns += returns

        self.ep_returns += [np.sum(rewards)]
        self.ep_lens    += [len(rewards)]

        self.path_idx = self.ptr
    
    def get(self):
        return(
            self.states,
            self.actions,
            self.returns,
            self.values
        )

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
                 minibatch_size=None,
                 num_steps=None):

        self.gamma         = args['gamma']
        self.lam           = args['lam']
        self.lr            = args['lr']
        self.eps           = args['eps']
        self.entropy_coeff = args['entropy_coeff']
        self.clip          = args['clip']
        self.minibatch_size    = args['minibatch_size']
        self.epochs        = args['epochs']
        self.num_steps     = args['num_steps']

        self.name = args['name']
        self.use_gae = args['use_gae']
        self.n_proc = args['num_procs']

        self.grad_clip = args['max_grad_norm']

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

        parser.add_argument("--minibatch_size", type=int, default=64,
                            help="Batch size for PPO updates")

        parser.add_argument("--epochs", type=int, default=10,
                            help="Number of optimization epochs per PPO update")

        parser.add_argument("--num_steps", type=int, default=5096,
                            help="Number of sampled timesteps per gradient estimate")

        parser.add_argument("--use_gae", type=bool, default=True,
                            help="Whether or not to calculate returns using Generalized Advantage Estimation")

        parser.add_argument("--num_procs", type=int, default=1,
                            help="Number of threads to train on")

        parser.add_argument("--max_grad_norm", type=float, default=0.5,
                            help="Value to clip gradients at.")

    def save(self, policy, env):
        save_path = os.path.join("./trained_models", "ppo")

        try:
            os.makedirs(save_path)
        except OSError:
            pass

        filetype = ".pt" # pytorch model

        if hasattr(env, 'ob_rms'):
            mean, std = env.ob_rms.mean, np.sqrt(env.ob_rms.var + 1E-8)
            policy.obs_mean = torch.Tensor(mean)
            policy.obs_std = torch.Tensor(std)

        torch.save(policy, os.path.join("./trained_models", self.name + filetype))

    @torch.no_grad()
    def sample(self, env, policy, min_steps, max_traj_len, deterministic=False):
        """
        Sample at least min_steps number of total timesteps, truncating 
        trajectories only if they exceed max_traj_len number of timesteps
        """
        memory = PPOBuffer(self.gamma, self.lam)

        num_steps = 0
        while num_steps < min_steps:
            state = torch.Tensor(env.reset())

            done = False
            value = 0
            traj_len = 0

            while not done and traj_len < max_traj_len:
                value, action = policy.act(state, deterministic)

                next_state, reward, done, _ = env.step(action.numpy())

                memory.store(state.numpy(), action.numpy(), reward, value.numpy())

                state = torch.Tensor(next_state)

                traj_len += 1
                num_steps += 1

            value, _ = policy.act(state)
            memory.finish_path(last_val=(not done) * value.numpy())
        
        return memory

    @torch.no_grad()
    def _sample(self, env_fn, policy, min_steps, max_traj_len, deterministic=False):
        """
        Sample at least min_steps number of total timesteps, truncating 
        trajectories only if they exceed max_traj_len number of timesteps
        """
        env = Vectorize([env_fn])

        memory = PPOBuffer(self.gamma, self.lam)

        num_steps = 0
        while num_steps < min_steps:
            state = torch.Tensor(env.reset())

            done = False
            value = 0
            traj_len = 0

            while not done and traj_len < max_traj_len:
                value, action = policy.act(state, deterministic)

                next_state, reward, done, _ = env.step(action.numpy())

                memory.store(state.numpy(), action.numpy(), reward, value.numpy())

                state = torch.Tensor(next_state)

                traj_len += 1
                num_steps += 1

            value, _ = policy.act(state)
            memory.finish_path(last_val=(not done) * value.numpy())
        
        return memory

    def sample_parallel(self, env_fn, policy, min_steps, max_traj_len, deterministic=False):
        import torch.multiprocessing as mp
        from functools import partial, reduce

        worker = partial(self._sample, env_fn, policy, min_steps, max_traj_len, deterministic)

        with mp.Pool(processes=self.n_proc) as pool:
            # Call pool of workers, don't apply any arguments
            # TODO: this is a weird use of starmap, maybe Process is more suited?
            result = pool.starmap(worker, [() for _ in range(self.n_proc)])

        def merge(buf1, buf2):
            buf2.states  += buf1.states
            buf2.actions += buf1.actions
            buf2.rewards += buf1.rewards
            buf2.values  += buf1.values
            buf2.returns += buf1.returns

            buf2.ep_returns += buf1.ep_returns
            buf2.ep_lens    += buf1.ep_lens

            return buf2

        memory = reduce(merge, result)
        return memory

    def train(self,
              env_fn,
              policy, 
              n_itr,
              normalize=None,
              logger=None):

        policy.train()

        env = Vectorize([env_fn]) # this will be useful for parallelism later
        
        if normalize is not None:
            env = normalize(env)

            mean, std = env.ob_rms.mean, np.sqrt(env.ob_rms.var + 1E-8)
            policy.obs_mean = torch.Tensor(mean)
            policy.obs_std = torch.Tensor(std)
            policy.train(0)

        env = Vectorize([env_fn])

        old_policy = deepcopy(policy)

        optimizer = optim.Adam(policy.parameters(), lr=self.lr, eps=self.eps)

        start_time = time.time()

        for itr in range(n_itr):
            print("********** Iteration {} ************".format(itr))

            if self.n_proc > 1:
                batch = self.sample_parallel(env_fn, policy, self.num_steps, 300)
            else:
                batch = self._sample(env_fn, policy, self.num_steps, 300) #TODO: fix this

            print("time elapsed: {:.2f} s".format(time.time() - start_time))

            observations, actions, returns, values = map(torch.Tensor, batch.get())

            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps)

            minibatch_size = self.minibatch_size or advantages.numel()

            print("timesteps in batch: %i" % advantages.numel())

            old_policy.load_state_dict(policy.state_dict())  # WAY faster than deepcopy

            for _ in range(self.epochs):
                losses = []
                sampler = BatchSampler(
                    SubsetRandomSampler(range(advantages.numel())),
                    minibatch_size,
                    drop_last=True
                )

                for indices in sampler:
                    indices = torch.LongTensor(indices)

                    obs_batch = observations[indices]
                    action_batch = actions[indices]

                    return_batch = returns[indices]
                    advantage_batch = advantages[indices]

                    values, pdf = policy.evaluate(obs_batch)

                    # TODO, move this outside loop?
                    with torch.no_grad():
                        _, old_pdf = old_policy.evaluate(obs_batch)
                        old_log_probs = old_pdf.log_prob(action_batch).sum(-1, keepdim=True)
                    
                    log_probs = pdf.log_prob(action_batch).sum(-1, keepdim=True)
                    
                    ratio = (log_probs - old_log_probs).exp()

                    cpi_loss = ratio * advantage_batch
                    clip_loss = ratio.clamp(1.0 - self.clip, 1.0 + self.clip) * advantage_batch
                    actor_loss = -torch.min(cpi_loss, clip_loss).mean()

                    critic_loss = 0.5 * (return_batch - values).pow(2).mean()

                    entropy_penalty = -self.entropy_coeff * pdf.entropy().mean()

                    # TODO: add ability to optimize critic and actor seperately, with different learning rates

                    optimizer.zero_grad()
                    (actor_loss + critic_loss + entropy_penalty).backward()

                    # Clip the gradient norm to prevent "unlucky" minibatches from 
                    # causing pathalogical updates
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), self.grad_clip)
                    optimizer.step()

                    losses.append([actor_loss.item(),
                                   pdf.entropy().mean().item(),
                                   critic_loss.item(),
                                   ratio.mean().item()])

                # TODO: add verbosity arguments to suppress this
                print(' '.join(["%g"%x for x in np.mean(losses, axis=0)]))

                # Early stopping 
                if kl_divergence(pdf, old_pdf).mean() > 0.02:
                    print("Max kl reached, stopping optimization early.")
                    break

            if logger is not None:
                test = self.sample(env, policy, 800, 400, deterministic=True)
                _, pdf     = policy.evaluate(observations)
                _, old_pdf = old_policy.evaluate(observations)

                entropy = pdf.entropy().mean().item()
                kl = kl_divergence(pdf, old_pdf).mean().item()

                logger.record("Return (test)", np.mean(test.ep_returns))
                logger.record("Return (batch)", np.mean(batch.ep_returns))
                logger.record("Mean Eplen",  np.mean(batch.ep_lens))
        
                logger.record("Mean KL Div", kl)
                logger.record("Mean Entropy", entropy)
                logger.dump()

            # TODO: add option for how often to save model
            if itr % 10 == 0:
                self.save(policy, env)



            
