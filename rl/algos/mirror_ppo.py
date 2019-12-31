import time
from copy import deepcopy
import torch
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import kl_divergence

import numpy as np
from rl.algos import PPO
from rl.policies.actor import Gaussian_FF_Actor, Gaussian_LSTM_Actor
from rl.policies.critic import FF_V, LSTM_V
from rl.envs.normalize import get_normalization_params, PreNormalizer

from rl.envs.wrappers import SymmetricEnv

import functools

# TODO:
# env.mirror() vs env.matrix?

# TODO: use magic to make this reuse more code (callbacks etc?)

class MirrorPPO(PPO):
    def update(self, policy, old_policy, optimizer,
               critic, critic_optimizer,
               observations, actions, returns, advantages,
               env_fn
    ):
        env = env_fn()
        mirror_observation = env.mirror_observation
        if env.clock_based:
            mirror_observation = env.mirror_clock_observation
        mirror_action = env.mirror_action

        minibatch_size = self.minibatch_size or advantages.numel()

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
                    # obs_batch = torch.cat(
                    #     [obs_batch,
                    #      obs_batch @ torch.Tensor(env.obs_symmetry_matrix)]
                    # ).detach()

                    action_batch = actions[indices]
                    # action_batch = torch.cat(
                    #     [action_batch,
                    #      action_batch @ torch.Tensor(env.action_symmetry_matrix)]
                    # ).detach()

                    return_batch = returns[indices]
                    # return_batch = torch.cat(
                    #     [return_batch,
                    #      return_batch]
                    # ).detach()

                    advantage_batch = advantages[indices]
                    # advantage_batch = torch.cat(
                    #     [advantage_batch,
                    #      advantage_batch]
                    # ).detach()

                    values = critic.act(obs_batch)
                    pdf = policy.evaluate(obs_batch)

                    # TODO, move this outside loop?
                    with torch.no_grad():
                        old_pdf = old_policy.evaluate(obs_batch)
                        old_log_probs = old_pdf.log_prob(action_batch).sum(-1, keepdim=True)
                    
                    log_probs = pdf.log_prob(action_batch).sum(-1, keepdim=True)
                    
                    ratio = (log_probs - old_log_probs).exp()

                    cpi_loss = ratio * advantage_batch
                    clip_loss = ratio.clamp(1.0 - self.clip, 1.0 + self.clip) * advantage_batch
                    actor_loss = -torch.min(cpi_loss, clip_loss).mean()

                    critic_loss = 0.5 * (return_batch - values).pow(2).mean()

                    # Mirror Symmetry Loss
                    deterministic_actions = policy(obs_batch)
                    if env.clock_based:
                        mir_obs = mirror_observation(obs_batch, env.clock_inds)
                        mirror_actions = policy(mir_obs)
                    else: 
                        mirror_actions = policy(mirror_observation(obs_batch))
                    mirror_actions = mirror_action(mirror_actions)

                    mirror_loss = 4 * (deterministic_actions - mirror_actions).pow(2).mean()

                    entropy_penalty = -self.entropy_coeff * pdf.entropy().mean()

                    optimizer.zero_grad()
                    (actor_loss + mirror_loss + entropy_penalty).backward()

                    # Clip the gradient norm to prevent "unlucky" minibatches from 
                    # causing pathalogical updates
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), self.grad_clip)
                    optimizer.step()

                    critic_optimizer.zero_grad()
                    critic_loss.backward()

                    # Clip the gradient norm to prevent "unlucky" minibatches from 
                    # causing pathalogical updates
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), self.grad_clip)
                    critic_optimizer.step()

                    losses.append([actor_loss.item(),
                                   pdf.entropy().mean().item(),
                                   critic_loss.item(),
                                   ratio.mean().item(),
                                   mirror_loss.item()])

                # TODO: add verbosity arguments to suppress this
                print(' '.join(["%g"%x for x in np.mean(losses, axis=0)]))

                # Early stopping 
                if kl_divergence(pdf, old_pdf).mean() > 0.02:
                    print("Max kl reached, stopping optimization early.")
                    break

    def train(self,
              env_fn,
              policy,
              policy_copy,
              critic,
              n_itr,
              logger=None):

        # old_policy = deepcopy(policy)
        old_policy = policy_copy

        optimizer = optim.Adam(policy.parameters(), lr=self.lr, eps=self.eps)
        critic_optimizer = optim.Adam(critic.parameters(), lr=self.lr, eps=self.eps)

        start_time = time.time()

        for itr in range(n_itr):
            print("********** Iteration {} ************".format(itr))

            sample_start = time.time()
            batch = self.sample_parallel(env_fn, policy, critic, self.num_steps, self.max_traj_len)

            print("time elapsed: {:.2f} s".format(time.time() - start_time))
            print("sample time elapsed: {:.2f} s".format(time.time() - sample_start))

            observations, actions, returns, values = map(torch.Tensor, batch.get())

            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps)

            minibatch_size = self.minibatch_size or advantages.numel()

            print("timesteps in batch: %i" % advantages.numel())
            self.total_steps += advantages.numel()

            old_policy.load_state_dict(policy.state_dict())  # WAY faster than deepcopy

            optimizer_start = time.time()

            self.update(policy, old_policy, optimizer, critic, critic_optimizer, observations, actions, returns, advantages, env_fn) 
           
            print("optimizer time elapsed: {:.2f} s".format(time.time() - optimizer_start))        


            if logger is not None:
                evaluate_start = time.time()
                test = self.sample_parallel(env_fn, policy, critic, 800 // self.n_proc, self.max_traj_len, deterministic=True)
                print("evaluate time elapsed: {:.2f} s".format(time.time() - evaluate_start))

                avg_eval_reward = np.mean(test.ep_returns)

                pdf     = policy.evaluate(observations)
                old_pdf = old_policy.evaluate(observations)

                entropy = pdf.entropy().mean().item()
                kl = kl_divergence(pdf, old_pdf).mean().item()

                logger.add_scalar("Test/Return", avg_eval_reward, itr)
                logger.add_scalar("Train/Return", np.mean(batch.ep_returns), itr)
                logger.add_scalar("Train/Mean Eplen", np.mean(batch.ep_lens), itr)
                logger.add_scalar("Train/Mean KL Div", kl, itr)
                logger.add_scalar("Train/Mean Entropy", entropy, itr)
                logger.add_scalar("Misc/Timesteps", self.total_steps, itr)

            # TODO: add option for how often to save model
            if np.mean(test.ep_returns) > self.max_return:
                self.max_return = np.mean(test.ep_returns)
                self.save(policy)

            print("Total time: {:.2f} s".format(time.time() - start_time))

def run_experiment(args):
    torch.set_num_threads(1) # see: https://github.com/pytorch/pytorch/issues/13757

    from apex import env_factory, create_logger

    # wrapper function for creating parallelized envs
    env_fn = env_factory(args.env_name, state_est=args.state_est, mirror=args.mirror, speed=args.speed)
    obs_dim = env_fn().observation_space.shape[0]
    action_dim = env_fn().action_space.shape[0]

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.previous is not None:
        policy = torch.load(args.previous)
        print("loaded model from {}".format(args.previous))
    else:
        if args.recurrent:
          policy = Gaussian_LSTM_Actor(obs_dim,
                                       action_dim,
                                       fixed_std=np.exp(-2),
                                       env_name=args.env_name
          )
          policy_copy = Gaussian_LSTM_Actor(obs_dim,
                                            action_dim,
                                            fixed_std=np.exp(-2),
                                            env_name=args.env_name
          )
          critic = LSTM_V(obs_dim)
        else:
          policy = Gaussian_FF_Actor(
              obs_dim, action_dim,
              env_name=args.env_name,
              nonlinearity=torch.nn.functional.relu, 
              bounded=True, 
              init_std=np.exp(-2), 
              learn_std=False,
              normc_init=False
          )
          critic = FF_V(
              obs_dim, 
              env_name=args.env_name,
              nonlinearity=torch.nn.functional.relu,
              normc_init=False
          )

        policy.obs_mean, policy.obs_std = map(torch.Tensor, get_normalization_params(iter=args.input_norm_steps, noise_std=1, policy=policy, env_fn=env_fn))
        critic.obs_mean = policy.obs_mean
        policy_copy.obs_mean = policy.obs_mean
        critic.obs_std = policy.obs_std
        policy_copy.obs_std = policy.obs_std

    print("obs_dim: {}, action_dim: {}".format(obs_dim, action_dim))

    #if args.mirror:
    #    algo = MirrorPPO(args=vars(args))
    #else:
    #    algo = PPO(args=vars(args))
    algo = PPO(args=vars(args))

    # create a tensorboard logging object
    logger = create_logger(args)

    print()
    print("Synchronous Distributed Proximal Policy Optimization:")
    print("\tenv:            {}".format(args.env_name))
    print("\tmax traj len:   {}".format(args.max_traj_len))
    print("\tseed:           {}".format(args.seed))
    print("\tmirror:         {}".format(args.mirror))
    print("\tnum procs:      {}".format(args.num_procs))
    print("\tlr:             {}".format(args.lr))
    print("\teps:            {}".format(args.eps))
    print("\tlam:            {}".format(args.lam))
    print("\tgamma:          {}".format(args.gamma))
    print("\tentropy coeff:  {}".format(args.entropy_coeff))
    print("\tclip:           {}".format(args.clip))
    print("\tminibatch size: {}".format(args.minibatch_size))
    print("\tepochs:         {}".format(args.epochs))
    print("\tnum steps:      {}".format(args.num_steps))
    print("\tuse gae:        {}".format(args.use_gae))
    print("\tmax grad norm:  {}".format(args.max_grad_norm))
    print("\tmax traj len:   {}".format(args.max_traj_len))
    print()

    algo.train(env_fn, policy, policy_copy, critic, args.n_itr, logger=logger)
