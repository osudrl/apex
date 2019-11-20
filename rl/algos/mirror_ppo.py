import time
from copy import deepcopy
import torch
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import kl_divergence

import numpy as np
from rl.algos import PPO
from rl.policies.actor import GaussianMLP_Actor
from rl.policies.critic import GaussianMLP_Critic
from rl.envs.normalize import get_normalization_params, PreNormalizer

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

    # # Environment
    # if(args.env in ["Cassie-v0", "Cassie-mimic-v0", "Cassie-mimic-walking-v0"]):
    #     # NOTE: importing cassie for some reason breaks openai gym, BUG ?
    #     from cassie import CassieEnv, CassieTSEnv, CassieIKEnv
    #     from cassie.no_delta_env import CassieEnv_nodelta
    #     from cassie.speed_env import CassieEnv_speed
    #     from cassie.speed_double_freq_env import CassieEnv_speed_dfreq
    #     from cassie.speed_no_delta_env import CassieEnv_speed_no_delta
    #     # set up cassie environment
    #     # import gym_cassie
    #     # env_fn = gym_factory(args.env_name)
    #     #env_fn = make_env_fn(state_est=args.state_est)
    #     #env_fn = functools.partial(CassieEnv_speed_dfreq, "walking", clock_based = True, state_est=args.state_est)
    #     env_fn = functools.partial(CassieIKEnv, clock_based=True, state_est=args.state_est)
    #     print(env_fn().clock_inds)
    #     obs_dim = env_fn().observation_space.shape[0]
    #     action_dim = env_fn().action_space.shape[0]

    #     # Mirror Loss
    #     if args.mirror:
    #         if args.state_est:
    #             # with state estimator
    #             env_fn = functools.partial(SymmetricEnv, env_fn, mirrored_obs=[0.1, 1, 2, 3, 4, -10, -11, 12, 13, 14, -5, -6, 7, 8, 9, 15, 16, 17, 18, 19, 20, -26, -27, 28, 29, 30, -21, -22, 23, 24, 25, 31, 32, 33, 37, 38, 39, 34, 35, 36, 43, 44, 45, 40, 41, 42, 46, 47, 48], mirrored_act=[-5, -6, 7, 8, 9, -0.1, -1, 2, 3, 4])
    #         else:
    #             # without state estimator
    #             env_fn = functools.partial(SymmetricEnv, env_fn, mirrored_obs=[0.1, 1, 2, 3, 4, 5, -13, -14, 15, 16, 17,
    #                                             18, 19, -6, -7, 8, 9, 10, 11, 12, 20, 21, 22, 23, 24, 25, -33,
    #                                             -34, 35, 36, 37, 38, 39, -26, -27, 28, 29, 30, 31, 32, 40, 41, 42],
    #                                             mirrored_act = [-5, -6, 7, 8, 9, -0.1, -1, 2, 3, 4])
    # else:
    #     import gym
    #     env_fn = gym_factory(args.env_name)
    #     #max_episode_steps = env_fn()._max_episode_steps
    #     obs_dim = env_fn().observation_space.shape[0]
    #     action_dim = env_fn().action_space.shape[0]
    #     max_episode_steps = 1000

    # wrapper function for creating parallelized envs
    env_fn = env_factory(args.env_name, state_est=args.state_est, mirror=args.mirror)
    obs_dim = env_fn().observation_space.shape[0]
    action_dim = env_fn().action_space.shape[0]

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.previous is not None:
        policy = torch.load(args.previous)
        print("loaded model from {}".format(args.previous))
    else:
        policy = GaussianMLP_Actor(
            obs_dim, action_dim,
            env_name=args.env_name,
            nonlinearity=torch.nn.functional.relu, 
            bounded=True, 
            init_std=np.exp(-2), 
            learn_std=False,
            normc_init=False
        )
        policy_copy = GaussianMLP_Actor(
            obs_dim, action_dim, 
            env_name=args.env_name,
            nonlinearity=torch.nn.functional.relu, 
            bounded=True, 
            init_std=np.exp(-2), 
            learn_std=False,
            normc_init=False
        )
        critic = GaussianMLP_Critic(
            obs_dim, 
            env_name=args.env_name,
            nonlinearity=torch.nn.functional.relu, 
            bounded=True, 
            init_std=np.exp(-2), 
            learn_std=False,
            normc_init=False
        )

        policy.obs_mean, policy.obs_std = map(torch.Tensor, get_normalization_params(iter=args.input_norm_steps, noise_std=1, policy=policy, env_fn=env_fn))
        critic.obs_mean = policy.obs_mean
        policy_copy.obs_mean = policy.obs_mean
        critic.obs_std = policy.obs_std
        policy_copy.obs_std = policy.obs_std

    policy.train(0)
    policy_copy.train(0)
    critic.train(0)

    print("obs_dim: {}, action_dim: {}".format(obs_dim, action_dim))

    if args.mirror:
        algo = MirrorPPO(args=vars(args))
    else:
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