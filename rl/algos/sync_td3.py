import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from rl.utils.remote_replay import ReplayBuffer
from rl.policies.actor import FF_Actor as O_Actor
from rl.policies.critic import Dual_Q_Critic as Critic

import functools

import ray

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

# Runs policy for X episodes and returns average reward. Optionally render policy
def evaluate_policy(env, policy, eval_episodes=10, max_traj_len=400):
    avg_reward = 0.0
    avg_eplen = 0.0
    for _ in range(eval_episodes):
        obs = env.reset()
        t = 0
        done_bool = 0.0
        while not done_bool:
            t += 1
            action = policy.select_action(np.array(obs), param_noise=None)
            obs, reward, done, _ = env.step(action)
            done_bool = 1.0 if t + 1 == max_traj_len else float(done)
            avg_reward += reward
        avg_eplen += t

    avg_reward /= eval_episodes
    avg_eplen /= eval_episodes

    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward, avg_eplen


# TODO: Make each worker collect fixed amount of experience / don't stop computation once episodes are done
def parallel_collect_experience(policy, env_fn, act_noise, min_steps, max_traj_len, num_procs=4):

    all_transitions = ray.get([collect_experience.remote(env_fn, policy, min_steps // num_procs, max_traj_len, act_noise) for i in range(num_procs)])

    merged_transitions = np.concatenate(all_transitions)

    return merged_transitions, len(merged_transitions)

# sample experience for one episode and send to replay buffer
@ray.remote
@torch.no_grad()
def collect_experience(env_fn, policy, min_steps, max_traj_len, act_noise):

    env = env_fn()

    local_buffer = ReplayBuffer(max_size=min_steps)

    # reset environment
    obs = env.reset()
    done = False
    episode_reward = 0
    episode_timesteps = 0


    while not done and episode_timesteps < max_traj_len:

        # select action from policy
        action = policy.select_action(obs)
        if act_noise != 0:
            action = (action + np.random.normal(0, act_noise, size=1)).clip(-1, 1)

        # Perform action
        new_obs, reward, done, _ = env.step(action)
        done_bool = 1.0 if episode_timesteps + 1 == max_traj_len else float(done)
        episode_reward += reward

        # Store data in replay buffer
        transition = (obs, new_obs, action, reward, done_bool)
        local_buffer.add(transition)

        # update state
        obs = new_obs

        # increment counters
        episode_timesteps += 1

    # episode is over, return all transitions from this episode (list of tuples)
    return local_buffer.get_all_transitions()


class TD3():
    def __init__(self, state_dim, action_dim, max_action, a_lr, c_lr, env_name='NOT_SET'):
        self.actor = O_Actor(state_dim, action_dim, max_action=max_action, env_name=env_name).to(device)
        self.actor_target = O_Actor(state_dim, action_dim, max_action=max_action, env_name=env_name).to(device)
        self.actor_perturbed = O_Actor(state_dim, action_dim, max_action=max_action, env_name=env_name).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=a_lr)

        self.critic = Critic(state_dim, action_dim, hidden_size=256, env_name=env_name).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_size=256, env_name=env_name).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=c_lr)

        self.max_action = max_action

    def perturb_actor_parameters(self, param_noise):
        """Apply parameter noise to actor model, for exploration"""
        self.actor_perturbed.load_state_dict(self.actor.state_dict())
        params = self.actor_perturbed.state_dict()
        for name in params:
            if 'ln' in name: 
                pass 
            param = params[name]
            param += torch.randn(param.shape).to(device) * param_noise.current_stddev

    def select_action(self, state, param_noise=None):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)

        self.actor.eval()

        if param_noise is not None:
            return self.actor_perturbed(state).cpu().data.numpy().flatten()
        else:
            return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):

        avg_q1, avg_q2, q_loss, pi_loss, avg_noise, avg_action = (0,0,0,0,0,0)

        for it in range(iterations):

            # Sample replay buffer
            x, y, u, r, d = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1 - d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Select action according to policy and add clipped noise
            noise = torch.FloatTensor(u).data.normal_(
                0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) +
                           noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)
            
            # Keep track of Q estimates for logging
            avg_q1 += current_Q1
            avg_q2 += current_Q2
            avg_action += next_action

            # Compute critic loss
            critic_loss = F.mse_loss(
                current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            
            # Keep track of Q loss for logging
            q_loss += critic_loss

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % policy_freq == 0:

                # Compute actor loss
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

                # Keep track of pi loss for logging
                pi_loss += actor_loss

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data)

        # prep info for logging
        avg_q1 /= iterations
        avg_q2 /= iterations
        q_loss /= iterations
        pi_loss /= iterations
        avg_action /= iterations

        return torch.mean(avg_q1), torch.mean(avg_q1), q_loss, pi_loss, avg_action

    def save(self, save_path):
        if not os.path.exists('trained_models/syncTD3/'):
            os.makedirs('trained_models/syncTD3/')

        print("Saving model")

        filetype = ".pt"  # pytorch model
        torch.save(self.actor, os.path.join(
            save_path, "actor" + filetype))
        torch.save(self.critic, os.path.join(
            save_path, "critic_model" + filetype))

    def load(self, model_path):
        actor_path = os.path.join(model_path, "actor.pt")
        critic_path = os.path.join(model_path, "critic.pt")
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor = torch.load(actor_path)
            self.actor.eval()
        if critic_path is not None:
            self.critic = torch.load(critic_path)
            self.critic.eval()

# TODO: create way to resume experiment by loading actor and critic pt files
def run_experiment(args):
    from apex import env_factory, create_logger

    # wrapper function for creating parallelized envs
    env_fn = env_factory(args.env_name, state_est=args.state_est, mirror=args.mirror, history=args.history)
    max_traj_len = args.max_traj_len

    # Start ray
    ray.init(num_gpus=0, include_webui=True, redis_address=args.redis_address)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env_fn().observation_space.shape[0]
    action_dim = env_fn().action_space.shape[0]
    max_action = 1.0
    #max_action = float(env.action_space.high[0])

    print()
    print("Synchronous Twin-Delayed Deep Deterministic policy gradients:")
    print("\tenv:            {}".format(args.env_name))
    print("\tmax traj len:   {}".format(args.max_traj_len))
    print("\tseed:           {}".format(args.seed))
    print("\tmirror:         {}".format(args.mirror))
    print("\tnum procs:      {}".format(args.num_procs))
    print("\tmin steps:      {}".format(args.min_steps))
    print("\ta_lr:           {}".format(args.a_lr))
    print("\tc_lr:           {}".format(args.c_lr))
    print("\ttau:            {}".format(args.tau))
    print("\tgamma:          {}".format(args.discount))
    print("\tact noise:      {}".format(args.act_noise))
    print("\tparam noise:    {}".format(args.param_noise))
    if(args.param_noise):
        print("\tnoise scale:    {}".format(args.noise_scale))
    print("\tbatch size:     {}".format(args.batch_size))
    
    print("\tpolicy noise:   {}".format(args.policy_noise))
    print("\tnoise clip:     {}".format(args.noise_clip))
    print("\tpolicy freq:    {}".format(args.policy_freq))
    print()

    # Initialize policy, replay buffer
    policy = TD3(state_dim, action_dim, max_action, a_lr=args.a_lr, c_lr=args.c_lr, env_name=args.env_name)

    replay_buffer = ReplayBuffer()

    # create a tensorboard logging object
    logger = create_logger(args)

    # Initialize param noise (or set to None)
    param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.05, desired_action_stddev=args.noise_scale, adaptation_coefficient=1.05) if args.param_noise else None

    total_timesteps = 0
    total_updates = 0
    timesteps_since_eval = 0
    episode_num = 0
    
    # Evaluate untrained policy once
    ret, eplen = evaluate_policy(env_fn(), policy)
    logger.add_scalar("Test/Return", ret, total_updates)
    logger.add_scalar("Test/Eplen", eplen, total_updates)

    policy.save(logger.dir)

    while total_timesteps < args.max_timesteps:

        # collect parallel experience and add to replay buffer
        merged_transitions, episode_timesteps = parallel_collect_experience(policy, env_fn, args.act_noise, args.min_steps, max_traj_len, num_procs=args.num_procs)
        replay_buffer.add_parallel(merged_transitions)
        total_timesteps += episode_timesteps
        timesteps_since_eval += episode_timesteps
        episode_num += args.num_procs

        # Logging rollouts
        print("Total T: {} Episode Num: {} Episode T: {}".format(total_timesteps, episode_num, episode_timesteps))

        # update the policy
        avg_q1, avg_q2, q_loss, pi_loss, avg_action = policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau, args.policy_noise, args.noise_clip, args.policy_freq)
        total_updates += episode_timesteps      # this is how many iterations we did updates for

        # Logging training
        logger.add_scalar("Train/avg_q1", avg_q1, total_updates)
        logger.add_scalar("Train/avg_q2", avg_q2, total_updates)
        logger.add_scalar("Train/q_loss", q_loss, total_updates)
        logger.add_scalar("Train/pi_loss", pi_loss, total_updates)
        logger.add_histogram("Train/avg_action", avg_action, total_updates)

        # Evaluate episode
        if timesteps_since_eval >= args.eval_freq:
            timesteps_since_eval = 0
            ret, eplen = evaluate_policy(env_fn(), policy)

            # Logging Eval
            logger.add_scalar("Test/Return", ret, total_updates)
            logger.add_scalar("Test/Eplen", eplen, total_updates)
            logger.add_histogram("Test/avg_action", avg_action, total_updates)

            # Logging Totals
            logger.add_scalar("Misc/Timesteps", total_timesteps, total_updates)
            logger.add_scalar("Misc/ReplaySize", replay_buffer.ptr, total_updates)

            print("Total T: {}\tEval Return: {}\t Eval Eplen: {}".format(total_timesteps, ret, eplen))

            if args.save_models:
                policy.save()

    # Final evaluation
    ret, eplen = evaluate_policy(env_fn(), policy)
    logger.add_scalar("Test/Return", ret, total_updates)
    logger.add_scalar("Test/Eplen", eplen, total_updates)

    # Final Policy Save
    if args.save_models:
        policy.save()