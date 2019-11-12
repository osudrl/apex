import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from torch.nn.utils.rnn import pad_sequence

class ReplayBuffer():
  def __init__(self, state_dim, action_dim, max_size):
    self.max_size   = int(max_size)
    self.state      = torch.zeros((self.max_size, state_dim))
    self.next_state = torch.zeros((self.max_size, state_dim))
    self.action     = torch.zeros((self.max_size, action_dim))
    self.reward     = torch.zeros((self.max_size, 1))
    self.not_done   = torch.zeros((self.max_size, 1))

    self.trajectory_idx = [0]
    self.trajectories = 0

    self.size = 1

  def push(self, state, action, next_state, reward, done):
    if self.size == self.max_size:
      print("\nBuffer full.")
      exit(1)

    idx = self.size-1

    self.state[idx]      = torch.Tensor(state)
    self.next_state[idx] = torch.Tensor(next_state)
    self.action[idx]     = torch.Tensor(action)
    self.reward[idx]     = reward
    self.not_done[idx]   = 1 - done

    if done:
      self.trajectory_idx.append(self.size)
      self.trajectories += 1

    self.size = min(self.size+1, self.max_size)

  def sample_trajectory(self, max_len):

    traj_idx = np.random.randint(0, self.trajectories-1)
    start_idx = self.trajectory_idx[traj_idx]
    end_idx = start_idx + 1

    while self.not_done[end_idx] == 1 and end_idx - start_idx < max_len:
      end_idx += 1
    end_idx += 1

    traj_states = self.state[start_idx:end_idx]
    next_states = self.next_state[start_idx:end_idx]
    actions     = self.action[start_idx:end_idx]
    rewards     = self.reward[start_idx:end_idx]
    not_dones   = self.not_done[start_idx:end_idx]

    # Return an entire episode
    return traj_states, actions, next_states, rewards, not_dones

  def sample(self, batch_size, sample_trajectories=False, max_len=1000):
    if sample_trajectories:
      # Collect raw trajectories from replay buffer
      raw_traj = [self.sample_trajectory(max_len) for _ in range(batch_size)]
      steps = sum([len(traj[0]) for traj in raw_traj])

      # Extract trajectory info into separate lists to be padded and batched
      states      = [traj[0] for traj in raw_traj]
      actions     = [traj[1] for traj in raw_traj]
      next_states = [traj[2] for traj in raw_traj]
      rewards     = [traj[3] for traj in raw_traj]
      not_dones   = [traj[4] for traj in raw_traj]

      # Pad all trajectories to be the same length, shape is (traj_len x batch_size x dim)
      states      = pad_sequence(states, batch_first=False)
      actions     = pad_sequence(actions, batch_first=False)
      next_states = pad_sequence(next_states, batch_first=False)
      rewards     = pad_sequence(rewards, batch_first=False)
      not_dones   = pad_sequence(not_dones, batch_first=False)

      return states, actions, next_states, rewards, not_dones, steps

    else:
      idx = np.random.randint(0, self.size, size=batch_size)
      return self.state[idx], self.action[idx], self.next_state[idx], self.reward[idx], self.not_done[idx], batch_size

class DPG():
  def __init__(self, actor, critic, a_lr, c_lr, discount=0.99, tau=0.001, center_reward=False, normalize=False):

    if actor.is_recurrent or critic.is_recurrent:
      self.recurrent = True
    else:
      self.recurrent = False

    self.behavioral_actor  = actor
    self.behavioral_critic = critic

    self.target_actor = copy.deepcopy(actor)
    self.target_critic = copy.deepcopy(critic)

    self.soft_update(1.0)

    self.actor_optimizer  = torch.optim.Adam(self.behavioral_actor.parameters(), lr=a_lr)
    self.critic_optimizer = torch.optim.Adam(self.behavioral_critic.parameters(), lr=c_lr, weight_decay=1e-2)

    self.discount   = discount
    self.tau        = tau
    self.center_reward = center_reward

    self.normalize = normalize

  def soft_update(self, tau):
    for param, target_param in zip(self.behavioral_critic.parameters(), self.target_critic.parameters()):
      target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    for param, target_param in zip(self.behavioral_actor.parameters(), self.target_actor.parameters()):
      target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

  def update_policy(self, replay_buffer, batch_size=256, traj_len=1000, grad_clip=None):
    states, actions, next_states, rewards, not_dones, steps = replay_buffer.sample(batch_size, sample_trajectories=self.recurrent, max_len=traj_len)

    with torch.no_grad():
      if self.normalize:
        states      = self.behavioral_actor.normalize_state(states, update=False)
        next_states = self.behavioral_actor.normalize_state(next_states, update=False)

      target_q = rewards + (not_dones * self.discount * self.target_critic(next_states, self.target_actor(next_states)))
    current_q = self.behavioral_critic(states, actions)

    critic_loss = F.mse_loss(current_q, target_q)

    self.critic_optimizer.zero_grad()
    critic_loss.backward()

    if grad_clip is not None:
      torch.nn.utils.clip_grad_norm_(self.behavioral_critic.parameters(), grad_clip)

    self.critic_optimizer.step()

    actor_loss = -self.behavioral_critic(states, self.behavioral_actor(states)).mean()

    self.actor_optimizer.zero_grad()
    actor_loss.backward()

    if grad_clip is not None:
      torch.nn.utils.clip_grad_norm_(self.behavioral_actor.parameters(), grad_clip)

    self.actor_optimizer.step()
    
    self.soft_update(self.tau)

    return critic_loss.item(), steps

def eval_policy(policy, env, evals=10, max_traj_len=1000):
  eval_reward = 0
  for _ in range(evals):
    state = env.reset()
    done = False
    timesteps = 0

    if hasattr(policy, 'init_hidden_state'):
      policy.init_hidden_state()

    while not done and timesteps < max_traj_len:
      state = policy.normalize_state(state, update=False)
      action = policy.forward(torch.Tensor(state)).detach().numpy()
      state, reward, done, _ = env.step(action)
      eval_reward += reward
      timesteps += 1

  return eval_reward/evals

def collect_experience(policy, env, replay_buffer, initial_state, steps, random_action=False, noise=0.2, do_trajectory=False, max_len=1000, normalize=False):
  if normalize:
    state = policy.normalize_state(torch.Tensor(initial_state))
  else:
    state = torch.Tensor(initial_state)

  if not random_action:
    a = policy.forward(torch.Tensor(initial_state)).detach().numpy() + np.random.normal(0, noise, size=policy.action_dim)
  else:
    a = np.random.randn(policy.action_dim)

  state_t1, r, done, _ = env.step(a)

  if done or steps > max_len:
    state_t1 = env.reset()
    done = True
    if hasattr(policy, 'init_hidden_state'):
      policy.init_hidden_state()

  replay_buffer.push(initial_state, a, state_t1.astype(np.float32), r, done)

  return state_t1, r, done

def run_experiment(args):
  from time import time

  from apex import env_factory, create_logger
  from rl.policies.critic import FF_Critic, LSTM_Critic
  from rl.policies.actor import FF_Actor, LSTM_Actor

  import locale, os
  locale.setlocale(locale.LC_ALL, '')

  # wrapper function for creating parallelized envs
  env = env_factory(args.env_name)()
  eval_env = env_factory(args.env_name)()

  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if hasattr(env, 'seed'):
    env.seed(args.seed)

  obs_space = env.observation_space.shape[0]
  act_space = env.action_space.shape[0]

  if args.recurrent:
    actor = LSTM_Actor(obs_space, act_space, hidden_size=args.hidden_size, env_name=args.env_name, hidden_layers=args.layers)
    critic = LSTM_Critic(obs_space, act_space, hidden_size=args.hidden_size, env_name=args.env_name, hidden_layers=args.layers)
  else:
    actor = FF_Actor(obs_space, act_space, hidden_size=args.hidden_size, env_name=args.env_name, hidden_layers=args.layers)
    critic = FF_Critic(obs_space, act_space, hidden_size=args.hidden_size, env_name=args.env_name, hidden_layers=args.layers)

  algo = DPG(actor, critic, args.a_lr, args.c_lr, discount=args.discount, tau=args.tau, center_reward=args.center_reward, normalize=args.normalize)

  replay_buff = ReplayBuffer(obs_space, act_space, args.timesteps)

  if algo.recurrent:
    print("Recurrent Deterministic Policy Gradients:")
  else:
    print("Deep Deterministic Policy Gradients:")
  print("\tenv:            {}".format(args.env_name))
  print("\tseed:           {}".format(args.seed))
  print("\ttimesteps:      {:n}".format(args.timesteps))
  print("\tactor_lr:       {}".format(args.a_lr))
  print("\tcritic_lr:      {}".format(args.c_lr))
  print("\tdiscount:       {}".format(args.discount))
  print("\ttau:            {}".format(args.tau))
  print("\tnorm reward:    {}".format(args.center_reward))
  print("\tbatch_size:     {}".format(args.batch_size))
  print("\twarmup period:  {:n}".format(args.start_timesteps))
  print()

  iter = 0
  episode_reward = 0
  episode_timesteps = 0

  # create a tensorboard logging object
  logger = create_logger(args)

  if args.save_actor is None:
    args.save_actor = os.path.join(logger.dir, 'actor.pt')

  if args.save_critic is None:
    args.save_critic = os.path.join(logger.dir, 'critic.pt')

  # Keep track of some statistics for each episode
  training_start = time()
  episode_start = time()
  episode_loss = 0
  update_steps = 0
  best_reward = None

  # Fill replay buffer, update policy until n timesteps have passed
  timesteps = 0
  state = env.reset().astype(np.float32)
  while timesteps < args.timesteps:
    buffer_ready = (algo.recurrent and iter > args.batch_size) or (not algo.recurrent and replay_buff.size > args.batch_size)
    warmup = timesteps < args.start_timesteps

    state, r, done = collect_experience(algo.behavioral_actor, env, replay_buff, state, episode_timesteps,
                                               max_len=args.traj_len,
                                               random_action=warmup,
                                               noise=args.expl_noise, 
                                               do_trajectory=algo.recurrent,
                                               normalize=algo.normalize)
    episode_reward += r
    episode_timesteps += 1
    timesteps += 1

    # Update the policy once our replay buffer is big enough
    if buffer_ready and done and not warmup:
      update_steps = 0
      if not algo.recurrent:
        num_updates = episode_timesteps * args.updates
      else:
        num_updates = args.updates
      for _ in range(num_updates):
        u_loss, u_steps = algo.update_policy(replay_buff, args.batch_size, traj_len=args.traj_len)
        episode_loss += u_loss / num_updates
        update_steps += u_steps

    if done:
      episode_elapsed = (time() - episode_start)
      episode_secs_per_sample = episode_elapsed / episode_timesteps
      logger.add_scalar(args.env_name + ' episode length', episode_timesteps, iter)
      logger.add_scalar(args.env_name + ' episode reward', episode_reward, iter)
      logger.add_scalar(args.env_name + ' critic loss', episode_loss, iter)

      completion = 1 - float(timesteps) / args.timesteps
      avg_sample_r = (time() - training_start)/timesteps
      secs_remaining = avg_sample_r * args.timesteps * completion
      hrs_remaining = int(secs_remaining//(60*60))
      min_remaining = int(secs_remaining - hrs_remaining*60*60)//60

      if iter % args.eval_every == 0 and iter != 0:
        eval_reward = eval_policy(algo.behavioral_actor, eval_env, max_traj_len=args.traj_len)
        logger.add_scalar(args.env_name + ' eval episode', eval_reward, iter)
        logger.add_scalar(args.env_name + ' eval timestep', eval_reward, timesteps)

        print("evaluation after {:4d} episodes | return: {:7.3f} | timesteps {:9n}{:100s}".format(iter, eval_reward, timesteps, ''))

        if best_reward is None or eval_reward > best_reward:
          torch.save(algo.behavioral_actor, args.save_actor)
          torch.save(algo.behavioral_critic, args.save_critic)
          best_reward = eval_reward
          print("\t(best policy so far! saving to {})".format(args.save_actor))

    try:
      print("episode {:5d} | episode timestep {:5d}/{:5d} | return {:5.1f} | update timesteps: {:7n} | {:3.1f}s/1k samples | approx. {:3d}h {:02d}m remain\t\t\t\t".format(
        iter, 
        episode_timesteps, 
        args.traj_len, 
        episode_reward, 
        update_steps, 
        1000*episode_secs_per_sample, 
        hrs_remaining, 
        min_remaining), end='\r')

    except NameError:
      pass

    if done:
      if hasattr(algo.behavioral_actor, 'init_hidden_state'):
        algo.behavioral_actor.init_hidden_state()

      episode_start, episode_reward, episode_timesteps, episode_loss = time(), 0, 0, 0
      iter += 1
