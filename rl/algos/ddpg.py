import pickle
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ReplayBuffer():
  def __init__(self, state_dim, action_dim, max_size):
    self.max_size   = int(max_size)
    self.state      = torch.zeros((self.max_size, state_dim))
    self.next_state = torch.zeros((self.max_size, state_dim))
    self.action     = torch.zeros((self.max_size, action_dim))
    self.reward     = torch.zeros((self.max_size, 1))
    self.not_done   = torch.zeros((self.max_size, 1))

    self.size = 1

  def push(self, state, action, next_state, reward, done):
    if self.size == self.max_size:
      idx = np.random.randint(0, self.size)
    else:
      idx = self.size-1

    self.state[idx]      = torch.Tensor(state)
    self.next_state[idx] = torch.Tensor(next_state)
    self.action[idx]     = torch.Tensor(action)
    self.reward[idx]     = reward
    self.not_done[idx]   = 1 - done

    self.size = min(self.size+1, self.max_size)

  def sample(self, batch_size):
    idx = np.random.randint(0, self.size, size=batch_size)
    return self.state[idx], self.action[idx], self.next_state[idx], self.reward[idx], self.not_done[idx]

class DDPG():
  def __init__(self, actor, critic, a_lr, c_lr, discount=0.99, tau=0.001):
    self.behavioral_actor  = actor
    self.behavioral_critic = critic

    self.target_actor = copy.deepcopy(actor)
    self.target_critic = copy.deepcopy(critic)

    self.actor_optimizer  = torch.optim.Adam(self.behavioral_actor.parameters(), lr=a_lr)
    self.critic_optimizer = torch.optim.Adam(self.behavioral_critic.parameters(), lr=c_lr, weight_decay=1e-2)

    self.discount   = discount
    self.tau        = tau

  def update_policy(self, replay_buffer, batch_size=256):
    states, actions, next_states, rewards, not_dones = replay_buffer.sample(batch_size)

    states      = states
    next_states = next_states
    actions     = actions

    target_q = rewards + (not_dones * self.discount * self.target_critic(next_states, self.target_actor(next_states))).detach()
    current_q = self.behavioral_critic(states, actions)

    critic_loss = F.mse_loss(current_q, target_q)

    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()

    actor_loss = -self.behavioral_critic(states, self.behavioral_actor(states)).mean()

    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()

    for param, target_param in zip(self.behavioral_critic.parameters(), self.target_critic.parameters()):
      target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    for param, target_param in zip(self.behavioral_actor.parameters(), self.target_actor.parameters()):
      target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

def eval_policy(policy, env, evals=10):
  eval_reward = 0
  for _ in range(evals):
    state = env.reset()
    done = False
    while not done:
      action = policy.forward(torch.Tensor(state)).detach().numpy()
      state, reward, done, _ = env.step(action)
      eval_reward += reward
  return eval_reward/evals

def run_experiment(args):
  from time import time

  from apex import gym_factory, create_logger
  from rl.policies.critic import FF_Critic
  from rl.policies.actor import FF_Actor

  import locale
  locale.setlocale(locale.LC_ALL, '')

  # wrapper function for creating parallelized envs
  env = gym_factory(args.env_name)()

  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  env.seed(args.seed)

  obs_space = env.observation_space.shape[0]
  act_space = env.action_space.shape[0]

  actor = FF_Actor(obs_space, act_space, hidden_size=args.hidden_size)
  critic = FF_Critic(obs_space, act_space, hidden_size=args.hidden_size)

  print("Deep Deterministic Policy Gradients:")
  print("\tenv:          {}".format(args.env_name))
  print("\tseed:         {}".format(args.seed))
  print("\ttimesteps:    {:n}".format(args.timesteps))
  print("\tactor_lr:     {}".format(args.actor_lr))
  print("\tcritic_lr:    {}".format(args.critic_lr))
  print("\tdiscount:     {}".format(args.discount))
  print("\ttau:          {}".format(args.tau))
  print()
  algo = DDPG(actor, critic, args.actor_lr, args.critic_lr, discount=args.discount, tau=args.tau)

  replay_buff = ReplayBuffer(obs_space, act_space, args.timesteps)

  iter = 0
  episode_reward = 0
  episode_timesteps = 0
  state = env.reset()

  # create a tensorboard logging object
  logger = create_logger(args)

  # do an initial, baseline evaluation
  eval_reward = eval_policy(algo.behavioral_actor, env)
  logger.add_scalar('Eval reward', eval_reward, 0)

  state = env.reset().astype(np.float32)

  # Fill replay buffer, update policy until n timesteps have passed
  training_start = time()
  episode_start = time()

  timesteps = 0
  while timesteps < args.timesteps:

    # Generate a transition and append to the replay buffer
    if timesteps > args.start_timesteps:
      action = algo.behavioral_actor.forward(torch.Tensor(state)).detach().numpy()
      action += np.random.normal(0, args.expl_noise, size=act_space)
    else:
      action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)
    replay_buff.push(state, action, next_state.astype(np.float32), reward, done)

    episode_reward += reward
    episode_timesteps += 1
    
    # Update the policy once our replay buffer is big enough
    if replay_buff.size > args.batch_size:
      algo.update_policy(replay_buff, batch_size=args.batch_size)

    # Do some fancy debug printing/logging
    if done or episode_timesteps > args.traj_len:
      episode_elapsed = (time() - episode_start)
      episode_secs_per_sample = episode_elapsed / episode_timesteps
      logger.add_scalar('Episode reward', episode_reward, iter)

      completion = 1 - float(timesteps) / args.timesteps
      avg_sample_r = (time() - training_start)/timesteps
      secs_remaining = avg_sample_r * args.timesteps * completion
      hrs_remaining = int(secs_remaining//(60*60))
      min_remaining = int(secs_remaining - hrs_remaining*60*60)//60

      print("episode {:5d} | {:3.1f}s/1k samples | approx. {:3d}:{:2d}m remain\t\t".format(iter, 1000*episode_secs_per_sample, hrs_remaining, min_remaining), end='\r')

      if iter % args.eval_every == 0 and iter != 0:
        eval_reward = eval_policy(algo.behavioral_actor, env)
        logger.add_scalar('Eval reward', eval_reward, timesteps)

        print("evaluation after {:4d} episodes | return: {:7.3f} | timesteps {:9n}\t\t\t".format(iter, eval_reward, timesteps))

      next_state, done = env.reset(), False
      episode_start, episode_reward, episode_timesteps = time(), 0, 0
      iter += 1

    timesteps += 1
    state = next_state 
