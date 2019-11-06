import pickle
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RDPG():
  def __init__(self, actor, critic, a_lr, c_lr, discount=0.99, tau=0.001, center_reward=False):
    self.behavioral_actor = actor
    self.behavioral_critic = critic

    self.target_actor = copy.deepcopy(actor)
    self.target_critic = copy.deepcopy(critic)

    self.actor_optimizer  = torch.optim.Adam(self.behavioral_actor.parameters(), lr=a_lr)
    self.critic_optimizer = torch.optim.Adam(self.behavioral_critic.parameters(), lr=c_lr, weight_decay=1e-2)

    self.discount   = discount
    self.tau        = tau
    self.center_reward = center_reward

  def update_policy(self, replay_buffer, batch_size=256):
    raise NotImplementedError

    # THIS WILL NOT WORK WITH RECURRENT POLICIES
    #states, actions, next_states, rewards, not_dones = replay_buffer.sample(batch_size)

    states      = states
    next_states = next_states
    actions     = actions

    if self.center_reward:
      rewards = self.behavioral_critic.normalize_reward(rewards)

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

    return critic_loss.item()

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
  from rl.policies.critic import LSTM_Critic
  from rl.policies.actor import LSTM_Actor

  import locale
  locale.setlocale(locale.LC_ALL, '')

  # wrapper function for creating parallelized envs
  env = gym_factory(args.env_name)()

  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  env.seed(args.seed)

  obs_space = env.observation_space.shape[0]
  act_space = env.action_space.shape[0]

  actor = LSTM_Actor(obs_space, act_space, hidden_size=args.hidden_size)
  critic = LSTM_Critic(obs_space, act_space, hidden_size=args.hidden_size)

  print("Deep Deterministic Policy Gradients:")
  print("\tenv:          {}".format(args.env_name))
  print("\tseed:         {}".format(args.seed))
  print("\ttimesteps:    {:n}".format(args.timesteps))
  print("\tactor_lr:     {}".format(args.actor_lr))
  print("\tcritic_lr:    {}".format(args.critic_lr))
  print("\tdiscount:     {}".format(args.discount))
  print("\ttau:          {}".format(args.tau))
  print("\tnorm reward:  {}".format(args.center_reward))
  print()

  algo = RDPG(actor, critic, args.actor_lr, args.critic_lr, discount=args.discount, tau=args.tau, center_reward=args.center_reward)

  replay_buff = ReplayBuffer(obs_space, act_space, args.timesteps)

  iter = 0
  episode_reward = 0
  episode_timesteps = 0
  state = env.reset()

  # create a tensorboard logging object
  logger = create_logger(args)





