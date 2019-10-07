import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from apex import gym_factory
import copy

# Based on https://github.com/sfujim/TD3/blob/master/DDPG.py

class Actor_tmp(nn.Module):
  def __init__(self, state_dim, output_dim, hidden_size=64):
    super(Actor_tmp, self).__init__()

    self.l1 = nn.Linear(state_dim, hidden_size)
    self.l2 = nn.Linear(hidden_size, hidden_size)
    self.l3 = nn.Linear(hidden_size, output_dim)

    self.recurrent = False

  def forward(self, state):
    x = F.relu(self.l1(state))
    x = F.relu(self.l2(x))
    return torch.tanh(self.l3(x))

class Critic_tmp(nn.Module):
  def __init__(self, state_dim, action_dim, hidden_size=64):
    super(Critic_tmp, self).__init__()

    self.l1 = nn.Linear(state_dim + action_dim, hidden_size)
    self.l2 = nn.Linear(hidden_size, hidden_size)
    self.l3 = nn.Linear(hidden_size, 1)

    self.recurrent = False

  def forward(self, state, action):
    x = torch.cat([state, action])
    x = F.relu(self.l1(x))
    x = F.relu(self.l2(x))
    return self.l3(x)

class ReplayBuffer():
  def __init__(self, state_dim, action_dim, max_size):
    self.max_size   = int(max_size)
    self.state      = np.zeros((self.max_size, state_dim))
    self.next_state = np.zeros((self.max_size, state_dim))
    self.action     = np.zeros((self.max_size, action_dim))
    self.reward     = np.zeros((self.max_size, 1))
    self.not_done   = np.zeros((self.max_size, 1))

    self.size = 1

  def push(self, state, action, next_state, reward, done):
    if self.size == self.max_size:
      idx = np.random.randint(0) % self.size
    else:
      idx = self.size-1

    self.state[idx]      = state
    self.next_state[idx] = next_state
    self.action[idx]     = action
    self.reward[idx]     = reward
    self.not_done[idx]   = 1 - done

    self.size = min(self.size+1, self.max_size)

  def sample(self, batch_size):
    idx = np.random.randint(0) % self.size
    return self.state[idx], self.action[idx], self.next_state[idx], self.reward[idx], self.not_done[idx]

class DDPG():
  def __init__(self, actor, critic, max_action, a_lr, c_lr, discount=0.99, tau=0.01):
    self.behavioral_actor  = actor
    self.behavioral_critic = critic

    #self.recurrent = True if actor.recurrent == True or critic.recurrent == True else False

    self.target_actor  = copy.deepcopy(actor)
    self.target_critic = copy.deepcopy(critic)

    self.actor_optimizer  = torch.optim.SGD(self.behavioral_actor.parameters(), lr=a_lr)
    self.critic_optimizer = torch.optim.SGD(self.behavioral_critic.parameters(), lr=a_lr)

    self.max_action = max_action
    self.discount   = discount
    self.tau        = tau

  def update_policy(self, replay_buffer, batch_size=64):
    states, actions, next_states, rewards, not_dones = replay_buffer.sample(batch_size)

    target_q = reward + (not_dones * self.discount * self.target_critic(next_states, self.target_actor(next_states))).detach()
    current_q = self.behavioral_critic(states, actions)

    critic_loss = F.mse_loss(current_q, target_q)

    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()

    actor_loss = -self.critic(states, self.actor(states)).mean()

    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()

    # Update the frozen target models
    for param, target_param in zip(self.behavioral_critic.parameters(), self.target_critic.parameters()):
      target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    for param, target_param in zip(self.behavioral_actor.parameters(), self.target_actor.parameters()):
      target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

def run_experiment(args):

  # wrapper function for creating parallelized envs
  env = gym_factory(args.env_name)()

  obs_space = env.observation_space.shape[0]
  act_space = env.action_space.shape[0]

  actor = Actor_tmp(obs_space, act_space)
  critic = Critic_tmp(obs_space, act_space)

  algo = DDPG(actor, critic, 1.0, args.actor_lr, args.critic_lr, discount=args.discount, tau=args.tau)

  replay_buff = ReplayBuffer(obs_space, act_space, args.timesteps)

  timesteps = 0
  state = env.reset()
  while timesteps < args.timesteps:
    state = torch.Tensor(state)

    action = actor.forward(state).detach()
    next_state, reward, done, _ = env.step(action)
    
    replay_buff.push(state, action, next_state, reward, done)

    if replay_buff.size > args.batch_size:
      algo.update_policy(replay_buff, batch_size=args.batch_size)

    state = next_state 

    if done:
      state = env.reset()
    env.render()





