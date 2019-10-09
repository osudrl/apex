import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from apex import gym_factory

# Based on https://github.com/sfujim/TD3/blob/master/DDPG.py

class Actor_tmp(nn.Module):
  def __init__(self, state_dim, output_dim, hidden_size=256):
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
  def __init__(self, state_dim, action_dim, hidden_size=256):
    super(Critic_tmp, self).__init__()

    #self.state_linear = nn.Linear(state_dim, hidden_size)
    #self.action_linear = nn.Linear(action_dim, hidden_size)

    self.l1 = nn.Linear(state_dim, hidden_size)
    self.l2 = nn.Linear(hidden_size + action_dim, hidden_size)
    self.l3 = nn.Linear(hidden_size, 1)

    self.recurrent = False

  def forward(self, state, action):

    #state = self.state_linear(state)
    #action = self.action_linear(action)
    #x = F.relu(torch.add(state, action))

    x = F.relu(self.l1(state))
    x = F.relu(self.l2(torch.cat([x, action], 1)))
    #x = F.relu(self.l2(x))
    return self.l3(x)

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
  def __init__(self, actor, critic, max_action, a_lr, c_lr, discount=0.99, tau=0.001):
    self.behavioral_actor  = actor
    self.behavioral_critic = critic

    #self.target_actor  = pickle.loads(pickle.dumps(actor))
    #self.target_critic = pickle.loads(pickle.dumps(critic))

    self.target_actor = copy.deepcopy(actor)
    self.target_critic = copy.deepcopy(critic)

    self.actor_optimizer  = torch.optim.Adam(self.behavioral_actor.parameters(), lr=a_lr)
    self.critic_optimizer = torch.optim.Adam(self.behavioral_critic.parameters(), lr=c_lr, weight_decay=1e-2)

    #self.max_action = max_action
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

    # Update the frozen target models
    for param, target_param in zip(self.behavioral_critic.parameters(), self.target_critic.parameters()):
      target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    for param, target_param in zip(self.behavioral_actor.parameters(), self.target_actor.parameters()):
      target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

def run_experiment(args):

  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  # wrapper function for creating parallelized envs
  env = gym_factory(args.env_name)()
  env.seed(args.seed)

  obs_space = env.observation_space.shape[0]
  act_space = env.action_space.shape[0]

  actor = Actor_tmp(obs_space, act_space)
  critic = Critic_tmp(obs_space, act_space)

  print("Deep Deterministic Policy Gradients:")
  print("\tenv:          {}".format(args.env_name))
  print("\tseed:         {}".format(args.seed))
  print("\ttimesteps:    {:n}".format(args.timesteps))
  print("\tactor_lr:     {}".format(args.actor_lr))
  print("\tcritic_lr:    {}".format(args.critic_lr))
  print("\tdiscount:     {}".format(args.discount))
  print("\ttau:          {}".format(args.tau))
  print()
  algo = DDPG(actor, critic, 1.0, args.actor_lr, args.critic_lr, discount=args.discount, tau=args.tau)

  replay_buff = ReplayBuffer(obs_space, act_space, args.timesteps)

  timesteps = 0
  iter = 0
  episode_reward = 0
  episode_timesteps = 0
  state = env.reset()
  eval_every = 100
  while timesteps < args.timesteps:

    if timesteps > args.start_timesteps:
      action_noise = np.random.normal(0, 0.2, size=act_space)
      action = algo.behavioral_actor.forward(torch.Tensor(state)).detach().numpy()
      action += action_noise
    else:
      action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)

    state = state.astype(np.float32)
    next_state = next_state.astype(np.float32)

    episode_reward += reward
    episode_timesteps += 1
    
    replay_buff.push(state, action, next_state, reward, done)

    if replay_buff.size > args.batch_size:
      algo.update_policy(replay_buff, batch_size=args.batch_size)

    if done or episode_timesteps > args.traj_len:
      print("Episode {}, return {:4.3f}, steps {:3d}".format(iter, episode_reward, episode_timesteps))
      state = env.reset()
      done = False
      state, done = env.reset(), False
      episode_reward, episode_timesteps, iter = 0, 0, iter+1

      if iter % eval_every == 0:
        eval_reward = 0
        eval_steps = 0
        for _ in range(10):
          state = env.reset()
          done = False
          while not done:
            action = algo.behavioral_actor.forward(torch.Tensor(state)).detach().numpy()
            state, reward, done, _ = env.step(action)
            eval_reward += reward
            eval_steps += 1
            #env.render()
        print("Episodes: {:4d} | Return: {:4.3f} | Timesteps {:n} | Evaluation steps: {:4.3f}\n".format(iter, eval_reward/10, timesteps, eval_steps))
        next_state = env.reset()

    timesteps += 1
    state = next_state 

