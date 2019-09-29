from rl.algos.ars import ARS
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import time

import locale
locale.setlocale(locale.LC_ALL, '')


class Policy(nn.Module):
  def __init__(self, state_dim, action_dim, hidden_size=32):
    super(Policy, self).__init__()

    self.l1 = nn.Linear(state_dim, hidden_size)
    self.l2 = nn.Linear(hidden_size, action_dim)

    for p in self.parameters():
      p.data = torch.zeros(p.shape)
  
  def forward(self, state):
    a = self.l1(state)
    return self.l2(a)


def eval_fn(policy, env, visualize=False, reward_shift=0):

  state = torch.tensor(env.reset()).float()
  rollout_reward = 0
  done = False

  timesteps = 0
  while not done:
    action = policy.forward(state).detach()

    if visualize:
      env.render()
    
    state, reward, done, _ = env.step(action)
    state = torch.tensor(state).float()
    rollout_reward += reward - reward_shift
    timesteps+=1
  return rollout_reward, timesteps

def train(policy_thunk, env_thunk, iters=1000, print_output=False, reset_every=10):
  algo = ARS(policy_thunk, env_thunk, deltas=64, step_size=0.005)

  def black_box(p):
    return eval_fn(p, env, reward_shift=1)

  avg_reward = 0
  timesteps = 0
  for i in range(iters):
    if not i % reset_every:
      avg_reward = 0
      print()

    start = time.time()
    samples = algo.step(black_box)
    elapsed = time.time() - start

    reward, _ = eval_fn(algo.policy, env)

    if print_output:
      timesteps += samples
      avg_reward += reward
      secs_per_sample = 1000 * elapsed / samples
      print("iter {:4d} | ret {:6.2f} | last {:3d} iters: {:6.2f} | {:0.4f}s per 1k steps | timesteps {:10n}".format(i+1, reward, (i%reset_every)+1, avg_reward/((i%reset_every)+1), secs_per_sample, timesteps), end="\r")

if __name__ == "__main__":
  iters = 1000
  reset_every = 10

  # wrapper function for creating parallelized envs
  def env_thunk():
    return gym.make("Hopper-v2")

  with env_thunk() as env:
    obs_space = env.observation_space.shape[0]
    act_space = env.action_space.shape[0]

  # wrapper function for creating parallelized policies
  def policy_thunk():
    return Policy(obs_space, act_space).float()

  train(policy_thunk, env_thunk, print_output=True)
