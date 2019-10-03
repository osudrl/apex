import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import time
import numpy as np

import locale
locale.setlocale(locale.LC_ALL, '')

from rl.algos.ars import ARS
from rl.policies import LinearMLP, RecurrentNet

def eval_fn(policy, env, reward_shift, traj_len, visualize=False, normalize=False):

  if hasattr(policy, 'init_hidden_state'):
    policy.init_hidden_state()

  state = torch.tensor(env.reset()).float()
  rollout_reward = 0
  done = False

  timesteps = 0
  while not done and timesteps < traj_len:
    action = policy.forward(state, update_normalizer=normalize).detach()

    state, reward, done, _ = env.step(action)
    state = torch.tensor(state).float()
    rollout_reward += reward - reward_shift
    timesteps+=1
  return rollout_reward, timesteps

def train(policy_thunk, env_thunk, args):
  algo = ARS(policy_thunk, env_thunk, deltas=args.deltas, step_size=args.lr, std=args.std, workers=args.workers)

  if args.algo == 'v2':
    normalize_states = True
  else:
    normalize_states = False

  def black_box(p, env):
    return eval_fn(p, env, args.reward_shift, args.traj_len, normalize=normalize_states)

  avg_reward = 0
  timesteps = 0
  i = 0
  while timesteps < args.timesteps:
    if not i % args.average_every:
      avg_reward = 0
      print()

    start = time.time()
    samples = algo.step(black_box)
    elapsed = time.time() - start
    reward, _ = eval_fn(algo.policy, env, 0, args.traj_len, normalize=normalize_states)

    timesteps += samples
    avg_reward += reward
    secs_per_sample = 1000 * elapsed / samples
    print(("iter {:4d} | "
           "ret {:6.2f} | "
           "last {:3d} iters: {:6.2f} | "
           "{:0.4f}s per 1k steps | "
           "timesteps {:10n}").format(i+1,  \
            reward, (i%args.average_every)+1,      \
            avg_reward/((i%args.average_every)+1), \
            secs_per_sample, timesteps),    \
            end="\r")
    i += 1

    torch.save(algo.policy, args.save_model)

if __name__ == "__main__":
  import argparse
  from apex import print_logo, gym_factory

  parser = argparse.ArgumentParser()
  parser.add_argument("--workers", type=int, default=4)
  parser.add_argument("--env_name",     "-e",   default="Hopper-v2")
  parser.add_argument("--hidden_size",          default=32)
  parser.add_argument("--seed",         "-s",   default=0, type=int)
  parser.add_argument("--timesteps",    "-t",   default=1e8, type=int)
  parser.add_argument("--load_model",   "-l",   default=None, type=str)
  parser.add_argument("--save_model",   "-m",   default="./trained_models/ars/ars.pt", type=str)
  parser.add_argument('--std',          "-sd",  default=0.005, type=float)
  parser.add_argument("--deltas",       "-d",   default=64, type=int)
  parser.add_argument("--lr",           "-lr",  default=0.01, type=float)
  parser.add_argument("--reward_shift", "-rs",  default=1, type=float)
  parser.add_argument("--traj_len",     "-tl",  default=1000, type=int)
  parser.add_argument("--recurrent",    "-r",  action='store_true')
  parser.add_argument("--algo",         "-a",   default='v1', type=str)

  parser.add_argument("--log_dir",      default="./logs/ars/experiments/", type=str)
  parser.add_argument("--average_every",default=10, type=int)
  args = parser.parse_args()

  if args.algo not in ['v1', 'v2']:
    print("Valid arguments for --algo are 'v1' and 'v2'")
    exit(1)

  print_logo(subtitle="Augmented Random Search for reinforcement learning")

  # wrapper function for creating parallelized envs
  env_thunk = gym_factory(args.env_name)

  with env_thunk() as env:
      obs_space = env.observation_space.shape[0]
      act_space = env.action_space.shape[0]

  # wrapper function for creating parallelized policies
  def policy_thunk():
    if args.load_model is not None:
      return torch.load(args.load_model)
    else:
      if not args.recurrent:
        policy = LinearMLP(obs_space, act_space, hidden_size=args.hidden_size).float()
      else:
        policy = RecurrentNet(obs_space, act_space, hidden_size=args.hidden_size).float()
      for p in policy.parameters():
        p.data = torch.zeros(p.shape)
      return policy

  print("Augmented Random Search:")
  print("\tenv:          {}".format(args.env_name))
  print("\tseed:         {}".format(args.seed))
  print("\ttimesteps:    {}".format(args.timesteps))
  print("\tstd:          {}".format(args.std))
  print("\tdeltas:       {}".format(args.deltas))
  print("\tstep size:    {}".format(args.lr))
  print("\treward shift: {}".format(args.reward_shift))
  print()

  train(policy_thunk, env_thunk, args)

