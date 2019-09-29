import time
import numpy as np
#import os
import torch
import ray

@ray.remote
class ARS_process(object):
  def __init__(self, policy_thunk, env_thunk, std=0.0075):
    self.policy = policy_thunk()
    self.env    = env_thunk()
    self.param_shape = [x.shape for x in self.policy.parameters()]
    self.std = std

  def update_policy(self, new_params):
    for p, new_p in zip(self.policy.parameters(), new_params):
      p.data = new_p.data

  def generate_delta(self):
    delta = [torch.normal(0, self.std, shape) for shape in self.param_shape]
    return delta

  def rollout(self, current_params, black_box, rollouts=1):
    self.update_policy(current_params)
    self.delta = self.generate_delta()

    ret = []
    timesteps = 0
    for _ in range(rollouts):
      for p, dp in zip(self.policy.parameters(), self.delta):
        p.data += dp;
      r_pos = black_box(self.policy)

      for p, dp in zip(self.policy.parameters(), self.delta):
        p.data -= 2*dp;
      r_neg = black_box(self.policy)

      for p, dp in zip(self.policy.parameters(), self.delta):
        p.data += dp;

      if isinstance(r_pos, tuple):
        timesteps += r_pos[1]
        r_pos = r_pos[0]

      if isinstance(r_neg, tuple):
        timesteps += r_neg[1]
        r_neg = r_neg[0]
      
      ret.append({'delta': self.delta, 'r_pos': r_pos, 'r_neg': r_neg, 'timesteps': timesteps})

    return ret

class ARS:
  def __init__(self, policy_thunk, env_thunk, step_size=0.02, std=0.0075, deltas=32, workers=4, top_n=None):
    self.std = std
    self.num_deltas = deltas
    self.num_workers = workers
    self.step_size = step_size
    self.policy = policy_thunk()

    if top_n is not None:
      self.top_n = top_n
    else:
      self.top_n = deltas

    if not ray.is_initialized():
      ray.init()

    self.workers = [ARS_process.remote(policy_thunk, env_thunk, std=std) for _ in range(workers)]

  def step(self, black_box):
    pid = ray.put(list(self.policy.parameters())) # place the current policy parameters in shared mem

    rollouts = self.num_deltas // self.num_workers # number of rollouts per worker

    rollout_ids = [w.rollout.remote(pid, black_box, rollouts) for w in self.workers] # do rollouts
    results = ray.get(rollout_ids) # retrieve rollout results from pool

    results = [item for sublist in results for item in sublist] # flattens list of lists

    r_pos = [item['r_pos'] for item in results]
    r_neg = [item['r_neg'] for item in results]
    delta = [item['delta'] for item in results]
    timesteps = sum([item['timesteps'] for item in results])

    r_std = np.std(r_pos + r_neg)

    # if use top performing directions
    if self.top_n < self.num_deltas:
      sorted_indices = np.argsort(np.maximum(r_pos, r_neg))
      r_pos = r_pos[sorted_indices]
      r_neg = r_neg[sorted_indices]
      delta = delta[sorted_indices]

    r_pos /= self.top_n * r_std
    r_neg /= self.top_n * r_std

    for r_p, r_n, d in zip(r_pos, r_neg, delta):
      for param, d_param in zip(self.policy.parameters(), d):
        param.data += self.step_size * (r_p - r_n) * d_param.data
    return timesteps
