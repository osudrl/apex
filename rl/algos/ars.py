import numpy as np
import torch
import ray
import time

# This function adapted from https://github.com/modestyachts/ARS/blob/master/code/shared_noise.py
# (Thanks to Horia Mania)
# In a nutshell, this created the deltas to be used in the experiment ahead of time.
# Over the course of the experiment, workers will access the deltas allocated and returned
# by this function.
@ray.remote
def create_shared_noise(seed=12345, count=2500000, std=1):
  rand_state = np.random.RandomState(seed)
  noise = np.random.RandomState(seed).randn(count).astype(np.float32) * std
  return noise

# This class adapted from https://github.com/modestyachts/ARS/blob/master/code/shared_noise.py
class SharedNoiseTable(object):
  def __init__(self, noise, param_shape, seed=0):
    self.rg = np.random.RandomState(seed)
    self.noise = noise
    self.param_shape = param_shape
    self.param_size = sum([np.prod(shape) for shape in param_shape])

    assert self.noise.dtype == np.float32

  def get_random_idx(self):
    return self.rg.randint(0, len(self.noise) - self.param_size + 1)

  def get_raw_noise(self, i):
    return self.noise[i:i+self.param_size]

  def get_delta(self, idx=None):

    if idx is None:
      idx = self.get_random_idx()

    raw_noise = self.get_raw_noise(idx)

    ret = []
    i = 0
    for x in self.param_shape:
      size = np.prod(x)
      chunk = raw_noise[i:i+size]
      #np.reshape(chunk, x)
      ret.append(np.reshape(chunk, x))
      #ret.append(chunk)

      i += size
    return idx, ret

@ray.remote
class ARS_process(object):
  def __init__(self, policy_thunk, env_thunk, deltas, std, process_seed):
    self.policy = policy_thunk()
    self.env    = env_thunk()
    self.param_shape = [x.shape for x in self.policy.parameters()]
    self.std = std

    self.deltas = SharedNoiseTable(deltas, self.param_shape, seed=process_seed)

  def update_policy(self, new_params):
    pass
    #for p, new_p in zip(self.policy.parameters(), new_params):
      #new_p.copy_(p)

  """
  def tmp_run_rollout(self, reward_shift=1):
    self.env.seed(0)
    state = torch.tensor(self.env.reset()).float()
    rollout_reward = 0
    done = False

    timesteps = 0
    while not done:
      action = self.policy.forward(state).detach()

      #if not timesteps:
        #print("first action:", action[:3])

      state, reward, done, _ = self.env.step(action)
      #self.env.render()
      state = torch.tensor(state).float()
      rollout_reward += reward - reward_shift
      timesteps+=1
    #print("end state:", state[:3])
    return rollout_reward, timesteps
  """

  def rollout(self, current_params, black_box, rollouts=1):
  #def rollout(self, current_params, rollouts=1):
    self.update_policy(current_params)
    idx, delta = self.deltas.get_delta()
    #print("STARTING WITH params {}".format([x.data[0] for x in self.policy.parameters()]))

    ret = []
    for _ in range(rollouts):
      timesteps = 0
      for p, dp in zip(self.policy.parameters(), delta):
        p.data += torch.from_numpy(self.std * dp);
      r_pos = black_box(self.policy, self.env)

      for p, dp in zip(self.policy.parameters(), delta):
        p.data -= 2*torch.from_numpy(self.std * dp);
      r_neg = black_box(self.policy, self.env)

      #for p, dp in zip(self.policy.parameters(), delta):
      #  p.data += torch.from_numpy(dp)
      #  pass

      if isinstance(r_pos, tuple):
        timesteps += r_pos[1]
        r_pos = r_pos[0]

      if isinstance(r_neg, tuple):
        timesteps += r_neg[1]
        r_neg = r_neg[0]
      
      ret.append({'delta_idx': idx, 'r_pos': r_pos, 'r_neg': r_neg, 'timesteps': timesteps})
    return ret

class ARS:
  def __init__(self, policy_thunk, env_thunk, step_size=0.02, std=0.0075, deltas=32, workers=4, top_n=None, seed=0):
    self.std = std
    self.num_deltas = deltas
    self.num_workers = workers
    self.step_size = step_size
    self.policy = policy_thunk()

    self.param_shape = [x.shape for x in self.policy.parameters()]

    if top_n is not None:
      self.top_n = top_n
    else:
      self.top_n = deltas

    if not ray.is_initialized():
      ray.init()

    deltas_id  = create_shared_noise.remote(seed=seed, std=std)
    noise = ray.get(deltas_id)

    self.deltas = SharedNoiseTable(noise, self.param_shape, seed=seed+7)
    self.workers = [ARS_process.remote(policy_thunk, env_thunk, deltas_id, std, seed+97+i) for i in range(workers)]
    #self.workers = [ARS_process.remote(policy_thunk, env_thunk, deltas_id, std, 97) for i in range(workers)]

  def step(self, black_box):
    start = time.time()
    pid = ray.put(list(self.policy.parameters())) # place the current policy parameters in shared mem

    rollouts = self.num_deltas // self.num_workers # number of rollouts per worker

    rollout_ids = [w.rollout.remote(pid, black_box, rollouts) for w in self.workers] # do rollouts
    #rollout_ids = [w.rollout.remote(pid, rollouts) for w in self.workers] # do rollouts
    results = ray.get(rollout_ids) # retrieve rollout results from pool

    #print(results)
    results = [item for sublist in results for item in sublist] # flattens list of lists

    r_pos = [item['r_pos'] for item in results]
    r_neg = [item['r_neg'] for item in results]
    delta_indices = [item['delta_idx'] for item in results]
    timesteps = sum([item['timesteps'] for item in results])


    print("\nMEAN TIMESTEPS FOR ROLLOUTS: ", np.mean([item['timesteps'] for item in results]))
    print([item['timesteps'] for item in results])
    print()

    #print("TIME TAKEN TO DO ROLLOUTS AND COLLATE: {}, TIMESTEPS {}".format(time.time() - start, timesteps))
    #print([item['timesteps'] for item in results])
    #input()

    delta = []
    for idx in delta_indices:
      _, d = self.deltas.get_delta(idx)
      #print("tensor at idx {} ended up being {}".format(idx, d.data[5:]))
      delta.append(d)

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
        #print("Adding {} to {}".format((self.step_size * (r_p - r_n) * torch.from_numpy(d_param).data)[:5], param.data[:5]))
        param.data += (self.step_size) / (self.top_n * r_std) * (r_p - r_n) * torch.from_numpy(d_param).data
        #input()
    return timesteps

