import torch
import hashlib, os, pickle
import sys, time
from cassie.quaternion_function import *
import tty
import termios
import select
import numpy as np
from functools import partial
from rl.envs import WrapEnv
from rl.envs.wrappers import SymmetricEnv
from cassie import CassieEnv
import ray


@ray.remote
class sample_worker(object):
    def __init__(self, id_num, env_fn, gamma, lam):
        self.id_num = id_num
        self.env = WrapEnv(env_fn)
        self.gamma = gamma
        self.lam = lam
        torch.set_num_threads(1)

    @torch.no_grad()
    def sample(self, policy, critic, min_steps, max_traj_len, deterministic=False):
        """
        Sample at least min_steps number of total timesteps, truncating 
        trajectories only if they exceed max_traj_len number of timesteps
        """
        # torch.set_num_threads(1) # By default, PyTorch will use multiple cores to speed up operations.
                                    # This can cause issues when Ray also uses multiple cores, especially on machines
                                    # with a lot of CPUs. I observed a significant speedup when limiting PyTorch 
                                    # to a single core - I think it basically stopped ray workers from stepping on each
                                    # other's toes.

        memory = PPOBuffer(self.gamma, self.lam)

        num_steps = 0
        while num_steps < min_steps:
            state = torch.Tensor(self.env.reset())

            done = False
            value = 0
            traj_len = 0

            while not done and traj_len < max_traj_len:
                action = policy(state, deterministic=False)
                value = critic(state)

                next_state, reward, done, _ = self.env.step(action.numpy())
                memory.store(state.numpy(), action.numpy(), reward, value.numpy())

                state = torch.Tensor(next_state)

                traj_len += 1
                num_steps += 1

            value = critic(state)
            memory.finish_path(last_val=(not done) * value.numpy())

        return memory

    def sample_single(self, policy, critic, max_traj_len, deterministic=False):
        memory = PPOBuffer(self.gamma, self.lam)

        state = torch.Tensor(self.env.reset())

        done = False
        value = 0
        traj_len = 0

        while not done and traj_len < max_traj_len:
            action = policy(state, deterministic=False)
            value = critic(state)

            next_state, reward, done, _ = self.env.step(action.numpy())
            memory.store(state.numpy(), action.numpy(), reward, value.numpy())

            state = torch.Tensor(next_state)

            traj_len += 1
            num_steps += 1

        value = critic(state)
        memory.finish_path(last_val=(not done) * value.numpy())

        return self.id_num, memory


class PPOBuffer:
    """
    A buffer for storing trajectory data and calculating returns for the policy
    and critic updates.
    This container is intentionally not optimized w.r.t. to memory allocation
    speed because such allocation is almost never a bottleneck for policy 
    gradient. 
    
    On the other hand, experience buffers are a frequent source of
    off-by-one errors and other bugs in policy gradient implementations, so
    this code is optimized for clarity and readability, at the expense of being
    (very) marginally slower than some other implementations. 
    (Premature optimization is the root of all evil).
    """
    def __init__(self, gamma=0.99, lam=0.95, use_gae=False):
        self.states  = []
        self.actions = []
        self.rewards = []
        self.values  = []
        self.returns = []

        self.ep_returns = [] # for logging
        self.ep_lens    = []

        self.gamma, self.lam = gamma, lam

        self.ptr = 0
        self.traj_idx = [0]
    
    def __len__(self):
        return len(self.states)

    def storage_size(self):
        return len(self.states)

    def store(self, state, action, reward, value):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        # TODO: make sure these dimensions really make sense
        self.states  += [state.squeeze(0)]
        self.actions += [action.squeeze(0)]
        self.rewards += [reward.squeeze(0)]
        self.values  += [value.squeeze(0)]

        self.ptr += 1
    
    def finish_path(self, last_val=None):
        self.traj_idx += [self.ptr]
        rewards = self.rewards[self.traj_idx[-2]:self.traj_idx[-1]]

        returns = []

        R = last_val.squeeze(0).copy() # Avoid copy?
        for reward in reversed(rewards):
            R = self.gamma * R + reward
            returns.insert(0, R) # TODO: self.returns.insert(self.path_idx, R) ? 
                                 # also technically O(k^2), may be worth just reversing list
                                 # BUG? This is adding copies of R by reference (?)

        self.returns += returns

        self.ep_returns += [np.sum(rewards)]
        self.ep_lens    += [len(rewards)]

    def get(self):
        return(
            self.states,
            self.actions,
            self.returns,
            self.values
        )

@ray.remote
@torch.no_grad()
def sample(env_fn, policy, critic, min_steps, max_traj_len, deterministic=False):
    """
    Sample at least min_steps number of total timesteps, truncating 
    trajectories only if they exceed max_traj_len number of timesteps
    """
    torch.set_num_threads(1) # By default, PyTorch will use multiple cores to speed up operations.
                                # This can cause issues when Ray also uses multiple cores, especially on machines
                                # with a lot of CPUs. I observed a significant speedup when limiting PyTorch 
                                # to a single core - I think it basically stopped ray workers from stepping on each
                                # other's toes.

    env = WrapEnv(env_fn) # TODO

    memory = PPOBuffer(.99, .95)

    num_steps = 0
    while num_steps < min_steps:
        state = torch.Tensor(env.reset())

        done = False
        value = 0
        traj_len = 0

        while not done and traj_len < max_traj_len:
            action = policy(state, deterministic=False)
            value = critic(state)

            next_state, reward, done, _ = env.step(action.numpy())
            memory.store(state.numpy(), action.numpy(), reward, value.numpy())

            state = torch.Tensor(next_state)

            traj_len += 1
            num_steps += 1

        value = critic(state)
        memory.finish_path(last_val=(not done) * value.numpy())

    return memory

def sample_parallel(env_fn, policy, critic, min_steps, max_traj_len, n_proc, deterministic=False):
    args = (env_fn, policy, critic, min_steps, max_traj_len, deterministic)

    real_proc = n_proc
    # if self.limit_cores:
    #     real_proc = 48 - 16*int(np.log2(60 / env_fn().simrate))
    #     print("limit cores active, using {} cores".format(real_proc))
    #     args = (self, env_fn, policy, critic, min_steps*self.n_proc // real_proc, max_traj_len, deterministic)
    result_ids = [sample.remote(*args) for _ in range(real_proc)]
    result = ray.get(result_ids)
    
    # O(n)
    def merge(buffers):
        merged = PPOBuffer(.99, .95)
        for buf in buffers:
            offset = len(merged)

            merged.states  += buf.states
            merged.actions += buf.actions
            merged.rewards += buf.rewards
            merged.values  += buf.values
            merged.returns += buf.returns

            merged.ep_returns += buf.ep_returns
            merged.ep_lens    += buf.ep_lens

            merged.traj_idx += [offset + i for i in buf.traj_idx[1:]]
            merged.ptr += buf.ptr

        return merged

    total_buf = merge(result)
    # if len(total_buf) > min_steps*self.n_proc * 1.5:
    #     self.limit_cores = 1
    return total_buf

def test_sample_parallel(env_fn, policy, critic, num_steps, max_traj_len, n_procs, ntrials):
    ray.init(num_cpus=n_procs)
    avg_time = 0 
    for i in range(ntrials):
        start_t = time.time()
        data = sample_parallel(env_fn, policy, critic, num_steps, 300, n_procs)
        curr_time = time.time() - start_t
        print("num samples: {}\ttotal time: {}".format(len(data), time.time() - start_t))
        if i != 0:      # Skip first iter in avg time for consistency
            avg_time += curr_time
    avg_time /= ntrials - 1
    print("avg time: ", avg_time)
    ray.shutdown()

def test_sample_parallel2(env_fn, policy, critic, min_steps, max_traj_len, n_proc, ntrials):
    ray.init(num_cpus=n_procs)
    # Make workers
    workers = [sample_worker.remote(i, env_fn, 0.99, 0.95) for i in range(n_proc)]
    avg_time = 0
    
    # TODO: Parallel merge? each worker merges with adjacent, and so on
    def merge(buffers):
        merged = PPOBuffer(.99, .95)
        for buf in buffers:
            offset = len(merged)

            merged.states  += buf.states
            merged.actions += buf.actions
            merged.rewards += buf.rewards
            merged.values  += buf.values
            merged.returns += buf.returns

            merged.ep_returns += buf.ep_returns
            merged.ep_lens    += buf.ep_lens

            merged.traj_idx += [offset + i for i in buf.traj_idx[1:]]
            merged.ptr += buf.ptr

        return merged

    for i in range(ntrials):
        start_t = time.time()
        result_ids = [workers[i].sample.remote(policy, critic, min_steps, max_traj_len, False) for i in range(n_proc)]
        result = ray.get(result_ids)
        total_buf = merge(result)
        curr_time = time.time() - start_t
        print("num samples: {}\ttotal time: {}".format(len(total_buf), time.time() - start_t))
        if i != 0:      # Skip first iter in avg time for consistency
            avg_time += curr_time
    avg_time /= ntrials - 1
    print("avg time: ", avg_time)
    ray.shutdown()
    
def test_sample_parallel3(env_fn, policy, critic, min_steps, max_traj_len, n_proc, ntrials):
    # Might need to run task in background thread so can kill leftover sampling actor tasks. 
    # See https://github.com/ray-project/ray/issues/854 and https://docs.ray.io/en/ray-0.4.0/actors.html#current-actor-limitations
    ray.init(num_cpus=n_procs)
    # Make workers
    workers = [sample_worker.remote(i, env_fn, 0.99, 0.95) for i in range(n_proc)]
    avg_time = 0

    # TODO: Parallel merge? each worker merges with adjacent, and so on
    def merge(buffers):
        merged = PPOBuffer(.99, .95)
        for buf in buffers:
            offset = len(merged)

            merged.states  += buf.states
            merged.actions += buf.actions
            merged.rewards += buf.rewards
            merged.values  += buf.values
            merged.returns += buf.returns

            merged.ep_returns += buf.ep_returns
            merged.ep_lens    += buf.ep_lens

            merged.traj_idx += [offset + i for i in buf.traj_idx[1:]]
            merged.ptr += buf.ptr

        return merged

    for i in range(ntrials):
        start_t = time.time()
        total_data = PPOBuffer(.99, .95)
        # Start workers
        result_ids = [workers[i].sample_single.remote(policy, critic, max_traj_len, False) for i in range(n_proc)]
        while len(total_data) < min_steps*n_proc:
            done_id = ray.wait(result_ids, num_returns=1, timeout=None)[0][0]
            worker_id, data = ray.get(done_id)
            # Start
            print(done_id)
            exit()

def test_sample_parallel4():
    # Same as test parallel3, but have separate actor that holds memory buffer and will do the merging while
    # other actors are collecting. Might save time by not having the merging be blocking, but constant
    # serializing and passing of data might be slow
    
    return 0

def sample_traj(cassie_env, policy, traj_len):
    state = cassie_env.reset_for_test()
    for i in range(traj_len):
        cassie_env.speed = 0
        cassie_env.orient_add = 0
        with torch.no_grad():
            action = policy.forward(torch.Tensor(state), deterministic=True).detach().numpy()
        state, reward, done, _ = cassie_env.step(action)
        if cassie_env.sim.qpos()[2] < 0.4:  # Failed, reset and record force
            pass


env_fn = partial(CassieEnv, state_est=True, dynamics_randomization=False, history=0)
path = "./trained_models/nodelta_neutral_StateEst_symmetry_speed0-3_freq1-2/"
policy = torch.load(path + "actor.pt")
critic = torch.load(path + "critic.pt")
# policy.eval()
cassie_env = env_fn()
start_t = time.time()
# sample_traj(cassie_env, policy, 2400)
# print("eval time: ", time.time() - start_t)
# exit()

n_procs = 6
num_steps = 4000 // n_procs
num_trials = 4


time.sleep(1)

test_sample_parallel(env_fn, policy, critic, num_steps, 300, n_procs, num_trials)
# exit()

test_sample_parallel2(env_fn, policy, critic, num_steps, 300, n_procs, num_trials)
# test_sample_parallel(env_fn, policy, critic, num_steps, 300, n_procs, num_trials)


