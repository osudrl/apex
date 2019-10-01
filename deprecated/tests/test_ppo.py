from collections import defaultdict

import pytest
import numpy as np

import torch

from rl.policies import GaussianMLP
from rl.algos import PPO

class SampleTesterEnv:
    def __init__(self, obs_dim, action_dim, done_state=10, gamma=0.99):
        """
        A simple environment that unit tests whether or the 
        experience buffer and trajectory sampling code are 
        producing the correct output. This is to test for things
        like off-by-one errors in the experience buffers or 
        reward-to-go calculations.

        In other words:

        Makes sure the "experience table" is of the form:

        --------------------
        s_0   |  a_0  |  r_0
        --------------------
        .           .      .
        .         .        .
        .       .          .
        --------------------
        s_T   |  a_T   | r_T
        --------------------
        s_T+1 |        |
        --------------------

        where entries are defined by the MDP transitions:

        s_0 -> (s_1, a_0, r_0) -> ... -> (s_T+1, a_T, r_T)

        """

        self.observation_space = np.zeros(obs_dim)
        self.action_space = np.zeros(action_dim)

        self.state = 0
        self.obs_dim = obs_dim

        # TODO: add GAE support?
        self.gamma = gamma

        self.done = done_state

        self.actions = []
    
    def step(self, action):
        self.state += 1

        output = np.ones(shape=(1, self.obs_dim)) * self.state

        done = (self.state % self.done) == 0

        #print("real: ", done)

        # the first reward corresponds to the second state
        reward = np.ones(shape=(1, 1)) * (self.state - 1)

        self.actions.append(action.squeeze(0)) # TODO

        return output, reward, done, None

    def reset(self):
        self.state = 0

        output = np.ones(shape=(1, self.obs_dim)) * self.state
        return output


@pytest.mark.parametrize("num_steps, obs_dim, action_dim", [
    (5, 1, 1),
    (10, 1, 1),
    (25, 1, 1),
    (10, 80, 10),
    (30, 80, 10),
    (35, 80, 10)
])
def test_ppo_sample(num_steps, obs_dim, action_dim):
    # useful for debugging
    np.set_printoptions(threshold=10000)
    torch.set_printoptions(threshold=10000)

    # TODO: test value est bootstrap for truncated trajectories

    gamma = 0.99

    env = SampleTesterEnv(obs_dim=obs_dim, action_dim=action_dim, gamma=gamma)
    policy = GaussianMLP(obs_dim, action_dim)

    # don't need to specify args that don't affect ppo.sample()
    args = defaultdict(lambda: None, {'gamma': gamma})

    algo = PPO(args)

    memory = algo.sample(env, policy, num_steps, 100)

    states, actions, rewards, returns = map(torch.Tensor,
        (memory.states, memory.actions, memory.rewards, memory.returns)
    )

    num_steps = states.shape[0] 

    assert states.shape == (num_steps, obs_dim)
    assert actions.shape == (num_steps, action_dim)
    assert rewards.shape == (num_steps, 1)
    assert rewards.shape == (num_steps, 1)

    expected_states = np.array([(np.ones(shape=(obs_dim,)) * (s % env.done)) for s in range(num_steps)])
    assert np.allclose(states, expected_states)

    expected_rewards = np.array([(np.ones(shape=(1)) * (s % env.done)) for s in range(num_steps)])
    assert np.allclose(rewards, expected_rewards)

    expected_actions = np.array(env.actions)
    assert np.allclose(actions, expected_actions)

    expected_returns, R = [], 0
    for r in reversed(expected_rewards):
        R = R * gamma + r

        expected_returns.insert(0, R.copy())

        if r == 0: # this only happens on initial state, so restart the return
            R = 0

    expected_returns = np.array(expected_returns)
    assert np.allclose(returns, expected_returns)


@pytest.mark.parametrize("num_steps, obs_dim, action_dim", [
    (5, 1, 1),
    (10, 1, 1),
    (25, 1, 1),
    (10, 80, 10),
    (30, 80, 10),
    (35, 80, 10)
])
def test_ppo_sample_parallel(num_steps, obs_dim, action_dim):
    # useful for debugging
    np.set_printoptions(threshold=10000)
    torch.set_printoptions(threshold=10000)

    # TODO: test value est bootstrap for truncated trajectories

    gamma = 0.99

    from functools import partial

    env = SampleTesterEnv(obs_dim=obs_dim, action_dim=action_dim, gamma=gamma)
    env_fn = partial(
        SampleTesterEnv, 
        obs_dim=obs_dim, 
        action_dim=action_dim, 
        gamma=gamma
    )

    policy = GaussianMLP(obs_dim, action_dim)

    # don't need to specify args that don't affect ppo.sample()
    args = defaultdict(lambda: None, {'gamma': gamma, 'num_procs': 4})

    algo = PPO(args)

    memory = algo.sample_parallel(env_fn, policy, num_steps, 100)

    expected_memory = algo.sample(env, policy, 40, 100)

    #breakpoint()

    assert np.allclose(memory.states, expected_memory.states)
    #assert np.allclose(memory.actions, expected_memory.actions)
    assert np.allclose(memory.rewards, expected_memory.rewards)
    #assert np.allclose(memory.returns, expected_memory.returns)
    assert np.allclose(memory.returns, expected_memory.returns)

    assert np.allclose(memory.ep_returns, expected_memory.ep_returns)
    assert np.allclose(memory.ep_lens, expected_memory.ep_lens)

test_ppo_sample_parallel(5, 1, 1)