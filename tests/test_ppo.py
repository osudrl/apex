from collections import defaultdict

import pytest
import numpy as np

import torch

from rl.policies import GaussianMLP
from rl.algos import PPO

class SampleTesterEnv:
    def __init__(self, obs_dim=80, action_dim=10, done_state=10, gamma=0.99):
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

        # the first reward corresponds to the second state
        
        reward = np.ones(shape=(1, 1)) * (self.state - 1)

        self.actions.append(action.squeeze(0)) # TODO

        return output, reward, done, None

    def reset(self):
        self.state = 0

        output = np.ones(shape=(1, self.obs_dim)) * self.state
        return output

    def test(self, states, actions, rewards, returns):
        # TODO: add off-by-one checks to diagnose common errors?

        num_steps = states.shape[0]

        expected_states = np.array([(np.ones(shape=(self.obs_dim,)) * (s % self.done)) for s in range(num_steps)])

        assert np.allclose(states, expected_states)

        expected_rewards = np.array([(np.ones(shape=(1)) * (s % self.done)) for s in range(num_steps)])

        assert np.allclose(rewards, expected_rewards)

        expected_actions = np.array(self.actions)

        assert np.allclose(actions, expected_actions)

        expected_returns, R = [], 0
        for r in reversed(expected_rewards):
            R = R * self.gamma + r

            expected_returns.insert(0, R.copy())

            if r == 0: # this is an initial state, so reset the return
                R = 0

        expected_returns = np.array(expected_returns)

        # compare to rllab magic reward-to-go equation
        # import scipy.signal
        # assert np.allclose(expected_returns, scipy.signal.lfilter([1], [1, float(-self.gamma)], expected_rewards[::-1], axis=0)[::-1])

        assert np.allclose(returns, expected_returns)


@pytest.mark.parametrize("num_steps", [
    10,
    30,
    35
])
def test_ppo_sample(num_steps):
    # useful for debugging
    np.set_printoptions(threshold=10000)
    torch.set_printoptions(threshold=10000)

    obs_dim = 80
    action_dim = 10

    gamma = 0.99

    env = SampleTesterEnv(obs_dim=obs_dim, action_dim=action_dim, gamma=gamma)
    policy = GaussianMLP(obs_dim, action_dim)

    # don't need to specify args that don't affect sample()
    args = defaultdict(lambda: None, {'gamma': gamma})

    algo = PPO(args)

    memory, _ = algo.sample_steps(env, policy, num_steps)

    # TODO: move this out of env?
    env.test(memory.states[:-1], memory.actions, memory.rewards, memory.returns[:-1])