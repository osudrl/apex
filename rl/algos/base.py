from torch.autograd import Variable
import torch

import numpy as np

class Rollout():
    def __init__(self, num_steps, obs_dim, action_dim, first_state):
        self.states = torch.zeros(num_steps + 1, obs_dim)
        self.states[0] = first_state

        self.actions = torch.zeros(num_steps, action_dim)
        self.rewards = torch.zeros(num_steps, 1)
        self.values = torch.zeros(num_steps + 1, 1)
        self.returns = torch.zeros(num_steps + 1, 1)
        self.masks = torch.ones(num_steps + 1, 1)

    def insert(self, step, state, action, value, reward, mask):
        self.states[step + 1] = state # why?
        self.actions[step] = action
        self.values[step] = value
        self.rewards[step] = reward
        self.masks[step] = mask
    
    def calculate_returns(self, next_value, gamma, tau):
        self.values[-1] = next_value
        gae = 0
        for step in reversed(range(self.rewards.size(0))):
            delta = self.rewards[step] + gamma * self.values[step + 1] * self.masks[step] - self.values[step]
            gae = delta + gamma * tau * gae * self.masks[step]
            self.returns[step] = gae + self.values[step]


class PolicyGradientAlgorithm():
    def train():
        raise NotImplementedError

    @staticmethod
    def add_arguments(parser):
        raise NotImplementedError

    def sample_steps(self, env, policy, num_steps):
        """Collect a set number of frames, as in the original paper."""
        rewards = []
        episode_reward = 0

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        state = env.reset()
        state = torch.Tensor(state)

        rollout = Rollout(num_steps, obs_dim, action_dim, state)
        for step in range(num_steps):
            value, action = policy.act(Variable(state))

            state, reward, done, _ = env.step(action.data.numpy())

            episode_reward += reward
            if done:
                state = env.reset()
                rewards.append(episode_reward)
                episode_reward = 0

            state = torch.Tensor(state)
            reward = torch.Tensor([reward])

            mask = torch.Tensor([0.0 if done else 1.0])
            #final_reward *= mask
            #final_reward += (1 - masks) * episode_rewards
            #episode_reward *= mask

            rollout.insert(step, state, action.data, value.data, reward, mask)
        
        if hasattr(self.policy, 'obs_filter'):
            self.policy.obs_filter.update(rollout.states[:-1])

        next_value, _ = self.policy(Variable(state))
        rollout.calculate_returns(next_value.data, .99, .95)
        
        return (rollout.states[:-1], 
               rollout.actions, 
               rollout.returns[:-1], 
               rollout.values,
               sum(rewards)/len(rewards))

    def rollout(self, env, policy, num_frames, critic_target="td_lambda"):
        """Collect a single rollout."""

        observations = []
        actions = []
        rewards = []
        returns = []
        advantages = []
        values = []

        obs = env.reset().ravel()[None, :]
        obs = torch.Tensor(obs)

        for _ in range(num_frames):
            value, action = policy.act(Variable(obs))

            next_obs, reward, done, _ = env.step(action.data.numpy().ravel())

            observations.append(obs)
            actions.append(action.data)

            rewards.append(torch.Tensor([[reward]]))

            obs = next_obs.ravel()[None, :]
            obs = torch.Tensor(obs)

            values.append(value.data)

            if done:
                break

        values.append(policy.act(Variable(obs))[0].data)

        R = torch.zeros(1, 1)
        advantage = torch.zeros(1, 1)
        for t in reversed(range(len(rewards))):
            R = self.discount * R + rewards[t]

            # generalized advantage estimation
            # see: https://arxiv.org/abs/1506.02438
            delta = rewards[t] + self.discount * values[t + 1] - values[t]
            gae = delta + gae * self.discount * self.tau

            #returns.append(R)
            returns.append(gae + values[t])
        """
        # GAE paper, footnote 2
        if critic_target == "td_lambda":
            returns = torch.cat(advantages[::-1]) + torch.cat(values[:-1])

        # GAE paper, equation 28
        elif critic_target == "td_one":
            returns = torch.cat(returns[::-1])
        """
        return dict(
            rewards=torch.cat(rewards),
            returns=torch.cat(returns[::-1]),
            advantages=torch.cat(advantages[::-1]),
            observations=torch.cat(observations),
            actions=torch.cat(actions),
        )
