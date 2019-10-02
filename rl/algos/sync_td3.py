import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from rl.utils.remote_replay import ReplayBuffer
from rl.policies.td3_actor_critic import Original_Actor as O_Actor, TD3Critic as Critic

import ray

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

# TODO: Make each worker collect fixed amount of experience / don't stop computation once episodes are done
def parallel_collect_experience(policy, env_fn, act_noise, min_steps, max_traj_len, num_procs=4):

    all_transitions = ray.get([collect_experience.remote(env_fn, policy, min_steps, max_traj_len, act_noise) for i in range(num_procs)])

    merged_transitions = np.concatenate(all_transitions)
    return merged_transitions, len(merged_transitions)

# sample experience for one episode and send to replay buffer
@ray.remote
def collect_experience(env_fn, policy, min_steps, max_traj_len, act_noise):

    env = env_fn()

    local_buffer = ReplayBuffer(max_size=max_traj_len)

    num_steps = 0
    # nested collection loop - collect experience until episode is over
    while num_steps < min_steps:
        
        # reset environment
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_timesteps = 0

        while not done and episode_timesteps < max_traj_len:

            # select action from policy
            action = policy.select_action(obs)
            if act_noise != 0:
                action = (action + np.random.normal(0, act_noise, size=1)).clip(-1, 1)

            # Perform action
            new_obs, reward, done, _ = env.step(action)
            done_bool = 1.0 if episode_timesteps + 1 == max_traj_len else float(done)
            episode_reward += reward

            # Store data in replay buffer
            transition = (obs, new_obs, action, reward, done_bool)
            local_buffer.add(transition)

            # update state
            obs = new_obs

    # episode is over, return all transitions from this episode (list of tuples)
    return local_buffer.get_all_transitions()

class TD3():
    def __init__(self, state_dim, action_dim, max_action, a_lr, c_lr):
        self.actor = O_Actor(state_dim, action_dim, max_action, 256, 256).to(device)
        self.actor_target = O_Actor(state_dim, action_dim, max_action, 256, 256).to(device)
        self.actor_perturbed = O_Actor(state_dim, action_dim, max_action, 256, 256).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=a_lr)

        self.critic = Critic(state_dim, action_dim, 256, 256).to(device)
        self.critic_target = Critic(state_dim, action_dim, 256, 256).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=c_lr)

        self.max_action = max_action

    def perturb_actor_parameters(self, param_noise):
        """Apply parameter noise to actor model, for exploration"""
        self.actor_perturbed.load_state_dict(self.actor.state_dict())
        params = self.actor_perturbed.state_dict()
        for name in params:
            if 'ln' in name: 
                pass 
            param = params[name]
            param += torch.randn(param.shape).to(device) * param_noise.current_stddev

    def select_action(self, state, param_noise=None):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)

        self.actor.eval()

        if param_noise is not None:
            return self.actor_perturbed(state).cpu().data.numpy().flatten()
        else:
            return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):

        avg_q1, avg_q2, avg_targ_q, q_loss, pi_loss, avg_noise, avg_action = (0,0,0,0,0,0,0)

        for it in range(iterations):

            # Sample replay buffer
            x, y, u, r, d = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1 - d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Select action according to policy and add clipped noise
            noise = torch.FloatTensor(u).data.normal_(
                0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) +
                           noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)
            
            # Keep track of Q estimates for logging
            avg_q1 += current_Q1
            avg_q2 += current_Q2
            avg_targ_q += target_Q
            avg_action += next_action

            # Compute critic loss
            critic_loss = F.mse_loss(
                current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            
            # Keep track of Q loss for logging
            q_loss += critic_loss

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % policy_freq == 0:

                # Compute actor loss
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

                # Keep track of pi loss for logging
                pi_loss += actor_loss

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data)

        # prep info for logging
        avg_q1 /= iterations
        avg_q2 /= iterations
        avg_targ_q /= iterations
        q_loss /= iterations
        pi_loss /= iterations
        avg_action /= iterations

        return torch.mean(avg_q1), torch.mean(avg_q1), torch.mean(avg_targ_q), q_loss, pi_loss, avg_action

    def save(self):
        if not os.path.exists('trained_models/syncTD3/'):
            os.makedirs('trained_models/syncTD3/')

        print("Saving model")

        filetype = ".pt"  # pytorch model
        torch.save(self.actor.state_dict(), os.path.join(
            "./trained_models/syncTD3", "actor_model" + filetype))
        torch.save(self.critic.state_dict(), os.path.join(
            "./trained_models/syncTD3", "critic_model" + filetype))

    def load(self, model_path):
        actor_path = os.path.join(model_path, "actor_model.pt")
        critic_path = os.path.join(model_path, "critic_model.pt")
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
            self.actor.eval()
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
            self.critic.eval()