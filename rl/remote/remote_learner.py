from rl.policies.td3_actor_critic import LN_Actor as Actor, LN_TD3Critic as Critic
from rl.remote.remote_evaluator import *

import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import time

import gym

import ray

@ray.remote(num_gpus=0)
class Learner(object):
    def __init__(self, env_fn, memory_server, learning_episodes, state_space, action_space,
                 batch_size=500, discount=0.99, tau=0.005, eval_update_freq=10,
                 target_update_freq=2000, evaluate_freq=50, num_of_evaluators=30):

        # THIS WORKS, but rest doesn't when trying to use GPU for learner
        print("This function is allowed to use GPUs {}.".format(ray.get_gpu_ids()))

        self.device = torch.device('cpu')

        # keep uninstantiated constructor for evaluator
        self.env_fn = env_fn

        self.env = env_fn()
        self.learning_episodes = learning_episodes
        #self.max_timesteps = max_timesteps

        #self.start_memory_size=1000
        self.batch_size=batch_size

        self.eval_update_freq = eval_update_freq
        self.target_update_freq = target_update_freq
        self.evaluate_freq = evaluate_freq     # how many steps before each eval

        self.num_of_evaluators = num_of_evaluators

        # counters
        self.step_count = 0
        self.eval_step_count = 0
        self.target_step_count = 0

        self.episode_count = 0
        self.eval_episode_count = 0

        self.update_counter = 0

        # hyperparams
        self.discount = discount
        self.tau = tau
        
        # results list
        self.results = []

        # experience replay
        self.memory = memory_server

        # env attributes
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])
        self.max_traj_len = 400

        # models and optimizers
        self.actor = Actor(self.state_dim, self.action_dim, self.max_action, 400, 300).to(self.device)
        self.actor_target = Actor(self.state_dim, self.action_dim, self.max_action, 400, 300).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(self.state_dim, self.action_dim, 400, 300).to(self.device)
        self.critic_target = Critic(self.state_dim, self.action_dim, 400, 300).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        # visdom plotter
        # self.plotter_id = plotter_id

        # evaluate untrained policy
        print('Untrained Policy: {}'.format( self.evaluate(trials=self.num_of_evaluators, num_of_workers=self.num_of_evaluators) ))

        # also dump ray timeline
        # ray.timeline(filename="./ray_timeline.json")

        self.update_and_evaluate()

    def increment_step_count(self):
        self.step_count += 1        # global step count

        # increment models' step counts
        self.eval_step_count += 1     # step count between calls of updating policy and targets (TD3)
        self.target_step_count += 1   # time between each eval

    def update_and_evaluate(self):

        # update eval model every 'eval_update_freq'
        if self.eval_step_count >= self.eval_update_freq:
            # reset step count
            self.eval_step_count = 0

            # update model
            self.update_eval_model()

    def increment_episode_count(self):
        self.episode_count += 1
        self.eval_episode_count += 1
        #print(self.episode_count)

        if self.eval_episode_count >= self.evaluate_freq:

            self.eval_episode_count = 0

            # evaluate learned policy
            self.results.append(self.evaluate(trials=self.num_of_evaluators, num_of_workers=self.num_of_evaluators))
            print('Episode {}: {}'.format(self.episode_count, self.results[-1]))

            # also save
            self.save()

            # also dump ray timelines
            # ray.timeline(filename="./ray_timeline.json")
            # ray.object_transfer_timeline(filename="./ray_object_transfer_timeline.json")

    def is_training_finished(self):
        return self.episode_count >= self.learning_episodes

    def update_eval_model(self, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        with ray.profile("Learner optimization loop", extra_data={'Episode count': str(self.episode_count)}):

            start_time = time.time()

            if ray.get(self.memory.storage_size.remote()) < self.batch_size:
                print("not enough experience yet")
                return

            # randomly sample a mini-batch transition from memory_server
            x, y, u, r, d = ray.get(self.memory.sample.remote(self.batch_size))
            state = torch.FloatTensor(x).to(self.device)
            action = torch.FloatTensor(u).to(self.device)
            next_state = torch.FloatTensor(y).to(self.device)
            done = torch.FloatTensor(1 - d).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)

            # Select action according to policy and add clipped noise
            noise = torch.FloatTensor(u).data.normal_(
                0, policy_noise).to(self.device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) +
                           noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * self.discount * target_Q).detach()

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(
                current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            self.update_counter += 1

            # Delayed policy updates
            if self.update_counter % policy_freq == 0:

                print("optimizing at timestep {} | time = {} | replay size = {} | episode count = {} | update count = {} ".format(self.step_count, time.time()-start_time, ray.get(self.memory.storage_size.remote()), self.episode_count, self.update_counter))

                # Compute actor loss
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data)

    def evaluate(self, trials=30, num_of_workers=30):

        start_time = time.time()

        # initialize evaluators
        evaluators = [evaluator.remote(self.env_fn, self.actor, max_traj_len=400)
                      for _ in range(num_of_workers)]

        total_rewards = 0

        for t in range(trials):
            # get result from a worker
            ready_ids, _ = ray.wait(evaluators, num_returns=1)

            # update total rewards
            total_rewards += ray.get(ready_ids[0])

            # remove ready_ids from the evaluators
            evaluators.remove(ready_ids[0])

            # start a new worker
            evaluators.append(evaluator.remote(self.env_fn, self.actor, self.max_traj_len))

        # return average reward
        avg_reward = total_rewards / trials
        self.memory.plot_learner_results.remote(self.step_count, avg_reward)

        print("eval time: {}".format(time.time()-start_time))

        return avg_reward

    def test(self):
        return 0

    def get_global_policy(self):
        return self.actor, self.is_training_finished()

    def get_global_timesteps(self):
        return self.step_count

    def get_results(self):
        return self.results, self.evaluate_freq

    def save(self):
        if not os.path.exists('trained_models/apex/'):
            os.makedirs('trained_models/apex/')

        print("Saving model")

        filetype = ".pt"  # pytorch model
        torch.save(self.actor.state_dict(), os.path.join(
            "./trained_models/apex", "global_policy" + filetype))
        torch.save(self.critic.state_dict(), os.path.join(
            "./trained_models/apex", "critic_model" + filetype))

    def load(self, model_path):
        actor_path = os.path.join(model_path, "global_policy.pt")
        critic_path = os.path.join(model_path, "critic_model.pt")
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
            self.actor.eval()
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
            self.critic.eval()