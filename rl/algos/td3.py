from rl.utils import AdaptiveParamNoiseSpec, distance_metric, evaluator, perturb_actor_parameters
from rl.policies.td3_actor_critic import LN_Actor as LN_Actor, LN_TD3Critic as Critic

import time
import os

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import gym

import ray

device = torch.device('cpu')


def select_action(perturbed_policy, unperturbed_policy, state, device, param_noise=None):
    state = torch.FloatTensor(state.reshape(1, -1)).to(device)

    unperturbed_policy.eval()

    if param_noise is not None:
        return perturbed_policy(state).cpu().data.numpy().flatten()
    else:
        return unperturbed_policy(state).cpu().data.numpy().flatten()

class ActorBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self.storage = []
        self.max_size = size
        self.ptr = 0
    
    def __len__(self):
        return len(self.storage)

    def storage_size(self):
        return len(self.storage)

    def add(self, data):
        if len(self.storage) < self.max_size:
            self.storage.append(data)
        self.storage[int(self.ptr)] = data
        self.ptr = (self.ptr + 1) % self.max_size
        #print("Added experience to replay buffer.")

    def get_transitions(self):
        ind = np.arange(0, len(self.storage))
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        #print("Sampled experience from replay buffer.")

        self.clear()

        return np.array(x), np.array(u)

    def clear(self):
        self.storage = []
        self.ptr = 0


@ray.remote
class Actor(object):
    def __init__(self, env_fn, learner_id, memory_id, action_dim, start_timesteps, load_freq, taper_load_freq, act_noise, noise_scale, param_noise, id):
        self.env = env_fn()

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])

        #self.policy = LN_Actor(self.state_dim, self.action_dim, self.max_action, 400, 300).to(device)
        self.policy_perturbed = LN_Actor(self.state_dim, self.action_dim, self.max_action, 400, 300).to(device)
        self.learner_id = learner_id
        self.memory_id = memory_id
        
        # Action noise
        self.start_timesteps = start_timesteps
        self.act_noise = act_noise
        
        # Initialize param noise (or set to none)
        self.noise_scale = noise_scale
        self.param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.05, desired_action_stddev=self.noise_scale, adaptation_coefficient=1.05) if param_noise else None

        # Termination condition: max episode length
        self.max_traj_len = 400

        # Counters
        self.actor_timesteps = 0
        self.taper_timesteps = 0
        self.episode_num = 0
        self.taper_load_freq = taper_load_freq  # taper load freq or not?
        self.load_freq = load_freq              # initial load frequency... make this taper down to 1 over time 

        # Local replay buffer
        self.local_buffer = ActorBuffer(self.max_traj_len * self.load_freq)

        self.id = id

        self.policy, self.training_done, self.global_timestep_at_last_update = ray.get(self.learner_id.get_global_policy.remote())


    def collect_experience(self):

        with ray.profile("Actor collection loop", extra_data={'Actor id': str(self.id)}):
            # collection loop -  COLLECTS EPISODES OF EXPERIENCE UNTIL training_done
            while True:

                cassieEnv = True

                if self.actor_timesteps % self.load_freq == 0:
                    # PUTTING WAIT ON THIS SHOULD MAKE THIS EXACT SAME AS NON-DISTRIBUTED, if using one actor
                    # Query learner for latest model and termination flag

                    self.policy, self.training_done, self.global_timestep_at_last_update = ray.get(self.learner_id.get_global_policy.remote())

                    #global_policy_state_dict, training_done = ray.get(self.learner_id.get_global_policy.remote())
                    #self.policy.load_state_dict(global_policy_state_dict)
                    

                    # If we have loaded a global model, we also need to update the param_noise based on the distance metric
                    if self.param_noise is not None:
                        states, perturbed_actions = self.local_buffer.get_transitions()
                        unperturbed_actions = np.array([select_action(self.policy_perturbed, self.policy, state, device, param_noise=None) for state in states])
                        dist = distance_metric(perturbed_actions, unperturbed_actions)
                        self.param_noise.adapt(dist)
                        print("loaded global model and adapted parameter noise")
                    else:
                        print("loaded global model")

                if self.training_done:
                    break

                obs = self.env.reset()
                done = False
                episode_reward = 0
                episode_timesteps = 0

                # nested collection loop - COLLECTS TIMESTEPS OF EXPERIENCE UNTIL episode is over
                while episode_timesteps < self.max_traj_len and not done:

                    #self.env.render()

                    # Param Noise
                    if self.param_noise:
                        perturb_actor_parameters(self.policy_perturbed, self.policy, self.param_noise, device)

                    # Select action randomly or according to policy
                    if self.actor_timesteps < self.start_timesteps:
                        #print("selecting action randomly {}".format(done_bool))
                        action = torch.randn(self.env.action_space.shape[0]) if cassieEnv is True else self.env.action_space.sample()
                        action = action.numpy()
                    else:
                        #print("selecting from policy")
                        action = select_action(self.policy_perturbed, self.policy, np.array(obs), device, param_noise=self.param_noise)
                        if self.act_noise != 0:
                            action = (action + np.random.normal(0, self.act_noise,
                                                                size=self.env.action_space.shape[0])).clip(self.env.action_space.low, self.env.action_space.high)

                    # Perform action
                    new_obs, reward, done, _ = self.env.step(action)
                    done_bool = 1.0 if episode_timesteps + 1 == self.max_traj_len else float(done)
                    episode_reward += reward

                    # Store data in replay buffer
                    transition = (obs, new_obs, action, reward, done_bool)
                    self.local_buffer.add(transition)
                    self.memory_id.add.remote(transition)

                    # call update from model server
                    self.learner_id.update_and_evaluate.remote()

                    # update state
                    obs = new_obs

                    # increment step counts
                    episode_timesteps += 1
                    self.actor_timesteps += 1

                    # increment global step count
                    self.learner_id.increment_step_count.remote()

                # episode is over, increment episode count and plot episode info
                self.episode_num += 1
                
                # pass episode details to visdom logger on memory server
                self.memory_id.plot_actor_results.remote(self.id, self.actor_timesteps, episode_reward)

                ray.wait([self.learner_id.increment_episode_count.remote()], num_returns=1)

                if self.taper_load_freq and self.taper_timesteps >= 2000:
                    self.load_freq = self.load_freq // 2
                    print("Increased load frequency")

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

        # highest reward tracker
        self.highest_return = 0

        # experience replay
        self.memory = memory_server

        # env attributes
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])
        self.max_traj_len = 400

        # models and optimizers
        self.actor = LN_Actor(self.state_dim, self.action_dim, self.max_action, 400, 300).to(self.device)
        self.actor_target = LN_Actor(self.state_dim, self.action_dim, self.max_action, 400, 300).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(self.state_dim, self.action_dim, 400, 300).to(self.device)
        self.critic_target = Critic(self.state_dim, self.action_dim, 400, 300).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        # visdom plotter
        # self.plotter_id = plotter_id

        # evaluate untrained policy
        # print('Untrained Policy: {}'.format( self.evaluate(trials=self.num_of_evaluators, num_of_workers=self.num_of_evaluators) ))

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

            # save policy with highest return so far
            if self.highest_return < self.results[-1]:
                self.highest_return = self.results[-1]

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
        return self.actor, self.is_training_finished(), self.step_count

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