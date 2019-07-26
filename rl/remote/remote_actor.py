from rl.policies.td3_actor_critic import LN_Actor as LN_Actor
from rl.utils import AdaptiveParamNoiseSpec, VisdomLinePlotter

import numpy as np
import torch

import ray

device = torch.device("cpu")

def select_action(Policy, state, device):
    state = torch.FloatTensor(state.reshape(1, -1)).to(device)

    Policy.eval()

    return Policy(state).cpu().data.numpy().flatten()

@ray.remote
class Actor(object):
    def __init__(self, env_fn, learner_id, memory_id, action_dim, start_timesteps, load_freq, taper_load_freq, act_noise, noise_scale, param_noise, id):
        self.env = env_fn()

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])

        #self.policy = LN_Actor(self.state_dim, self.action_dim, self.max_action, 400, 300).to(device)
        self.learner_id = learner_id
        self.memory_id = memory_id
        
        self.start_timesteps = start_timesteps
        self.act_noise = act_noise
        self.noise_scale = noise_scale
        self.param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.05, desired_action_stddev=self.noise_scale, adaptation_coefficient=1.05) if param_noise else None

        self.max_traj_len = 400

        self.actor_timesteps = 0
        self.taper_timesteps = 0
        self.episode_num = 0
        self.taper_load_freq = taper_load_freq
        self.load_freq = load_freq             # initial load frequency... make this taper down to 1 over time 

        self.id = id

        self.policy, self.training_done = ray.get(self.learner_id.get_global_policy.remote())


    def collect_experience(self):

        with ray.profile("Actor collection loop", extra_data={'Actor id': str(self.id)}):
            # collection loop -  COLLECTS EPISODES OF EXPERIENCE UNTIL training_done
            while True:

                cassieEnv = True

                if self.actor_timesteps % self.load_freq == 0:
                    # PUTTING WAIT ON THIS SHOULD MAKE THIS EXACT SAME AS NON-DISTRIBUTED, if using one actor
                    # Query learner for latest model and termination flag

                    self.policy, self.training_done = ray.get(self.learner_id.get_global_policy.remote())

                    #global_policy_state_dict, training_done = ray.get(self.learner_id.get_global_policy.remote())
                    #self.policy.load_state_dict(global_policy_state_dict)
                    print("loaded global model")

                # self.policy, self.training_done = ray.get(self.learner_id.get_global_policy.remote())
                # print("loaded global model")

                if self.training_done:
                    break

                obs = self.env.reset()
                done = False
                episode_reward = 0
                episode_timesteps = 0

                # nested collection loop - COLLECTS TIMESTEPS OF EXPERIENCE UNTIL episode is over
                while episode_timesteps < self.max_traj_len and not done:

                    #self.env.render()

                    # Select action randomly or according to policy
                    if self.actor_timesteps < self.start_timesteps:
                        #print("selecting action randomly {}".format(done_bool))
                        action = torch.randn(self.env.action_space.shape[0]) if cassieEnv is True else self.env.action_space.sample()
                        action = action.numpy()
                    else:
                        #print("selecting from policy")
                        action = select_action(self.policy, np.array(obs), device)
                        if self.act_noise != 0:
                            action = (action + np.random.normal(0, self.act_noise,
                                                                size=self.env.action_space.shape[0])).clip(self.env.action_space.low, self.env.action_space.high)

                    # Perform action
                    new_obs, reward, done, _ = self.env.step(action)
                    done_bool = 1.0 if episode_timesteps + 1 == self.max_traj_len else float(done)
                    episode_reward += reward

                    # Store data in replay buffer
                    self.memory_id.add.remote((obs, new_obs, action, reward, done_bool))

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