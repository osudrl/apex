import torch
import ray

from rl.policies.td3_actor_critic import LN_Actor as Actor

import numpy as np

device = torch.device("cpu")

def select_action(Policy, state, device):
    state = torch.FloatTensor(state.reshape(1, -1)).to(device)

    Policy.eval()

    return Policy(state).cpu().data.numpy().flatten()

@ray.remote
class evaluator():
    def __init__(self, env_fn, max_traj_len, render_policy=False):
        self.env = env_fn()

        self.max_traj_len = max_traj_len

        self.render_policy = render_policy

    def evaluate_policy(self, policy, eval_episodes):

        avg_reward = 0.0
        avg_eplen = 0.0

        for _ in range(eval_episodes):

            state = self.env.reset()
            done = False

            # evaluate performance of the passed model for one episode
            while avg_eplen < self.max_traj_len and not done:
                if self.render_policy:
                    env.render()

                # use model's greedy policy to predict action
                action = select_action(policy, np.array(state), device)

                # take a step in the simulation
                next_state, reward, done, _ = self.env.step(action)

                # update state
                state = next_state

                # increment total_reward and step count
                avg_reward += reward
                avg_eplen += 1

        avg_reward /= eval_episodes
        avg_eplen /= eval_episodes

        return avg_reward, avg_eplen