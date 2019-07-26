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
def evaluator(env, policy, max_traj_len):

    env = env()

    state = env.reset()
    total_reward = 0
    steps = 0
    done = False

    # evaluate performance of the passed model for one episode
    while steps < max_traj_len and not done:
        # use model's greedy policy to predict action
        action = select_action(policy, np.array(state), device)

        # take a step in the simulation
        next_state, reward, done, _ = env.step(action)

        # update state
        state = next_state

        # increment total_reward and step count
        total_reward += reward
        steps += 1

    return total_reward


def evaluate_policy(env, policy, max_episode_steps, eval_episodes=1):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        t = 0
        done_bool = 0.0
        while not done_bool:
            env.render()
            t += 1
            action = select_action(policy, np.array(obs), device)
            obs, reward, done, _ = env.step(action)
            done_bool = 1.0 if t + 1 == max_episode_steps else float(done)
            avg_reward += reward

    avg_reward /= eval_episodes

    # print("---------------------------------------")
    # print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    # print("---------------------------------------")
    return avg_reward