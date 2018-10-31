import pickle
import torch

from cassie import CassieEnv
from rl.envs import Normalize, Vectorize
from rl.policies import GaussianMLP

import matplotlib
from matplotlib import pyplot as plt
plt.style.use('ggplot')

import numpy as np


policy, ob_rms = torch.load("trained_models/model2.pt")

def make_env_fn():
    def _thunk():
        return CassieEnv("cassie/trajectory/stepdata.bin")
    return _thunk

env_fn = make_env_fn()

env = Normalize(Vectorize([env_fn]))

env.ob_rms = ob_rms

#policy = GaussianMLP(env.observation_space.shape[0], env.action_space.shape[0])


def visualize(env, policy, trj_len):

    with torch.no_grad():
        state = torch.Tensor(env.reset())

        for t in range(trj_len):
            _, action = policy.act(state, True)

            state, reward, done, _ = env.step(action.data.numpy())

            if done:
                state = env.reset()

            state = torch.Tensor(state)

            env.render()

def cassie_policyplot(env, policy, trj_len):
    cassie_action = ["hip roll", "hip yaw", "hip pitch", "knee", "foot"]

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    y_delta = np.zeros((trj_len, action_dim))
    y_ref   = np.zeros((trj_len, action_dim))
    X       = np.zeros((trj_len, obs_dim))

    with torch.no_grad():
        state = torch.Tensor(env.reset())

        for t in range(trj_len):
            _, action = policy.act(state, True)

            X[t, :] = state.data.numpy()
            y_delta[t, :] = action.data.numpy() # policy delta

            # oooof this is messy/hackish
            ref_pos, _ = env.venv.envs[0].get_ref_state(env.venv.envs[0].phase)
            y_ref[t, :] = ref_pos[env.venv.envs[0].pos_idx] # base PD target

            state, reward, done, _ = env.step(action.data.numpy())

            state = torch.Tensor(state)

    # one row for each leg
    plot_rows = 2 
    plot_cols = action_dim // 2

    fig, axes = plt.subplots(plot_rows, plot_cols, figsize=(20, 10))

    for r in range(plot_rows):     # 2 legs
        for c in range(plot_cols): # 5 actions
            a = r * plot_cols + c
            axes[r][c].plot(np.arange(trj_len), y_delta[:, a], "C0", label="delta")
            axes[r][c].plot(np.arange(trj_len), y_ref[:, a], "C1", label="reference")
            axes[r][c].plot(np.arange(trj_len), y_delta[:, a] + y_ref[:, a], "C2--", label="summed")

            axes[0][c].set_xlabel(cassie_action[c])
            axes[0][c].xaxis.set_label_position('top') 
        axes[r][0].set_ylabel(["left leg", "right leg"][r])
    
    plt.tight_layout()
    axes[0][0].legend(loc='upper left')
    plt.show()

visualize(env, policy, 75)
cassie_policyplot(env, policy, 75)


def policyplot(env, policy, trj_len):
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    y = np.zeros((trj_len, action_dim))
    X = np.zeros((trj_len, obs_dim))

    state = torch.Tensor(env.reset())
    for t in range(trj_len):

        _, action = policy(state, True)

        X[t, :] = state.data.numpy()
        y[t, :] = action.data.numpy()

        state, _, _, _ = env.step(action.data.numpy())

    fig, axes = plt.subplots(1, action_dim)

    
    for a in range(action_dim):
        axes[a].plot(np.arange(trj_len), y[:, a])

    plt.show()