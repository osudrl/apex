# TODO: organize this file
import argparse
import pickle
import torch
import time

from cassie import CassieEnv
from rl.envs import Normalize, Vectorize
from rl.policies import GaussianMLP, EnsemblePolicy

import matplotlib
from matplotlib import pyplot as plt
plt.style.use('ggplot')

import numpy as np
np.set_printoptions(precision=2, suppress=True)


# TODO: add .dt to all environments. OpenAI should do the same...
def visualize(env, policy, trj_len, deterministic=True, dt=0.033, speedup=1):
    R = []
    r_ep = 0
    done = False

    with torch.no_grad():
        state = torch.Tensor(env.reset())

        
        for t in range(trj_len):
            _, action = policy.act(state, deterministic)

            state, reward, done, _ = env.step(action.data.numpy())

            r_ep += reward

            if done:
                state = env.reset()
                R += [r_ep]
                r_ep = 0

            state = torch.Tensor(state)

            env.render()

            time.sleep(dt / speedup)
        
        if not done:
            R += [r_ep]

        print("avg reward:", sum(R)/len(R))

            

def cassie_policyplot(env, policy, trj_len, render=False, title=None):
    cassie_action = ["hip roll", "hip yaw", "hip pitch", "knee", "foot"]

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    y_delta = np.zeros((trj_len, action_dim))
    y_ref   = np.zeros((trj_len, action_dim))
    X       = np.zeros((trj_len, action_dim)) # not real X

    with torch.no_grad():
        state = torch.Tensor(env.reset())

        for t in range(trj_len):
            _, action = policy.act(state, True)

            #X[t, :] = state.data.numpy()
            y_delta[t, :] = action.data.numpy() # policy delta

            # oooof this is messy/hackish
            ref_pos, _ = env.venv.envs[0].get_ref_state(env.venv.envs[0].phase)
            y_ref[t, :] = ref_pos[env.venv.envs[0].pos_idx] # base PD target

            X[t, :] = np.copy(env.venv.envs[0].sim.qpos())[env.venv.envs[0].pos_idx]

            state, reward, done, _ = env.step(action.data.numpy())

            state = torch.Tensor(state)

            if render:
                env.render()
                time.sleep(0.033)

    # one row for each leg
    plot_rows = 2 
    plot_cols = action_dim // 2

    fig, axes = plt.subplots(plot_rows, plot_cols, figsize=(20, 10))

    if title is not None:
        fig.suptitle(title, fontsize=16)

    for r in range(plot_rows):     # 2 legs
        for c in range(plot_cols): # 5 actions
            a = r * plot_cols + c
            axes[r][c].plot(np.arange(trj_len), y_delta[:, a], "C0", label="delta")
            axes[r][c].plot(np.arange(trj_len), y_ref[:, a], "C1", label="reference")
            axes[r][c].plot(np.arange(trj_len), y_delta[:, a] + y_ref[:, a], "C2--", label="summed")

            axes[r][c].plot(np.arange(trj_len), X[:, a], "g--", label="result")

            axes[0][c].set_xlabel(cassie_action[c])
            axes[0][c].xaxis.set_label_position('top') 
        axes[r][0].set_ylabel(["left leg", "right leg"][r])
    
    plt.tight_layout()

    if title is not None:
        plt.subplots_adjust(top=0.93)


    axes[0][0].legend(loc='upper left')
    plt.show()


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

def make_env_fn():
    def _thunk():
        return CassieEnv("cassie/trajectory/stepdata.bin")
    return _thunk


parser = argparse.ArgumentParser(description="Run a model, including visualization and plotting.")
parser.add_argument("-p", "--model_path", type=str, default="/trained_models/model1.ptn",
                    help="File path for model to test")
parser.add_argument("-x", "--no-visualize", dest="visualize", default=True, action='store_false',
                    help="Don't render the policy.")
parser.add_argument("-g", "--graph", dest="plot", default=False, action='store_true',
                    help="Graph the output of the policy.")

parser.add_argument("--glen", type=int, default=150,
                    help="Length of trajectory to graph.")
parser.add_argument("--vlen", type=int, default=75,
                    help="Length of trajectory to visualize")

parser.add_argument("--noise", default=False, action="store_true",
                    help="Visualize policy with exploration.")

parser.add_argument("--new", default=False, action="store_true",
                   help="Visualize new (untrained) policy")

args = parser.parse_args()


# TODO: add command line arguments for normalization on/off, and for ensemble policy?

if __name__ == "__main__":
    if args.new:
        env_fn = make_env_fn()
        env = Vectorize([env_fn])

        obs_dim = env_fn().observation_space.shape[0] 
        action_dim = env_fn().action_space.shape[0]

        policy = GaussianMLP(obs_dim, action_dim, nonlinearity="relu", init_std=np.exp(-2), learn_std=False)

        if args.visualize:
            visualize(env, policy, args.vlen, deterministic=not args.noise)
        
        if args.plot:
            cassie_policyplot(env, policy, args.glen)

    else:
        policy, rms = torch.load(args.model_path)

        env_fn = make_env_fn()
        env = Normalize(Vectorize([env_fn]))

        env.ob_rms, env.ret_rms = rms

        env.ret = env.ret_rms is not None

        if args.visualize:
            visualize(env, policy, args.vlen, deterministic=not args.noise)
        
        if args.plot:
            cassie_policyplot(env, policy, args.glen)

    exit()


    # TODO: try averaging obs_norm? to seperate obs normalization for each
    # averaging obs_norm probably wont work as all policies are expecting different normalization parameters
    # ob normalization should therefore should either be part of the policy or somehow syncronized
    # across experiments. The former is easier to implement
    # policy could keep track of its own obs_rms and the env could take it in and update it?
    # ^ might break OpenAI gym compatibility

    # other possibility: maybe norm parameters converge on their own over time? Although if they did
    # an ensemble probably wouldn't change behavior

    # SOLUTION: give trained policies an ob_rms parameter to normalize their own observations,
    # keep normalization calculation in environment for parallelization

    # NOTE: reward normalization affects stuff

    #### Ensemble policy
    vpolicy = []

    mmean = 0
    mvar = 0
    for i in range(1, 16):
        policy, rms = torch.load("trained_models/modelv{}.pt".format(i))

        #print(np.copy(rms.mean).mean())
        #print(np.copy(rms.var).mean())

        mmean += np.copy(rms.mean)
        mvar += np.copy(rms.var)

        vpolicy += [policy]

        env.ob_rms = rms

        print("visualizing policy {}".format(i))
        # if i == 15:
        visualize(env, policy, 100)
            #cassie_policyplot(env, policy, 100, "policy {}".format(i))

    mmean /= 10
    mvar /= 10

    env.ob_rms.mean = mmean
    env.ob_rms.var = mvar

    #for i in range(10):
        
    # NOTE: reference trajectory for ensemble policy looks weird?
    # Because it's the NORMALIZED reference trajectory^^
    # but still, weird
    policy = EnsemblePolicy(vpolicy)

    #visualize(env, policy, 100)
    cassie_policyplot(env, policy, 75, render=True)