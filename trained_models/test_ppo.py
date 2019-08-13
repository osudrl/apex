# TODO: organize this file
import argparse
import pickle
import torch

import time

# NOTE: importing cassie for some reason breaks openai gym, BUG ?
from cassie import CassieEnv, CassieTSEnv
from cassie.no_delta_env import CassieEnv_nodelta
from cassie.speed_env import CassieEnv_speed
from cassie.speed_double_freq_env import CassieEnv_speed_dfreq
from cassie.speed_no_delta_env import CassieEnv_speed_no_delta

from rl.envs import Normalize, Vectorize
from rl.policies import GaussianMLP, EnsemblePolicy

import matplotlib
from matplotlib import pyplot as plt
plt.style.use('ggplot')

import numpy as np
np.set_printoptions(precision=2, suppress=True)

from cassie.cassiemujoco import pd_in_t

#TODO: convert all of this into ipynb


# TODO: add .dt to all environments. OpenAI should do the same...
def visualize(env, policy, trj_len, deterministic=True, dt=0.033, speedup=1):
    R = []
    r_ep = 0
    done = False

    with torch.no_grad():
        state = torch.Tensor(env.reset())

        
        for t in range(trj_len):

            #start = time.time()
            _, action = policy.act(state, deterministic)
            #print("policy time: ", time.time() - start)

            #start = time.time()
            state, reward, done, _ = env.step(action.data.numpy())
            #print("env time: ", time.time() - start)

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
        print("avg timesteps:", trj_len / len(R))

            

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

        _, action = policy.act(state, True)

        X[t, :] = state.data.numpy()
        y[t, :] = action.data.numpy()

        state, _, _, _ = env.step(action.data.numpy())

    fig, axes = plt.subplots(1, action_dim)

    
    for a in range(action_dim):
        axes[a].plot(np.arange(trj_len), y[:, a])

    plt.show()

# TODO: add no_grad to all of these

def get_rewards(env, policy, num_trj, deterministic=False):
    r_ep = 0
    R = []
    done = False

    state = torch.Tensor(env.reset())
    while len(R) < num_trj:
        _, action = policy.act(state, deterministic)
        state, reward, done, _ = env.step(action.data.numpy())

        r_ep += reward

        if done:
            state = env.reset()
            R += [r_ep[0]]
            r_ep = 0


        state = torch.Tensor(state)

    if not done:
        R += [r_ep[0]]
    
    return R

# TODO: add n_trials arg, pass in policy function instead

def plot_reward_dist(env, policy, num_trj, deterministic=False):
    r_ep = 0
    R = []
    done = False

    state = torch.Tensor(env.reset())
    while len(R) < num_trj:
        _, action = policy.act(state, deterministic)
        state, reward, done, _ = env.step(action.data.numpy())

        r_ep += reward

        if done:
            state = env.reset()
            R += [r_ep[0]]
            r_ep = 0

        state = torch.Tensor(state)

    if not done:
        R += [r_ep[0]]

    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.hist(R, edgecolor='black', label=policy.__class__.__name__) # TODO: make labels less hacky

def plot_action_dist(env, policy, num_trj, deterministic=False):
    n = 0
    A = []
    done = False

    state = torch.Tensor(env.reset())
    while n < num_trj:
        _, action = policy.act(state, deterministic)
        state, reward, done, _ = env.step(action.data.numpy())

        A += [action.mean().data.numpy()]

        if done:
            state = env.reset()
            n += 1


        state = torch.Tensor(state)

    print(np.std(A))

    plt.xlabel("Mean Action")
    plt.ylabel("Frequency")
    plt.hist(A, edgecolor='black', label=policy.__class__.__name__) # TODO: make labels less hacky

def print_semantic_state(s):
    # only works for full state for now
    assert s.numel() == 80

    x = [None] * 20

    # TODO: add mask variable to env to specifiy which states are used

    x [ 0] = "Pelvis y"
    x [ 1] = "Pelvis z"
    x [ 2] = "Pelvis qw"
    x [ 3] = "Pelvis qx"
    x [ 4] = "Pelvis qy"
    x [ 5] = "Pelvis qz"
    x [ 6] = "Left hip roll"         #(Motor [0])
    x [ 7] = "Left hip yaw"          #(Motor [1])
    x [ 8] = "Left hip pitch"        #(Motor [2])
    x [ 9] = "Left knee"             #(Motor [3])
    x [10] = "Left shin"                        #(Joint [0])
    x [11] = "Left tarsus"                      #(Joint [1])
    x [12] = "Left foot"             #(Motor [4], Joint [2])
    x [13] = "Right hip roll"        #(Motor [5])
    x [14] = "Right hip yaw"         #(Motor [6])
    x [15] = "Right hip pitch"       #(Motor [7])
    x [16] = "Right knee"            #(Motor [8])
    x [17] = "Right shin"                       #(Joint [3])
    x [18] = "Right tarsus"                     #(Joint [4])
    x [19] = "Right foot"            #(Motor [9], Joint [5])

    x_dot = [None] * 20

    x_dot[ 0] = "Pelvis x"
    x_dot[ 1] = "Pelvis y"
    x_dot[ 2] = "Pelvis z"
    x_dot[ 3] = "Pelvis wx"
    x_dot[ 4] = "Pelvis wy"
    x_dot[ 5] = "Pelvis wz"
    x_dot[ 6] = "Left hip roll"         #(Motor [0])
    x_dot[ 7] = "Left hip yaw"          #(Motor [1])
    x_dot[ 8] = "Left hip pitch"        #(Motor [2])
    x_dot[ 9] = "Left knee"             #(Motor [3])
    x_dot[10] = "Left shin"                        #(Joint [0])
    x_dot[11] = "Left tarsus"                      #(Joint [1])
    x_dot[12] = "Left foot"             #(Motor [4], Joint [2])
    x_dot[13] = "Right hip roll"        #(Motor [5])
    x_dot[14] = "Right hip yaw"         #(Motor [6])
    x_dot[15] = "Right hip pitch"       #(Motor [7])
    x_dot[16] = "Right knee"            #(Motor [8])
    x_dot[17] = "Right shin"                       #(Joint [3])
    x_dot[18] = "Right tarsus"                     #(Joint [4])
    x_dot[19] = "Right foot"            #(Motor [9], Joint [5])

    s_obs = s[:, :40]
    s_ref = s[:, 40:]

    print("\nObserved position")
    for i in range(20):
        if s_obs[:, i].item() > 1:
            print("{0}: \r\t\t\t{1:.2f}".format(x[i], s_obs[:, i].item()))
    
    print("\nObserved velocity")
    for i in range(20):
        if s_obs[:, 20 + i].item() > 1:
            print("{0}: \r\t\t\t{1:.2f}".format(x_dot[i], s_obs[:, 20 + i].item()))

    print("\nReference position")
    for i in range(20):
        if s_ref[:, i].item() > 1:
            print("{0}: \r\t\t\t{1:.2f}".format(x[i], s_ref[:, i].item()))

    print("\nReference velocity")
    for i in range(20):
        if s_ref[:, 20 + i].item() > 1:
            print("{0}: \r\t\t\t{1:.2f}".format(x_dot[i], s_ref[:, 20 + i].item()))

# my idea for extending perturbation saliency to dynamical systems
def saliency(policy, state, naive=False):
    scores = torch.zeros_like(state)
    for i in range(state.size(1)):
        score = 0

        s_prime = state.clone()
        # sample values from (e - 0.25, e + 0.25)
        max_r = 0.25

        # change in value/related to number of samples
        dr = 0.05

        # current change
        r = -max_r # start at min r
        if not naive:
            while r <= max_r:
                # TODO: visualize s_prime
                # would require setting qpos and visualizing
                # setting qpos would require knowing the mapping of indices
                # from s to qpos
                s_prime[:, i] = state[:, i] + r

                v, a = policy.act(state)
                v_prime, a_prime = policy.act(s_prime)

                score += 0.5 * torch.norm(v - v_prime, 2)

                r += dr
        else:
            s_prime[:, i] = 0

            v, a = policy.act(state)
            v_prime, a_prime = policy.act(s_prime)

            score += 0.5 * torch.norm(v - v_prime, 2)


        score /= 2 * max_r
        scores[:, i] = score
    scores = scores / torch.max(scores) * 10
    print_semantic_state(scores)

# more plots:
# state visitation
# value function
# reward distribution

def make_env_fn(state_est=False):
    def _thunk():
        return CassieEnv("walking", clock_based=True, state_est=state_est)
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
    torch.set_num_threads(1) # see: https://github.com/pytorch/pytorch/issues/13757 

    if args.new:
        env_fn = make_env_fn(state_est=args.state_est)

        # env_fn = make_cassie_env("walking", clock_based=True)
        # env_fn = functools.partial(CassieEnv_speed, "walking", clock_based=True, state_est=False)
        # env_fn = functools.partial(CassieEnv_nodelta, "walking", clock_based=True, state_est=False)
        # env_fn = functools.partial(CassieEnv_speed_dfreq, "walking", clock_based = True, state_est=args.state_est)

        env = Vectorize([env_fn])

        obs_dim = env_fn().observation_space.shape[0] 
        action_dim = env_fn().action_space.shape[0]

        policy = GaussianMLP(obs_dim, action_dim, nonlinearity="relu", init_std=np.exp(-2), learn_std=False)

        # policy2 = ActorCriticNet(obs_dim, action_dim, [256, 256])

        # #print(policy,  sum(p.numel() for p in policy.parameters()))
        # #print(policy2,  sum(p.numel() for p in policy2.parameters()))

        # plot_action_dist(env, policy, 100)
        # plot_action_dist(env, policy2, 100)
        # plt.legend()
        # plt.show()

        # R1 = []
        # R2 = []
        # R3 = []
        # for _ in range(5):
        #     R1 += get_rewards(env, GaussianMLP(obs_dim, action_dim, nonlinearity="relu", init_std=np.exp(-2), learn_std=False), 50)
        #     R2 += get_rewards(env, ActorCriticNet(obs_dim, action_dim, [256, 256]), 50)
        #     R3 += get_rewards(env, XieNet(obs_dim, action_dim, [256, 256]), 50)

        
        # plt.xlabel("Return")
        # plt.ylabel("Frequency")
        # plt.hist(R1, edgecolor='black', label="GaussianMLP")
        # plt.hist(R2, edgecolor='black', label="ActorCriticNet")
        # plt.hist(R3, edgecolor='black', label="XieNet")
        # plt.legend()
        # plt.show()

        # plot_reward_dist(env, policy2, 500)

        if args.visualize:
            visualize(env, policy, args.vlen, deterministic=not args.noise)
        
        if args.plot:
            cassie_policyplot(env, policy, args.glen)

    else:
        policy = torch.load(args.model_path)
        policy.train(0)

        env_fn = make_env_fn()
        
        env = env_fn()

        # while True:
        #     state = torch.Tensor(env.reset())
        #     print("phase: {0}".format(env.venv.envs[0].phase))

        #     saliency(policy, state, naive=False)

        #     env.venv.envs[0].sim.step_pd(pd_in_t())

        #     env.render()
        #     input()

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
        policy = torch.load("trained_models/modelv{}.pt".format(i))

        vpolicy += [policy]

        print("visualizing policy {}".format(i))
        # if i == 15:
        visualize(env, policy, 1)
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