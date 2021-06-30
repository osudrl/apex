import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import time, os, sys

from util import env_factory
from rl.policies.actor import Gaussian_FF_Actor

import argparse
import pickle

# Get policy to plot from args, load policy and env
parser = argparse.ArgumentParser()
# General args
parser.add_argument("--path", type=str, default=None, help="path to folder containing policy and run details")
parser.add_argument("--pre_steps", type=int, default=300, help="Number of \"presteps\" to take for the policy to stabilize before data is recorded")
parser.add_argument("--num_steps", type=int, default=100, help="Number of steps to record")
parser.add_argument("--pre_speed", type=float, default=0.5, help="Commanded action during the presteps")
parser.add_argument("--plot_speed", type=float, default=0.5, help="Commanded action during the actual recorded steps")
args = parser.parse_args()

run_args = pickle.load(open(args.path + "experiment.pkl", "rb"))

print("pre steps: {}\tnum_steps: {}".format(args.pre_steps, args.num_steps))
print("pre speed: {}\tplot_speed: {}".format(args.pre_speed, args.plot_speed))

# RUN_NAME = run_args.run_name if run_args.run_name != None else "plot"

# RUN_NAME = "7b7e24-seed0"
# POLICY_PATH = "../trained_models/ppo/Cassie-v0/" + RUN_NAME + "/actor.pt"

# Load environment and policy
if (not hasattr('run_args', 'simrate')):
    run_args.simrate = 50
print("simrate:", run_args.simrate)
# run_args.simrate = 60
env_fn = env_factory(run_args.env_name, traj=run_args.traj, simrate=run_args.simrate, state_est=run_args.state_est, no_delta=run_args.no_delta, dynamics_randomization=run_args.dyn_random, 
                    mirror=False, clock_based=run_args.clock_based, learn_gains=run_args.learn_gains, reward=run_args.reward, history=run_args.history)
cassie_env = env_fn()
cassie_env.learn_gains = False
policy = torch.load(os.path.join(args.path, "actor.pt"))
policy.eval()

if hasattr(policy, 'init_hidden_state'):
    policy.init_hidden_state()

obs_dim = cassie_env.observation_space.shape[0] # TODO: could make obs and ac space static properties
action_dim = cassie_env.action_space.shape[0]

no_delta = cassie_env.no_delta
limittargs = False
lininterp = False
offset = cassie_env.offset
print("phaselen:", cassie_env.phaselen)
# exit()

num_steps = args.num_steps
pre_steps = args.pre_steps
simrate = cassie_env.simrate
num_cycles = 10
targets = np.zeros((80*simrate, 10, num_cycles))

pos_idx = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
vel_idx = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]

# Execute policy and save torques
with torch.no_grad():
    state = cassie_env.reset_for_test()
    cassie_env.update_speed(3.5)
    for i in range(80*3):
        action = policy.forward(torch.Tensor(state), deterministic=True).detach().numpy()
        state, reward, done, _ = cassie_env.step(action)
        state = torch.Tensor(state)
    print("phase:", cassie_env.phase)
    print("counter:", cassie_env.counter)
    # exit()
    for k in range(num_cycles):
        for i in range(80):
            action = policy.forward(torch.Tensor(state), deterministic=True).detach().numpy()
            for j in range(simrate):
                target = action + cassie_env.offset
                targets[i*simrate+j, :, k] = target
                
                cassie_env.step_simulation(action)

            cassie_env.time  += 1
            cassie_env.phase += cassie_env.phase_add

            if cassie_env.phase >= cassie_env.phaselen:
                cassie_env.phase -= cassie_env.phaselen
                cassie_env.counter += 1

            state = cassie_env.get_full_state()
        # cassie_env.render()

print("phase:", cassie_env.phase)
print("counter:", cassie_env.counter)
print("var:", np.max(np.std(targets, axis=2)))
avg_targ = np.mean(targets, axis=2)
print(avg_targ.shape)

# Graph PD target data
fig, ax = plt.subplots(2, 5, figsize=(15, 5))
t = np.linspace(0, 80*simrate*0.0005, 80*simrate)
titles = ["Hip Roll", "Hip Yaw", "Hip Pitch", "Knee", "Foot"]
ax[0][0].set_ylabel("PD Target")
ax[1][0].set_ylabel("PD Target")
for i in range(5):
    ax[0][i].plot(t, avg_targ[:, i])
    ax[0][i].set_title("Left " + titles[i])
    ax[1][i].plot(t, avg_targ[:, i+5])
    ax[1][i].set_title("Right " + titles[i])
    ax[1][i].set_xlabel("Time (sec)")

plt.tight_layout()
plt.show()

# fig, ax = plt.subplots(2, 5, figsize=(15, 5))
# t = np.linspace(0, 80*simrate*num_cycles*0.0005, 80*num_cycles*simrate)
# full_targ = np.zeros((80*simrate*num_cycles, 10))
# for i in range(num_cycles):
#     full_targ[80*simrate*i:80*simrate*(i+1), :] = targets[:, :, i]
# # full_targ = np.reshape(targets, (80*simrate*num_cycles, 10))
# print("full targ:", full_targ.shape)
# titles = ["Hip Roll", "Hip Yaw", "Hip Pitch", "Knee", "Foot"]
# ax[0][0].set_ylabel("PD Target")
# ax[1][0].set_ylabel("PD Target")
# for i in range(5):
#     ax[0][i].plot(t, full_targ[:, i])
#     ax[0][i].set_title("Left " + titles[i])
#     ax[1][i].plot(t, full_targ[:, i+5])
#     ax[1][i].set_title("Right " + titles[i])
#     ax[1][i].set_xlabel("Time (sec)")

# plt.tight_layout()
# plt.show()

np.save("./sim_pol_targets.npy", avg_targ)
