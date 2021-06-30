import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import time, os, sys

from util import env_factory
from rl.policies.actor import Gaussian_FF_Actor

import argparse
import pickle

def vel_est(t, y, order=2):
    p = np.polyfit(t, y, order)
    x = t[-1]
    vel = 0
    for i in range(len(p)-1):
        vel += p[i]*x**(order-i-1)
    return vel

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
env_fn = env_factory(run_args.env_name, traj=run_args.traj, simrate=run_args.simrate, state_est=run_args.state_est, no_delta=run_args.no_delta, dynamics_randomization=run_args.dyn_random, 
                    mirror=False, clock_based=run_args.clock_based, learn_gains=run_args.learn_gains, reward=run_args.reward, history=run_args.history)
cassie_env = env_fn()
cassie_env_smooth = env_fn()
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
window_size = 3
order = 2
t = np.linspace(0, 0.0005*(window_size - 1), window_size)

num_steps = args.num_steps
pre_steps = args.pre_steps
simrate = cassie_env.simrate
torque = np.zeros((num_steps*simrate, 10))
torque_smooth = np.zeros((num_steps*simrate, 10))
motor_pos = np.zeros((num_steps*simrate, 10))
motor_vel = np.zeros((num_steps*simrate, 10))
mj_pos = np.zeros((num_steps*simrate, 10))
mj_vel = np.zeros((num_steps*simrate, 10))
mj_pos_smooth = np.zeros((num_steps*simrate, 10))
mj_vel_smooth = np.zeros((num_steps*simrate, 10))
filter_vel = np.zeros((num_steps*simrate, 10))

pos_idx = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
vel_idx = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]

# Execute policy and save torques
with torch.no_grad():
    state = cassie_env.reset_for_test()
    state_smooth = cassie_env_smooth.reset_for_test()
    cassie_env_smooth.true_vel = True
    cassie_env.update_speed(args.pre_speed)
    cassie_env_smooth.update_speed(args.pre_speed)
    for i in range(pre_steps):
        action = policy.forward(torch.Tensor(state), deterministic=True).detach().numpy()
        state = cassie_env.step_basic(action)
        state = torch.Tensor(state)
        action_smooth = policy.forward(torch.Tensor(state_smooth), deterministic=True).detach().numpy()
        state_smooth = cassie_env_smooth.step_basic_ownPD(action_smooth)
        # state_smooth = cassie_env_smooth.step_basic(action)
        state_smooth = torch.Tensor(state_smooth)

    cassie_env.update_speed(args.plot_speed)
    cassie_env_smooth.update_speed(args.plot_speed)
    for i in range(num_steps):
        action = policy.forward(torch.Tensor(state), deterministic=True).detach().numpy()
        action_smooth = policy.forward(torch.Tensor(state_smooth), deterministic=True).detach().numpy()
        for j in range(simrate):
            cassie_env.step_sim_basic(action)
            cassie_env_smooth.step_sim_basic_ownPD(action_smooth)
            # cassie_env_smooth.step_sim_basic(action)
            curr_qpos = cassie_env.sim.qpos()
            curr_qvel = cassie_env.sim.qvel()
            curr_qpos_smooth = cassie_env_smooth.sim.qpos()
            curr_qvel_smooth = cassie_env_smooth.sim.qvel()
            curr_ind = i*simrate+j
            
            # motor_pos[curr_ind, :] = cassie_env.cassie_state.motor.position[:]
            # motor_vel[curr_ind, :] = cassie_env.cassie_state.motor.velocity[:]
            mj_pos[curr_ind, :] = np.array(curr_qpos)[cassie_env.pos_idx]
            mj_vel[curr_ind, :] = np.array(curr_qvel)[cassie_env.vel_idx]
            mj_pos_smooth[curr_ind, :] = np.array(curr_qpos_smooth)[cassie_env.pos_idx]
            mj_vel_smooth[curr_ind, :] = np.array(curr_qvel_smooth)[cassie_env.vel_idx]
            torque[curr_ind, :] = cassie_env.cassie_state.motor.torque[:]
            torque_smooth[curr_ind, :] = cassie_env_smooth.cassie_state.motor.torque[:]
            # if curr_ind >= window_size:
            #     for k in range(10):
            #         filter_vel[curr_ind, k] = vel_est(t, mj_pos[curr_ind-window_size:curr_ind, k], order=order)
            #     target = action + cassie_env.offset
            #     for k in range(5):
            #         self_torque[curr_ind, k] = cassie_env.P[k] * (target[k] - motor_pos[curr_ind, k]) + cassie_env.D[k]*(-filter_vel[curr_ind, k])
            #         self_torque[curr_ind, k+5] = cassie_env.P[k] * (target[k+5] - motor_pos[curr_ind, k+5]) + cassie_env.D[k]*(-filter_vel[curr_ind, k+5])


        cassie_env.phase += cassie_env.phase_add

        if cassie_env.phase > cassie_env.phaselen:
            cassie_env.phase -= cassie_env.phaselen
            cassie_env.counter += 1

        state = cassie_env.get_full_state()

        cassie_env_smooth.phase += cassie_env_smooth.phase_add

        if cassie_env_smooth.phase > cassie_env_smooth.phaselen:
            cassie_env_smooth.phase -= cassie_env_smooth.phaselen
            cassie_env_smooth.counter += 1

        state_smooth = cassie_env_smooth.get_full_state()


start_ind = 0#2900
end_ind = -1#3000#6000

fig, ax = plt.subplots(2, 5, figsize=(15, 5))
time = np.linspace(0, num_steps*simrate*0.0005, num_steps*simrate)
titles = ["Hip Roll", "Hip Yaw", "Hip Pitch", "Knee", "Foot"]
ax[0][0].set_ylabel("Pos [rad]")
ax[1][0].set_ylabel("Pos [rad]")
print(mj_pos_smooth[0, :])

for i in range(5):
    ax[0][i].plot(time[start_ind:end_ind], mj_pos[start_ind:end_ind, i], label="orig")
    ax[0][i].plot(time[start_ind:end_ind], mj_pos_smooth[start_ind:end_ind, i], label="smooth", c='C1')
    # ax[0][i].plot(time[start_ind:end_ind:50], motor_pos[start_ind:end_ind:50, i], label="est_sub")
    ax[0][i].set_title("Left " + titles[i])
    ax[0][i].legend()

    ax[1][i].plot(time[start_ind:end_ind], mj_pos[start_ind:end_ind, i+5], label="orig")
    ax[1][i].plot(time[start_ind:end_ind], mj_pos_smooth[start_ind:end_ind, i+5], label="smooth")
    # ax[1][i].plot(time[start_ind:end_ind:50], motor_pos[start_ind:end_ind:50, i+5], label="est_sub")
    ax[1][i].set_title("Right " + titles[i])
    ax[1][i].legend()
    ax[1][i].set_xlabel("Time (sec)")

fig.suptitle("Motor Pos")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

fig, ax = plt.subplots(2, 5, figsize=(15, 5))
titles = ["Hip Roll", "Hip Yaw", "Hip Pitch", "Knee", "Foot"]
ax[0][0].set_ylabel("Vel [rad/s]")
ax[1][0].set_ylabel("Vel [rad/s]")

for i in range(5):
    ax[0][i].plot(time[start_ind:end_ind], mj_vel[start_ind:end_ind, i], label="orig")
    ax[0][i].plot(time[start_ind:end_ind], mj_vel_smooth[start_ind:end_ind, i], label="smooth")
    # ax[0][i].plot(time[start_ind:end_ind], filter_vel[start_ind:end_ind, i], label="poly_fit")
    # ax[0][i].plot(time[start_ind:end_ind:50], motor_vel[start_ind:end_ind:50, i], label="est_sub")
    ax[0][i].set_title("Left " + titles[i])
    ax[0][i].legend()

    ax[1][i].plot(time[start_ind:end_ind], mj_vel[start_ind:end_ind, i+5], label="orig")
    ax[1][i].plot(time[start_ind:end_ind], mj_vel_smooth[start_ind:end_ind, i+5], label="smooth")
    # ax[1][i].plot(time[start_ind:end_ind], filter_vel[start_ind:end_ind, i+5], label="poly_fit")
    # ax[1][i].plot(time[start_ind:end_ind:50], motor_vel[start_ind:end_ind:50, i+5], label="est_sub")
    ax[1][i].set_title("Right " + titles[i])
    ax[1][i].legend()
    ax[1][i].set_xlabel("Time (sec)")

fig.suptitle("Motor Vel")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

fig, ax = plt.subplots(2, 5, figsize=(15, 5))
titles = ["Hip Roll", "Hip Yaw", "Hip Pitch", "Knee", "Foot"]
ax[0][0].set_ylabel("Torque [N*m]")
ax[1][0].set_ylabel("Torque [N*m]")

for i in range(5):
    ax[0][i].plot(time[start_ind:end_ind], torque[start_ind:end_ind, i], label="lib")
    ax[0][i].plot(time[start_ind:end_ind], torque_smooth[start_ind:end_ind, i], label="self")
    ax[0][i].set_title("Left " + titles[i])

    ax[1][i].plot(time[start_ind:end_ind], torque[start_ind:end_ind, i+5], label="lib")
    ax[1][i].plot(time[start_ind:end_ind], torque_smooth[start_ind:end_ind, i+5], label="self")
    ax[1][i].set_title("Right " + titles[i])
    ax[1][i].set_xlabel("Time (sec)")

fig.suptitle("Motor Torque")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()