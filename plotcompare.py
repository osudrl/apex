import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import time

from rl.utils import renderpolicy
from cassie import CassieEnv
from cassie.no_delta_env import CassieEnv_nodelta
from cassie.speed_env import CassieEnv_speed
from cassie.speed_double_freq_env import CassieEnv_speed_dfreq
from cassie.speed_no_delta_neutral_foot_env import CassieEnv_speed_no_delta_neutral_foot
from cassie.standing_env import CassieEnv_stand
from cassie.speed_sidestep_env import CassieEnv_speed_sidestep

from rl.policies import GaussianMLP

# Load environment and policy
cassie_env = CassieEnv_speed_no_delta_neutral_foot("walking", clock_based=True, state_est=True)

obs_dim = cassie_env.observation_space.shape[0] # TODO: could make obs and ac space static properties
action_dim = cassie_env.action_space.shape[0]

offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])
num_steps = 28*2
pre_steps = 28*2

pos_idx = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
vel_idx = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]

pol_names = ["fwrd_walk_StateEst_speed-05-3_freq1-2", "fwrd_walk_StateEst_speed-05-3_freq1-2_footvelpenalty_(1)", "fwrd_walk_StateEst_speed-05-3_freq1-2_footvelbonus"]
policies = []
total_pelaccel = []
total_footvel = []
total_GRFs = []
for i in range(3):
    policy = torch.load("./trained_models/{}.pt".format(pol_names[i]))
    policy.eval()
    pelaccel = np.zeros(num_steps*60)
    footvel = np.zeros((num_steps*60, 2))
    GRFs = np.zeros((num_steps*60, 2))
    with torch.no_grad():
        state = torch.Tensor(cassie_env.reset_for_test())
        cassie_env.speed = .5
        cassie_env.phase_add = 1
        for j in range(pre_steps):
            _, action = policy.act(state, True)
            state, reward, done, _ = cassie_env.step(action.data.numpy())
            state = torch.Tensor(state)
        for m in range(10):
            for j in range(num_steps):
                _, action = policy.act(state, True)
                action = action.data.numpy()
                for k in range(60):
                    cassie_env.step_simulation(action)
                    curr_qpos = cassie_env.sim.qpos()
                    pelaccel[j*60+k] += cassie_env.cassie_state.pelvis.translationalAcceleration[2]#np.linalg.norm(cassie_env.cassie_state.pelvis.translationalAcceleration)
                    footvel[j*60+k, :] = [cassie_env.lfoot_vel, cassie_env.rfoot_vel]
                    GRFs[j*60+k, :] = cassie_env.sim.get_foot_forces()

                cassie_env.time  += 1
                cassie_env.phase += cassie_env.phase_add

                if cassie_env.phase > cassie_env.phaselen:
                    cassie_env.phase = 0
                    cassie_env.counter += 1

                state = cassie_env.get_full_state()
                state = torch.Tensor(state)
    pelaccel  = pelaccel / 10
    print("max pel accel: ", np.max(pelaccel))
    print("max foot vel: ", np.max(footvel))
    total_pelaccel.append(pelaccel)
    total_footvel.append(footvel)
    total_GRFs.append(GRFs)

# Graph compare pelaccel data
fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
t = np.linspace(0, (num_steps-1)*60*0.0005, num_steps*60)
ax[0].set_ylabel("Pelvis Z acceleration (m/s^2)")
ax[0].plot(t, total_pelaccel[0][:], label="original")
ax[0].plot(t, total_pelaccel[1][:], label="acceleration penalty")
ax[0].set_title("Pelvis Z acceleration Comparison")
ax[0].set_xlabel("Time (sec)")
ax[0].legend()
# ax[1].set_ylabel("Left Foot Z velocity (m/s)")
ax[1].plot(t, total_pelaccel[0][:], label="original")
ax[1].plot(t, total_pelaccel[2][:], label="acceleration bonus")
ax[1].set_title("Pelvis Z acceleration Comparison")
ax[1].set_xlabel("Time (sec)")
ax[1].legend()
# fig, ax = plt.subplots(figsize=(15, 5))
# t = np.linspace(0, num_steps-1*0.0005, num_steps*60)
# ax.set_ylabel("Pelvis Z acceleration (m/s^2)")
# labels = ["original", "acceleration penalty", "acceleration bonus"]
# for i in range(3):
#     ax.plot(t, total_pelaccel[i], label=labels[i])
# plt.legend()
# ax.set_title("Pelvis Z accerlation Comparison")
# ax.set_xlabel("Time (sec)")

plt.tight_layout()
plt.savefig("./paper_plots/pelaccel_compare.png")

# Graph compare footvel data
fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
t = np.linspace(0, (num_steps-1)*60*0.0005, num_steps*60)
ax[0].set_ylabel("Left Foot Z velocity (m/s)")
ax[0].plot(t, total_footvel[0][:, 0], label="original")
ax[0].plot(t, total_footvel[1][:, 0], label="foot velocity penalty")
ax[0].set_title("Left Foot Z velocity Comparison")
ax[0].set_xlabel("Time (sec)")
ax[0].legend()
# ax[1].set_ylabel("Left Foot Z velocity (m/s)")
ax[1].plot(t, total_footvel[0][:, 0], label="original")
ax[1].plot(t, total_footvel[2][:, 0], label="foot velocity bonus")
ax[1].set_title("Left Foot Z velocity Comparison")
ax[1].set_xlabel("Time (sec)")
ax[1].legend()

plt.tight_layout()
plt.savefig("./paper_plots/footvel_compare.png")

# Graph compare footvel data
fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
t = np.linspace(0, (num_steps-1)*60*0.0005, num_steps*60)
ax[0].set_ylabel("Left GRF (N)")
ax[0].plot(t, total_GRFs[0][:, 0], label="original")
ax[0].plot(t, total_GRFs[1][:, 0], label="foot velocity penalty")
ax[0].set_title("Left Foot GRF Comparison")
ax[0].set_xlabel("Time (sec)")
ax[0].legend()
# ax[1].set_ylabel("Left GRF (N)")
ax[1].plot(t, total_GRFs[0][:, 0], label="original")
ax[1].plot(t, total_GRFs[2][:, 0], label="foot velocity bonus")
ax[1].set_title("Left Foot GRF Comparison")
ax[1].set_xlabel("Time (sec)")
ax[1].legend()

plt.tight_layout()
plt.savefig("./paper_plots/GRF_compare.png")