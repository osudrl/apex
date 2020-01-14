import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import time
import pickle

from rl.utils import renderpolicy
from cassie.speed_no_delta_neutral_foot_env import CassieEnv_speed_no_delta_neutral_foot
from cassie.speed_sidestep_env import CassieEnv_speed_sidestep

from rl.policies import GaussianMLP

class CassieFootTrajectory:
    def __init__(self, filepath):
        with open(filepath, "rb") as f:
            trajectory = pickle.load(f)

        # com_xyz = np.copy(trajectory["qpos"][:, 0:3])
        # rfoot_relative = np.copy(trajectory["rfoot"])
        # lfoot_relative = np.copy(trajectory["lfoot"])
        # self.rfoot = com_xyz + rfoot_relative
        # self.lfoot = com_xyz + lfoot_relative
        self.rfoot = trajectory["rfoot"]
        self.lfoot = trajectory["lfoot"]
    
    def __len__(self):
        return len(self.rfoot)

# Load environment and policy
simrate = 15
cassie_env = CassieEnv_speed_no_delta_neutral_foot("walking", simrate = simrate, clock_based=True, state_est=True)
# cassie_env = CassieEnv_speed_sidestep("walking", simrate=simrate, clock_based=True, state_est=True)
obs_dim = cassie_env.observation_space.shape[0] # TODO: could make obs and ac space static properties
action_dim = cassie_env.action_space.shape[0]

offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])
file_prefix = "fwrd_walk_StateEst_speed-05-1_freq1_foottraj_land0.2_simrate15_(1)"
# file_prefix = "sidestep_StateEst_justfootmatch_simrate15_bigweight"
# file_prefix = "sidestep_StateEst_speedmatch_footytraj_doublestance_time0.4_land0.4_vels_avgdiff_simrate15_bigweight_actpenalty_retrain_(2)"
policy = torch.load("./trained_models/{}.pt".format(file_prefix))
# policy.bounded = False
policy.eval()
# foot_traj = CassieFootTrajectory("./cassie/trajectory/foottraj_doublestance_time0.4_land0.4_h0.2.pkl")
foot_traj = CassieFootTrajectory("./cassie/trajectory/foottraj_doublestance_time0.4_land0.2_vels.pkl")
# foot_traj = CassieFootTrajectory("./cassie/trajectory/foottraj_doublestance_time0.4_land1.0_h0.2.pkl")

num_steps = cassie_env.phaselen + 1
print("num_steps: ", num_steps)
pre_steps = num_steps*5
state_est_foot = np.zeros((num_steps*simrate, 6))
mj_foot_pos = np.zeros((num_steps*simrate, 6))
foot_vel = np.zeros((num_steps*simrate, 2))
traj_vel = np.zeros((len(foot_traj), 6))

pos_idx = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
vel_idx = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]
# Execute policy and save torques
with torch.no_grad():
    state = torch.Tensor(cassie_env.reset_for_test())
    cassie_env.speed = 0
    # cassie_env.side_speed = .2
    cassie_env.phase_add = 1.0
    # print("first phase: ", cassie_env.phase)
    for i in range(pre_steps):
        _, action = policy.act(state, True)
        state, reward, done, _ = cassie_env.step(action.data.numpy())
        state = torch.Tensor(state)
        # print("phase: ", cassie_env.phase)
    # print("phase at begin: ", cassie_env.phase)
    for i in range(num_steps):
        _, action = policy.act(state, True)
        action = action.data.numpy()
        for j in range(simrate):
            target = action + offset
            real_action = action

            cassie_env.step_simulation(real_action)
            curr_qpos = cassie_env.sim.qpos()
            curr_qvel = cassie_env.sim.qvel()
            curr_foot = np.concatenate((cassie_env.cassie_state.leftFoot.position, cassie_env.cassie_state.rightFoot.position))
            curr_foot += np.concatenate((cassie_env.cassie_state.pelvis.position, cassie_env.cassie_state.pelvis.position))
            state_est_foot[i*simrate+j, :] = curr_foot
            mj_foot = np.zeros(6)
            cassie_env.sim.foot_pos(mj_foot)
            mj_foot_pos[i*simrate+j, :] = mj_foot
            foot_vel[i*simrate+j, :] = [cassie_env.lfoot_vel, cassie_env.rfoot_vel]

            # foot_vel[i*60+j, :] = np.concatenate((cassie_env.cassie_state.leftFoot.footTranslationalVelocity, cassie_env.cassie_state.rightFoot.footTranslationalVelocity))

        cassie_env.time  += 1
        cassie_env.phase += cassie_env.phase_add

        if cassie_env.phase > cassie_env.phaselen:
            cassie_env.phase = 0
            cassie_env.counter += 1

        state = cassie_env.get_full_state()
        state = torch.Tensor(state)

# Graph foot pos data
# fig, ax = plt.subplots(6, 2, figsize=(6, 12), sharex=True, sharey='row')
# t = np.linspace(0, num_steps*60*0.0005, num_steps*60)
# ax[5][0].set_xlabel("Time (sec)")
# ax[5][1].set_xlabel("Time (sec)")
# sides = ["Left", "Right"]
# titles = [" Foot X Position", " Foot Y Position", " Foot Z Position"]
# labels = ["State Est", "MuJoCo"]
# data = [state_est_foot, mj_foot_pos]
# for i in range(2):
#     # ax[i][0].set_ylabel(labels[i])
#     ax[0][i].set_title(labels[i])
#     for j in range(2):
#         for k in range(3):
#             ax[3*j+k][i].plot(t, data[i][:, 3*j+k])
#             ax[3*j+k][i].set_ylabel(sides[j] + titles[k])
#             # ax[1][i].plot(t, mj_foot_pos[:, 3*i+2])
#             # ax[1][i].set_title(sides[i] + " mj foot z pos")
#             # ax[1][i].set_ylabel("Z Position (m)")
#     # for j in range(3):
#     #     ax[j+2][i].plot(t, foot_vel[:, 3*i+j])
#     #     ax[j+2][i].set_title(sides[i] + titles[j+1])
#     #     ax[j+2][i].set_ylabel("Velocity (m/s)")    

# plt.tight_layout()
# plt.savefig("./foot_compare.png")
# exit()

# Graph foot traj pos data
total_traj = np.concatenate((foot_traj.lfoot, foot_traj.rfoot), axis=1)
for i in range(1, len(foot_traj)):
    traj_vel[i, :] = (total_traj[i, :] - total_traj[i-1, :]) / (0.4 / (841 - int(1682 / 5)))
fig, ax = plt.subplots(2, 2, figsize=(12, 5), sharex=True)
t = np.linspace(0, num_steps*simrate*0.0005, num_steps*simrate)
sides = ["Left", "Right"]
ax[1][0].set_xlabel("Time (sec)")
ax[1][1].set_xlabel("Time (sec)")
for i in range(2):
    ax[i][0].set_ylabel("Position (m)")
    ax[i][0].set_title(sides[i] + " Foot Z Position")
    ax[i][0].plot(t, mj_foot_pos[:, 3*i+2], label="Policy")
    ax[i][0].plot(t, total_traj[:-2, 3*i+2], label="Ref Traj")
    ax[i][0].legend()

for i in range(2):
    ax[i][1].set_ylabel("Velocity (m/2)")
    ax[i][1].set_title(sides[i] + " Foot Z Velocity")
    ax[i][1].plot(t, foot_vel[:, i], label="Policy")
    ax[i][1].plot(t, traj_vel[:-2, 3*i+2], label="Ref Traj")
    ax[i][1].legend()

# titles = [" Foot X Position", " Foot Y Position", " Foot Z Position"]
# labels = ["Policy", "Ref Traj"]
# print("total traj: ", total_traj.shape)
# for i in range(3):
#     ax[1][i].set_xlabel("Time (sec)")
# for i in range(2):
#     ax[i][0].set_ylabel("Position (m)")
#     for j in range(3):
#         ax[i][j].set_title(sides[i] + titles[j])
#         ax[i][j].plot(t, mj_foot_pos[:, 3*i+j], label="Policy")
#         ax[i][j].plot(t, total_traj[:-2, 3*i+j], label="Ref Traj")

# plt.legend()
plt.tight_layout()
plt.savefig("./foot_traj_compare.png")