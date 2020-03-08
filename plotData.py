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

def avg_pols(policies, state):
    total_act = np.zeros(10)
    for policy in policies:
        _, action = policy.act(state, True)
        total_act += action.data.numpy()
    return total_act / len(policies)

# Load environment and policy
# cassie_env = CassieEnv("walking", clock_based=True, state_est=False)
# cassie_env = CassieEnv_nodelta("walking", clock_based=True, state_est=False)
# cassie_env = CassieEnv_speed("walking", clock_based=True, state_est=True)
# cassie_env = CassieEnv_speed_dfreq("walking", clock_based=True, state_est=False)
# cassie_env = CassieEnv_speed_no_delta_neutral_foot("walking", clock_based=True, state_est=True)
cassie_env = CassieEnv_speed_sidestep("walking", simrate = 60, clock_based=True, state_est=True)
# cassie_env = CassieEnv_stand(state_est=False)

obs_dim = cassie_env.observation_space.shape[0] # TODO: could make obs and ac space static properties
action_dim = cassie_env.action_space.shape[0]

do_multi = False
no_delta = True
limittargs = False
lininterp = False
offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])

# file_prefix = "5k_footorient_smoothcost_jointreward_randjoint_seed10"
# file_prefix = "sidestep_StateEst_speedmatch_(3)"#_actpenalty_retrain"
file_prefix = "sidestep_StateEst_speedmatch_torquesmoothcost"
# file_prefix = "nodelta_neutral_StateEst_symmetry_speed0-3_freq1-2"
policy = torch.load("./trained_models/{}.pt".format(file_prefix))
policy.bounded = False
# policy = torch.load("./trained_models/nodelta_neutral_StateEst_symmetry_speed0-3_freq1-2.pt")
policy.eval()

policies = []
if do_multi:
    # for i in range(5, 12):
    #     policy = torch.load("./trained_models/regular_spring"+str(i)+".pt")
    #     policy.eval()
    #     policies.append(policy)
    # policy = torch.load("./trained_models/Normal.pt")
    # policy.eval()
    # policies.append(policy)
    # policy = torch.load("./trained_models/stiff_StateEst_step.pt")
    # policy.eval()
    # policies.append(policy)
    for i in [10, 20, 30, 40, 50]:
        policy = torch.load("./trained_models/5k_randfric_footorient_yvel_seed{}.pt".format(i))
        policy.eval()
        policies.append(policy)

num_steps = 100
pre_steps = 300
simrate = 60
torques = np.zeros((num_steps*simrate, 10))
GRFs = np.zeros((num_steps*simrate, 2))
targets = np.zeros((num_steps*simrate, 10))
heights = np.zeros(num_steps*simrate)
speeds = np.zeros(num_steps*simrate)
foot_pos = np.zeros((num_steps*simrate, 6))
mj_foot_pos = np.zeros((num_steps*simrate, 6))
foot_vel = np.zeros((num_steps*simrate, 6))
actions = np.zeros((num_steps*simrate, 10))
pelaccel = np.zeros(num_steps*simrate)
pelheight = np.zeros(num_steps*simrate)
act_diff = np.zeros(num_steps*simrate)
actuated_pos = np.zeros((num_steps*simrate, 10))
actuated_vel = np.zeros((num_steps*simrate, 10))
foot_orient_cost = np.zeros((num_steps*simrate, 2))
pel_orient_cost = np.zeros(num_steps*simrate)
straight_cost = np.zeros(num_steps*simrate)
forward_cost = np.zeros(num_steps*simrate)
torque_cost = np.zeros(num_steps*simrate)
smooth_cost = np.zeros(num_steps*simrate)
torque_diff = np.zeros((num_steps*simrate, 10))
yvel_cost = np.zeros(num_steps*simrate)
footheightvel_cost = np.zeros((num_steps*simrate, 2))
diff_foot_vel = np.zeros((num_steps*simrate, 6))
prev_action = None
prev_torques = None
prev_foot = None
pos_idx = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
vel_idx = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]
neutral_foot_orient = np.array([-0.24790886454547323, -0.24679713195445646, -0.6609396704367185, 0.663921021343526])
orient_targ = np.array([1, 0, 0, 0])
# Execute policy and save torques
with torch.no_grad():
    state = torch.Tensor(cassie_env.reset_for_test())
    cassie_env.speed = 0.5
    # cassie_env.side_speed = .2
    cassie_env.phase_add = 1
    for i in range(pre_steps):
        if not do_multi:
            _, action = policy.act(state, True)
            state, reward, done, _ = cassie_env.step(action.data.numpy())
        else:
            action = avg_pols(policies, state)
            state, reward, done, _ = cassie_env.step(action)
        state = torch.Tensor(state)
    for i in range(num_steps):
        if not do_multi:
            _, action = policy.act(state, True)
            action = action.data.numpy()
        else:
            action = avg_pols(policies, state)
            # state, reward, done, _ = cassie_env.step(action)
        # targets[i, :] = action
        lin_steps = int(60 * 3/4)  # Number of steps to interpolate over. Should be between 0 and self.simrate
        alpha = 1 / lin_steps
        cassie_env.speed = 0.5
        for j in range(simrate):
            if no_delta:
                target = action + offset
            else:
                ref_pos, ref_vel = cassie_env.get_ref_state(cassie_env.phase + cassie_env.phase_add)
                target = action + ref_pos[cassie_env.pos_idx]
            if limittargs:
                h = 0.0001
                Tf = 1.0 / 300.0
                alpha = h / (Tf + h)
                real_action = (1-alpha)*cassie_env.prev_action + alpha*target
                actions[i*simrate+j, :] = real_action
            elif lininterp:  
                if prev_action is not None:
                    real_action = (1-alpha)*prev_action + alpha*action
                    if alpha < 1:
                        alpha += 1 / lin_steps
                    else:
                        alpha = 1
                else:
                    real_action = action
                actions[i*simrate+j, :] = real_action
            else:
                real_action = action
                actions[i*simrate+j, :] = action
            targets[i*simrate+j, :] = target

            cassie_env.step_simulation(real_action)
            curr_qpos = cassie_env.sim.qpos()
            curr_qvel = cassie_env.sim.qvel()
            torques[i*simrate+j, :] = cassie_env.cassie_state.motor.torque[:]
            GRFs[i*simrate+j, :] = cassie_env.sim.get_foot_forces()
            heights[i*simrate+j] = curr_qpos[2]
            speeds[i*simrate+j] = cassie_env.sim.qvel()[0]
            curr_foot = np.concatenate((cassie_env.cassie_state.leftFoot.position, cassie_env.cassie_state.rightFoot.position))
            curr_foot += np.concatenate((cassie_env.cassie_state.pelvis.position, cassie_env.cassie_state.pelvis.position))
            mj_foot = np.zeros(6)
            cassie_env.sim.foot_pos(mj_foot)
            mj_foot_pos[i*simrate+j, :] = mj_foot
            foot_pos[i*simrate+j, :] = curr_foot
            # print("left foot height: ", cassie_env.cassie_state.leftFoot.position[2])
            foot_vel[i*simrate+j, :] = np.concatenate((cassie_env.cassie_state.leftFoot.footTranslationalVelocity, cassie_env.cassie_state.rightFoot.footTranslationalVelocity))
            pelaccel[i*simrate+j] = cassie_env.cassie_state.pelvis.translationalAcceleration[2]#np.linalg.norm(cassie_env.cassie_state.pelvis.translationalAcceleration)
            pelheight[i*simrate+j] = cassie_env.cassie_state.pelvis.position[2]
            actuated_pos[i*simrate+j, :] = [curr_qpos[k] for k in pos_idx]
            actuated_vel[i*simrate+j, :] = [curr_qvel[k] for k in vel_idx]
            if prev_action is not None:
                act_diff[i*simrate+j] = np.linalg.norm(action - prev_action)
            else:
                act_diff[i*simrate+j] = 0
            if prev_foot is not None:
                diff_foot_vel[i*simrate+j, :] = (mj_foot - prev_foot) / 0.0005
            else: 
                diff_foot_vel[i*simrate+j, :] = np.zeros(6)
            # Calculate costs for reward
            foot_orient_cost[i*simrate+j, :] = 100*(1 - np.array([np.inner(neutral_foot_orient, cassie_env.sim.xquat("left-foot")) ** 2, np.inner(neutral_foot_orient, cassie_env.sim.xquat("right-foot")) ** 2]))
            pel_orient_cost[i*simrate+j] = (1 - np.inner(orient_targ, curr_qpos[3:7]) ** 2)
            forward_cost[i*simrate+j] = np.abs(curr_qvel[0] - cassie_env.speed)
            yvel_cost[i*simrate+j] = np.abs(curr_qvel[1] - 0)
            if forward_cost[i*simrate+j] < 0.05:
                forward_cost[i*simrate+j] = 0
            straight_cost[i*simrate+j] = 8*np.abs(curr_qpos[1] - 0)
            if np.abs(curr_qpos[1] - 0) < 0.05:
                straight_cost[i*simrate+j] = 0
            if (1 - np.inner(orient_targ, curr_qpos[3:7]) ** 2) < 5e-3:
                pel_orient_cost[i*simrate+j] = 0
            else:
                pel_orient_cost[i*simrate+j] *= 30
            if yvel_cost[i*simrate+j] < 0.05:
                yvel_cost[i*simrate+j] = 0
            torque_cost[i*simrate+j] = 0.00004*np.linalg.norm(np.square(torques[i*simrate+j]))
            if prev_torques is not None:
                torque_diff[i*simrate+j, :] = torques[i*simrate+j] - prev_torques
                # smooth_cost[i*simrate+j] = 0.00075*np.linalg.norm(np.square(torques[i*simrate+j] - prev_torques))
            else:
                torque_diff[i*simrate+j, :] = np.zeros(10)
                # smooth_cost[i*simrate+j] = 0
            smooth_cost[i*simrate+j] = 0.0001*np.linalg.norm(np.square(torque_diff[i*simrate+j, :]))
            prev_torques = torques[i*simrate+j]
            footheightvel_cost[i*simrate+j, 0] = 20 * (0.2 - mj_foot[2])**2 * np.linalg.norm(diff_foot_vel[i*simrate+j, 0:2])
            footheightvel_cost[i*simrate+j, 1] = 20 * (0.2 - mj_foot[5])**2 * np.linalg.norm(diff_foot_vel[i*simrate+j, 3:5])
            prev_foot = mj_foot

        prev_action = action

        cassie_env.time  += 1
        cassie_env.phase += cassie_env.phase_add

        if cassie_env.phase > cassie_env.phaselen:
            cassie_env.phase = 0
            cassie_env.counter += 1

        state = cassie_env.get_full_state()
        state = torch.Tensor(state)

# Graph torque data
fig, ax = plt.subplots(2, 5, figsize=(15, 5))
t = np.linspace(0, num_steps-1, num_steps*simrate)
titles = ["Hip Roll", "Hip Yaw", "Hip Pitch", "Knee", "Foot"]
ax[0][0].set_ylabel("Torque")
ax[1][0].set_ylabel("Torque")
for i in range(5):
    ax[0][i].plot(t, torques[:, i])
    ax[0][i].set_title("Left " + titles[i])
    ax[1][i].plot(t, torques[:, i+5])
    ax[1][i].set_title("Right " + titles[i])
    ax[1][i].set_xlabel("Timesteps (0.03 sec)")

plt.tight_layout()
plt.savefig("./apex_plots/"+file_prefix+"_torques.png")

# Graph GRF data
fig, ax = plt.subplots(2, figsize=(10, 5))
t = np.linspace(0, num_steps-1, num_steps*simrate)
ax[0].set_ylabel("GRFs")

ax[0].plot(t, GRFs[:, 0])
ax[0].set_title("Left Foot")
ax[0].set_xlabel("Timesteps (0.03 sec)")
ax[1].plot(t, GRFs[:, 1])
ax[1].set_title("Right Foot")
ax[1].set_xlabel("Timesteps (0.03 sec)")

plt.tight_layout()
plt.savefig("./apex_plots/"+file_prefix+"_GRFs.png")

# Graph PD target data
fig, ax = plt.subplots(2, 5, figsize=(15, 5))
t = np.linspace(0, num_steps-1, num_steps*simrate)
titles = ["Hip Roll", "Hip Yaw", "Hip Pitch", "Knee", "Foot"]
ax[0][0].set_ylabel("PD Target")
ax[1][0].set_ylabel("PD Target")
for i in range(5):
    ax[0][i].plot(t, targets[:, i])
    ax[0][i].set_title("Left " + titles[i])
    ax[1][i].plot(t, targets[:, i+5])
    ax[1][i].set_title("Right " + titles[i])
    ax[1][i].set_xlabel("Timesteps (0.03 sec)")

plt.tight_layout()
plt.savefig("./apex_plots/"+file_prefix+"_targets.png")

# Graph action data
fig, ax = plt.subplots(2, 5, figsize=(15, 5))
t = np.linspace(0, num_steps-1, num_steps*simrate)
titles = ["Hip Roll", "Hip Yaw", "Hip Pitch", "Knee", "Foot"]
ax[0][0].set_ylabel("Action")
ax[1][0].set_ylabel("Action")
for i in range(5):
    ax[0][i].plot(t, actions[:, i])
    ax[0][i].set_title("Left " + titles[i])
    ax[1][i].plot(t, actions[:, i+5])
    ax[1][i].set_title("Right " + titles[i])
    ax[1][i].set_xlabel("Timesteps (0.03 sec)")

plt.tight_layout()
plt.savefig("./apex_plots/"+file_prefix+"_actions.png")

# Graph state data
fig, ax = plt.subplots(2, 2, figsize=(10, 5))
t = np.linspace(0, num_steps-1, num_steps*simrate)
ax[0][0].set_ylabel("norm")
ax[0][0].plot(t, pelaccel[:])
ax[0][0].set_title("Pel Z Accel")
ax[0][1].set_ylabel("m/s")
ax[0][1].plot(t, np.linalg.norm(torques, axis=1))
ax[0][1].set_title("Torque Norm")
titles = ["Left", "Right"]
for i in range(2):
    ax[1][i].plot(t, mj_foot_pos[:, 3*i+2])
    ax[1][i].set_title(titles[i] + " Foot")
    ax[1][i].set_xlabel("Timesteps (0.03 sec)")

plt.tight_layout()
plt.savefig("./apex_plots/"+file_prefix+"_state.png")

# Graph feet qpos data
fig, ax = plt.subplots(5, 2, figsize=(12, 6), sharex=True, sharey='row')
t = np.linspace(0, num_steps*60*0.0005, num_steps*simrate)
ax[3][0].set_xlabel("Time (sec)")
ax[3][1].set_xlabel("Time (sec)")
sides = ["Left", "Right"]
titles = [" Foot Z Position", " Foot X Velocity", " Foot Y Velocity", " Foot Z Velocity"]
for i in range(2):
    # ax[0][i].plot(t, foot)
    ax[0][i].plot(t, foot_pos[:, 3*i+2])
    ax[0][i].set_title(sides[i] + titles[0])
    ax[0][i].set_ylabel("Z Position (m)")
    ax[1][i].plot(t, mj_foot_pos[:, 3*i+2])
    ax[1][i].set_title(sides[i] + " mj foot z pos")
    ax[1][i].set_ylabel("Z Position (m)")
    for j in range(3):
        ax[j+2][i].plot(t, diff_foot_vel[:, 3*i+j])
        ax[j+2][i].set_title(sides[i] + titles[j+1])
        ax[j+2][i].set_ylabel("Velocity (m/s)")    

plt.tight_layout()
plt.savefig("./apex_plots/"+file_prefix+"_feet.png")

# Graph phase portrait for actuated joints
fig, ax = plt.subplots(1, 5, figsize=(15, 4))
titles = ["Hip Roll", "Hip Yaw", "Hip Pitch", "Knee", "Foot"]
ax[0].set_ylabel("Velocity")
# ax[1][0].set_ylabel("Velocity")
for i in range(5):
    ax[i].plot(actuated_pos[:, i], actuated_vel[:, i])
    ax[i].plot(actuated_pos[:, i+5], actuated_vel[:, i+5])
    ax[i].set_title(titles[i])
    # ax[1][i].plot(actuated_pos[:, i+5], actuated_vel[:, i+5])
    # ax[1][i].set_title("Right " + titles[i])
    ax[i].set_xlabel("Angle")

plt.tight_layout()
plt.savefig("./apex_plots/"+file_prefix+"_phaseportrait.png")

# Misc Plotting
fig, ax = plt.subplots()
t = np.linspace(0, num_steps-1, num_steps*simrate)
# ax.set_ylabel("norm")
# ax.set_title("Action - Prev Action Norm")
# ax.plot(t, act_diff)
ax.set_ylabel("Height (m)")
ax.set_title("Pelvis Height")
ax.plot(t, pelheight)
plt.savefig("./apex_plots/"+file_prefix+"_misc.png")

# Cost Plotting
fig, ax = plt.subplots(5, 2, figsize=(15, 8))
t = np.linspace(0, num_steps*simrate*0.0005, num_steps*simrate)
# ax.set_ylabel("norm")
# ax.set_title("Action - Prev Action Norm")
# ax.plot(t, act_diff)
ax[0][0].plot(t, forward_cost)
ax[0][0].set_title("Forward Cost")

ax[0][1].plot(t, pel_orient_cost)
ax[0][1].set_title("Pelvis Orient Cost")

ax[1][1].plot(t, yvel_cost)
ax[1][1].set_title("Y Velocity Cost")

ax[1][0].plot(t, straight_cost)
ax[1][0].set_title("Straight Cost")

ax[2][0].set_ylabel("Cost")
ax[2][0].set_title("Left Foot Orient Cost")
ax[2][0].plot(t, foot_orient_cost[:, 0])
ax[2][1].set_ylabel("Cost")
ax[2][1].set_title("Right Foot Orient Cost")
ax[2][1].plot(t, foot_orient_cost[:, 1])

ax[3][0].plot(t, torque_cost)
ax[3][0].set_title("Torque Cost")
ax[3][1].plot(t, smooth_cost)
ax[3][1].set_title("Smooth Cost")

ax[4][0].plot(t, footheightvel_cost[:, 0])
ax[4][0].set_title("Left Foot Height Vel Cost")
ax[4][1].plot(t, footheightvel_cost[:, 1])
ax[4][1].set_title("Right Foot Height Vel Cost")

plt.tight_layout()
plt.savefig("./apex_plots/"+file_prefix+"_costs.png")

# Torque diff plotting

fig, ax = plt.subplots(2, 5, figsize=(15, 6))
t = np.linspace(0, num_steps*simrate*0.0005, num_steps*simrate)

titles = ["Hip Roll", "Hip Yaw", "Hip Pitch", "Knee", "Foot"]
ax[0][0].set_ylabel("Torque Difference")
ax[1][0].set_ylabel("Torque Difference")
for i in range(5):
    ax[0][i].plot(t, torque_diff[:, i])
    ax[0][i].set_title("Left " + titles[i])
    ax[1][i].plot(t, torque_diff[:, i+5])
    ax[1][i].set_title("Right " + titles[i])
    ax[1][i].set_xlabel("Time (sec)")

plt.tight_layout()
plt.savefig("./apex_plots/"+file_prefix+"_torquediff.png")