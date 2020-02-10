import pickle
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cassie
import time
from tempfile import TemporaryFile


FILE_PATH = "./hardware_logs/aslip_unified_no_delta_80_TS_only_sim/"
FILE_NAME = "2020-01-27_10:26_logfinal"


logs = pickle.load(open(FILE_PATH + FILE_NAME + ".pkl", "rb")) #load in file with cassie data

# data = {"time": time_log, "output": output_log, "input": input_log, "state": state_log, "target_torques": target_torques_log,\
# "target_foot_residual": target_foot_residual_log}

time = logs["time"]
states_rl = np.array(logs["input"])
states = logs["state"]
nn_output = logs["output"]
trajectory_steps = logs["trajectory"]
speeds = logs["speed"]

numStates = len(states)
pelvis      = np.zeros((numStates, 3))
foot_pos_left = np.zeros((numStates, 6))
foot_pos_right = np.zeros((numStates, 6))
# trajectory_log = np.zeros((numStates, 10))

j=0
for s in states:
    pelvis[j, :] = s.pelvis.translationalVelocity[:]
    foot_pos_left[j, :] = np.reshape(np.asarray([s.leftFoot.position[:],s.leftFoot.position[:]]), (6))
    foot_pos_right[j, :] = np.reshape(np.asarray([s.rightFoot.position[:],s.rightFoot.position[:]]), (6))
    
    j += 1

# Save stuff for later
SAVE_NAME = FILE_PATH + FILE_NAME + '.npz'
# np.savez(SAVE_NAME, time = time, motor = motors, joint = joints, torques_measured=torques_mea, left_foot_force = ff_left, right_foot_force = ff_right, left_foot_pos = foot_pos_left, right_foot_pos = foot_pos_right, trajectory = trajectory_log)
np.savez(SAVE_NAME, time = time, pelvis=pelvis, left_foot_pos = foot_pos_left, right_foot_pos = foot_pos_right)

##########################################
# Plot everything (except for ref traj)
##########################################

ax1 = plt.subplot(1,1,1)
ax1.plot(time[:], speeds[:], label='speed command')
ax1.plot(time[:], states_rl[:,61], label='ROM COM x velocity')
ax1.plot(time[:], pelvis[:,0], label='pelvis x velocity')
ax1.set_xlabel('Time')
ax1.set_ylabel('m/s')
ax1.legend(loc='upper right')
ax1.set_title('Varying Vels')
plt.show()