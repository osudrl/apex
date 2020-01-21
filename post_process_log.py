import pickle
from matplotlib import pyplot as plt
import numpy as np
import cassie
import time
from tempfile import TemporaryFile


POLICY_NAME = "aslip_unified_0_v5"
FILE_PATH = "./hardware_logs/"
FILE_NAME = "2020-01-20_16:51_log0"

logs = pickle.load(open(FILE_PATH + POLICY_NAME + "/" + FILE_NAME + ".pkl", "rb")) #load in file with cassie data

# data = {"time": time_log, "output": output_log, "input": input_log, "state": state_log, "target_torques": target_torques_log,\
# "target_foot_residual": target_foot_residual_log}

time = logs["time"]
states_rl = logs["input"]
states = logs["state"]
nn_output = logs["output"]
trajectory_steps = logs["trajectory"]

numStates = len(states)
motors_log      = np.zeros((numStates, 10))
joints_log      = np.zeros((numStates, 6))
torques_mea_log = np.zeros((numStates, 10))
left_foot_forces_log = np.zeros((numStates, 6))
right_foot_forces_log = np.zeros((numStates, 6))
left_foot_pos_log = np.zeros((numStates, 6))
right_foot_pos_log = np.zeros((numStates, 6))
trajectory_log = np.zeros((numStates, 10))

j=0
for s in states:
    motors_log[j, :] = s.motor.position[:]
    joints_log[j, :] = s.joint.position[:]
    torques_mea_log[j, :] = s.motor.torque[:]
    left_foot_forces_log[j, :] = np.reshape(np.asarray([s.leftFoot.toeForce[:],s.leftFoot.heelForce[:]]), (6))
    right_foot_forces_log[j, :] = np.reshape(np.asarray([s.rightFoot.toeForce[:],s.rightFoot.heelForce[:]]), (6))
    left_foot_pos_log[j, :] = np.reshape(np.asarray([s.leftFoot.position[:],s.leftFoot.position[:]]), (6))
    right_foot_pos_log[j, :] = np.reshape(np.asarray([s.rightFoot.position[:],s.rightFoot.position[:]]), (6))
    
    trajectory_log[j, :] = trajectory_steps[j][:]

    j += 1

# j = 0
# for t in trajectory_steps:
#     trajectory_log[j, :] = t[:]
#     j += 1

SAVE_NAME = FILE_PATH + POLICY_NAME + "/" + FILE_NAME + '.npz'
np.savez(SAVE_NAME, time = time, motor = motors_log, joint = joints_log, torques_measured=torques_mea_log, left_foot_force = left_foot_forces_log, right_foot_force = right_foot_forces_log, left_foot_pos = left_foot_pos_log, right_foot_pos = right_foot_pos_log, trajectory = trajectory_log)
