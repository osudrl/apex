#File to take real time data from cassie and put into q_pos format
#Writes qpos_replay object to file "q_pos_replay.pickle"
#

import pickle
from matplotlib import pyplot as plt
import numpy as np
# import cassie
import time
import quaternion_function as qf
from tempfile import TemporaryFile

logs = pickle.load(open("cassie/hardware_logs/2019-11-18_13_31_nodelta_neutral_StateEst_symmetry_speed0_3_freq1_2_muTor_0_001_logfinal.pkl", "rb")) #load in file with cassie data

states = logs["state"]
time = logs["time"]
inputs = np.array(logs["input"])

# input_idx = np.arange(0,len(inputs))
# some = inputs[3, 48]
# plt.plot(input_idx, inputs[:,48])
# plt.show()

numStates = len(states)
numInputs = len(inputs)

qpos_replay = np.zeros((numStates, 35)) #set initial values to zero
motor = np.zeros((numStates, 10))
joint = np.zeros((numStates, 6))
inputs_out = np.zeros((numInputs, 49))
time_slice = time[4449:5067]
inputs_slice = inputs[4449:5067, :]

yay = np.sqrt(408**2 + 40**2)
t_init = time[0] #get initial time yes

j = 0
k = 0
l = 0
inv_pelvis = qf.inverse_quaternion(states[0].pelvis.orientation[:]) #Get inverse quaternian to ensure robot starts going straight
q_offset = qf.quaternion_product([1,0,0,0],inv_pelvis)


for s in states:
   
    time[j] = time[j] - t_init #start time at zero
   # vel.append(np.linalg.norm(s.pelvis.translationalVelocity[:]))
   # print(s.pelvis.position[1])
    
    #####################QPos object ########################    
    
    #Make sure robot is always facing forward by rotating by the initial offset
    ones = np.array([1])
    qv = np.concatenate((ones, s.pelvis.position[:]), axis=0)
    pv = qf.quaternion_product(q_offset, qv)  
    q_off_inv = qf.inverse_quaternion(q_offset)
    rot_pos = qf.quaternion_product(pv, q_off_inv)

    qpos_replay[j, 0:3] = rot_pos[1:4] # Pelvis X, Y, Z

    qpos_replay[j, 3:7] = qf.quaternion_product(q_offset, s.pelvis.orientation[:]) # Pelvis Orientation
    #Left side
    qpos_replay[j, 7:10] = s.motor.position[0:3]#double check this is correct!! (left hip roll, pitch, and yaw)
    qpos_replay[j, 10:14] = [0.966,0,0,-0.259] #######Still Do########
    qpos_replay[j, 14] = s.motor.position[3] #knee
    qpos_replay[j, 15:17] = s.joint.position[0:2] # shin and tarsus
    qpos_replay[j, 17] = 0 ######Still Do########
    qpos_replay[j, 18] = s.motor.position[4] + 0.11 #Set left foot crank w offset from foot
    qpos_replay[j, 19] = -qpos_replay[j, 18] - 0.0184 #Set left plantar rod w offset from foot crank
    qpos_replay[j, 20] = s.motor.position[4] # check if correct (Motor [4], Joint [2])
    #Right Side
    qpos_replay[j, 21:24] = s.motor.position[5:8] #double check (right hip roll, pitch, and yaw)
    qpos_replay[j, 24:28] = [0.966,0,0,-0.259] #######Still Do########
    qpos_replay[j, 28] = s.motor.position[8] #right knee
    qpos_replay[j, 29:31] = s.joint.position[3:5] # Right shin and tarsus
    qpos_replay[j, 31] = s.rightFoot.position[0] #######Still Do########
    qpos_replay[j, 32] = s.motor.position[9] + 0.11 #Set right foot crank w offset from foot
    qpos_replay[j, 33] = -qpos_replay[j, 32] - 0.0184 #Set right plantar rod w offset from foot crank
    qpos_replay[j, 34] = s.motor.position[9]

    ######################Motor Object#######################
    motor[j, 0:3] = qpos_replay[j, 7:10]
    motor[j, 3] = qpos_replay[j, 14]
    motor[j, 4:8] = qpos_replay[j, 20:24]
    motor[j, 8] = qpos_replay[j, 28]
    motor[j, 9] = qpos_replay[j, 34]

    ######################Joint Object#########################
    joint[j, 0:2] = qpos_replay[j, 15:17]
    joint[j, 2] = qpos_replay[j, 20]
    joint[j, 3:5] = qpos_replay[j, 29:31]
    joint[j, 5] = qpos_replay[j, 34]

    ######################Target Object#########################
    #target[j, 0:10] = 
   # pelvis_pos.append(s.pelvis.position[:])
    j = j + 1

for t in inputs_slice:
    if np.array_equal(inputs_slice[l, 46:48], [0,1]):
        zero_phase_index = l
        inputs_slice = inputs_slice[l:, :]
        time_slice = time_slice[l:]
        break
    l = l + 1
time = time_slice

for t in inputs_slice:
    inputs_out[k, 1:49] = t[1:49]
    k = k + 1

np.savez('outfile2.npz', qpos_replay = qpos_replay, time = time, motor = motor, joint = joint, inputs = inputs_out)
