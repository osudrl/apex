import numpy as np
import matplotlib.pyplot as plt
from tempfile import TemporaryFile

# from cassie_env import CassieEnv
from trajectory.trajectory import CassieTrajectory
#from mujoco.cassiemujoco import *
import time as t
traj = CassieTrajectory("/home/robot/Desktop/apex/cassie/trajectory/stepdata.bin")
# env = CassieEnv("walking")

qpos_traj = traj.qpos
time_traj = traj.time

tt = traj.time
#u = pd_in_t()

# load your data
data = np.load('cassie/outfile.npz')
motor = data['motor']
joint = data['joint']
qpos = data['qpos_replay']
time = data['time']

delt_t = time[4] - time[3]
delt_t_traj = time_traj[4] - time_traj[3]
same_time = delt_t / delt_t_traj
time_traj = time_traj * same_time

#time = time * (60/2000)
numStates = len(qpos)

# np.savetxt("test_arr.txt", qpos[0:1000, 34])
print("Made it")
# test actual trajectory

rand = np.random.randint(1, 101, 1000)

#log data
plt.subplot(2,2,1)
plt.plot(time[0:500], motor[0:500,4], 'r')
plt.plot(time[0:500], motor[0:500, 9], 'k')

plt.subplot(2,2,2)
plt.plot(time[1200:1300], joint[1200:1300,2], 'r')
plt.plot(time[1200:1300], joint[1200:1300, 5], 'k')

plt.subplot(2,2,3)
plt.plot(time[1200:1300], qpos[1200:1300,20], 'r')
plt.plot(time[1200:1300], qpos[1200:1300, 34], 'k')

#trajectory data
plt.subplot(2,2,4)
plt.plot(time_traj[:], qpos_traj[:,20], 'r')
plt.plot(time_traj[:], qpos_traj[:, 34], 'k')
plt.show()

#trajectory data

plt.plot(tt[:], qpos_traj[:,32] + qpos_traj[:, 33], 'r')
# plt.plot(tt[:], qpos_traj[:,19], 'b')
# plt.plot(tt[:], qpos_traj[:, 20], 'k')
plt.show()