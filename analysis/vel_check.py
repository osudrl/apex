import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt
import time, os, sys

from cassie.cassiemujoco.cassieUDP import *
from cassie.cassiemujoco.cassiemujoco import *

import argparse
import pickle

def sg_filter(x, m, k=0):
    """
    x = Vector of sample times
    m = Order of the smoothing polynomial
    k = Which derivative
    """
    mid = int(len(x) / 2)
    a = x - x[mid]
    expa = lambda x: map(lambda i: i**x, a)    
    A = np.r_[map(expa, range(0,m+1))].transpose()
    Ai = np.linalg.pinv(A)

    return Ai[1]

def smooth(x, y, size=5, order=2, deriv=0):

    n = len(x)
    m = size

    result = np.zeros(n)

    for i in xrange(m, n-m):
        start, end = i - m, i + m + 1
        f = sg_filter(x[start:end], order, deriv)
        result[i] = np.dot(f, y[start:end])

    return result

def vel_est(t, y, order=2):
    p = np.polyfit(t, y, order)
    x = t[-1]
    vel = 0
    for i in range(len(p)-1):
        vel += p[i]*x**(order-i-1)
    return vel

# t = [0, 0.1, 0.2, 0.3, 0.4]
# y = [0, 0.2, 0.6, 1.2, 2.0]
# vel_est(t, y, order=5)
# exit()

cassie_sim = CassieSim("./cassie/cassiemujoco/cassie.xml")

P = np.array([100,  100,  88,  96,  50])
D = np.array([10.0, 10.0, 8.0, 9.6, 5.0])
P[0] *= 0.8
D[0] *= 0.8
P[1] *= 0.8
D[1] *= 0.8

zero_u = pd_in_t()
nominal_u = pd_in_t()
nominal_targ = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])

motor_pos = np.zeros((10000, 10))
motor_vel = np.zeros((10000, 10))
joint_vel = np.zeros((10000, 4))
mj_pos = np.zeros((10000, 10))
mj_vel = np.zeros((10000, 10))
mj_joint_vel = np.zeros((10000, 4))
torque = np.zeros((10000, 10))
window_size = 3
order = 2
t = np.linspace(0, 0.0005*(window_size - 1), window_size)
print("t", t)
filter_vel = np.zeros((10000, 10))
pos_idx = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
vel_idx = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]

# pos_idx = [0, 1, 2, 7, 13, 14, 15, 16, 21, 27]
# vel_idx = [0, 1, 2, 6, 12, 13, 14, 15, 19, 25]

for i in range(5):
    zero_u.leftLeg.motorPd.pGain[i] = 0.0
    zero_u.leftLeg.motorPd.dGain[i] = 0.0
    zero_u.rightLeg.motorPd.pGain[i] = 0.0
    zero_u.rightLeg.motorPd.dGain[i] = 0.0
    zero_u.leftLeg.motorPd.pTarget[i] = 0.0
    zero_u.rightLeg.motorPd.pTarget[i] = 0.0
    zero_u.leftLeg.motorPd.dTarget[i]  = 0
    zero_u.rightLeg.motorPd.dTarget[i] = 0
    zero_u.leftLeg.motorPd.torque[i] = 0.0
    zero_u.rightLeg.motorPd.torque[i] = 0.0

    nominal_u.leftLeg.motorPd.pGain[i] = P[i]
    nominal_u.leftLeg.motorPd.dGain[i] = D[i]
    nominal_u.rightLeg.motorPd.pGain[i] = P[i]
    nominal_u.rightLeg.motorPd.dGain[i] = D[i]
    nominal_u.leftLeg.motorPd.pTarget[i] = nominal_targ[i]
    nominal_u.rightLeg.motorPd.pTarget[i] = nominal_targ[i+5]
    nominal_u.leftLeg.motorPd.dTarget[i]  = 0
    nominal_u.rightLeg.motorPd.dTarget[i] = 0
    nominal_u.leftLeg.motorPd.torque[i] = 0.0
    nominal_u.rightLeg.motorPd.torque[i] = 0.0

for i in range(6000):
    # cassie_sim.set_qvel(np.zeros(32))
    cassie_sim.step_pd(zero_u)
    print("zero time {}".format(i), end="\r")

# cassie_sim.set_qvel(np.zeros(32))

for i in range(10000):
    state = cassie_sim.step_pd(nominal_u)
    motor_pos[i, :] = state.motor.position[:]
    motor_vel[i, :] = state.motor.velocity[:]
    joint_vel[i, :] = np.concatenate([state.joint.velocity[0:2], state.joint.velocity[3:5]])
    torque[i, :] = state.motor.torque[:]
    curr_qpos = np.array(cassie_sim.qpos())
    curr_qvel = np.array(cassie_sim.qvel())
    mj_pos[i, :] = curr_qpos[pos_idx]
    mj_vel[i, :] = curr_qvel[vel_idx]
    mj_joint_vel[i, :] = curr_qvel[[10, 11, 17, 18]]
    if i >= window_size:
        for j in range(10):
            filter_vel[i, j] = vel_est(t, mj_pos[i-window_size:i, j], order=order)

    print("motor time {}".format(i), end="\r")

motor_pos_change = motor_pos[1:, :] - motor_pos[0:-1, :]



start_ind = 8000
end_ind = 8200#10000
print("min change:", np.max(np.abs(motor_pos_change[start_ind:end_ind, :]), axis=0))


fig, ax = plt.subplots(2, 5, figsize=(15, 5))
time = np.linspace(0, 10000*0.0005, 10000)
titles = ["Hip Roll", "Hip Yaw", "Hip Pitch", "Knee", "Foot"]
ax[0][0].set_ylabel("Pos [rad]")
ax[1][0].set_ylabel("Pos [rad]")

for i in range(5):
    ax[0][i].scatter(time[start_ind:end_ind], motor_pos[start_ind:end_ind, i], label="est")
    ax[0][i].set_ylim(np.min(motor_pos[start_ind:end_ind, i]), np.max(motor_pos[start_ind:end_ind, i]))
    print(np.min(motor_pos[start_ind:end_ind, i]), np.max(motor_pos[start_ind:end_ind, i]))
    # ax[0][i].plot(time[start_ind:end_ind], motor_pos[start_ind:end_ind, i], label="est", color="C0")
    ax[0][i].scatter(time[start_ind:end_ind], mj_pos[start_ind:end_ind, i], label="mj")
    # ax[0][i].plot(time[start_ind:end_ind:50], motor_pos[start_ind:end_ind:50, i], label="est_sub")
    ax[0][i].set_title("Left " + titles[i])
    ax[0][i].legend()

    ax[1][i].plot(time[start_ind:end_ind], motor_pos[start_ind:end_ind, i+5], label="est")
    ax[1][i].plot(time[start_ind:end_ind], mj_pos[start_ind:end_ind, i+5], label="mj")
    # ax[1][i].plot(time[start_ind:end_ind:50], motor_pos[start_ind:end_ind:50, i+5], label="est_sub")
    ax[1][i].set_title("Right " + titles[i])
    ax[1][i].legend()
    ax[1][i].set_xlabel("Time (sec)")

fig.suptitle("Motor Pos")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

fig, ax = plt.subplots(2, 5, figsize=(15, 5))
time = np.linspace(0, 10000*0.0005, 10000)
titles = ["Hip Roll", "Hip Yaw", "Hip Pitch", "Knee", "Foot"]
ax[0][0].set_ylabel("Vel [rad/s]")
ax[1][0].set_ylabel("Vel [rad/s]")

for i in range(5):
    ax[0][i].plot(time[start_ind:end_ind], motor_vel[start_ind:end_ind, i], label="est")
    ax[0][i].plot(time[start_ind:end_ind], mj_vel[start_ind:end_ind, i], label="mj")
    ax[0][i].plot(time[start_ind:end_ind], filter_vel[start_ind:end_ind, i], label="my est")
    # ax[0][i].plot(time[start_ind:end_ind:50], motor_vel[start_ind:end_ind:50, i], label="est_sub")
    ax[0][i].set_title("Left " + titles[i])
    ax[0][i].legend()

    ax[1][i].plot(time[start_ind:end_ind], motor_vel[start_ind:end_ind, i+5], label="est")
    ax[1][i].plot(time[start_ind:end_ind], mj_vel[start_ind:end_ind, i+5], label="mj")
    ax[1][i].plot(time[start_ind:end_ind], filter_vel[start_ind:end_ind, i+5], label="my est")
    # ax[1][i].plot(time[start_ind:end_ind:50], motor_vel[start_ind:end_ind:50, i+5], label="est_sub")
    ax[1][i].set_title("Right " + titles[i])
    ax[1][i].legend()
    ax[1][i].set_xlabel("Time (sec)")

fig.suptitle("Motor Vel")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

fig, ax = plt.subplots(2, 2, figsize=(15, 5))
time = np.linspace(0, 10000*0.0005, 10000)
titles = ["Shin", "Tarsus"]
ax[0][0].set_ylabel("Vel [rad/s]")
ax[1][0].set_ylabel("Vel [rad/s]")

for i in range(2):
    ax[0][i].plot(time[start_ind:end_ind], joint_vel[start_ind:end_ind, i], label="est")
    ax[0][i].plot(time[start_ind:end_ind], mj_joint_vel[start_ind:end_ind, i], label="mj")
    ax[0][i].set_title("Left " + titles[i])
    ax[0][i].legend()

    ax[1][i].plot(time[start_ind:end_ind], joint_vel[start_ind:end_ind, i+2], label="est")
    ax[1][i].plot(time[start_ind:end_ind], mj_joint_vel[start_ind:end_ind, i+2], label="mj")
    ax[1][i].set_title("Right " + titles[i])
    ax[1][i].legend()
    ax[1][i].set_xlabel("Time (sec)")

fig.suptitle("Joint Vel")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

fig, ax = plt.subplots(2, 5, figsize=(15, 5))
time = np.linspace(0, 10000*0.0005, 10000)
titles = ["Hip Roll", "Hip Yaw", "Hip Pitch", "Knee", "Foot"]
ax[0][0].set_ylabel("Torque [N/m]")
ax[1][0].set_ylabel("Torque [N/m]")

for i in range(5):
    ax[0][i].plot(time[start_ind:end_ind], torque[start_ind:end_ind, i])
    ax[0][i].set_title("Left " + titles[i])

    ax[1][i].plot(time[start_ind:end_ind], torque[start_ind:end_ind, i+5])
    ax[1][i].set_title("Right " + titles[i])
    ax[1][i].set_xlabel("Time (sec)")

fig.suptitle("Motor Torque")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()