from .cassiemujoco import pd_in_t, CassieSim, CassieVis

from .trajectory import CassieTrajectory

from math import floor

import numpy as np 
import os
import random

from scipy.interpolate import CubicSpline

#import pickle

def get_trajectory(peak, stride, phase, stance=0.15):
    """
    Foot and center of mass trajectory generator

    The center of mass trajectory is a line q(t) = (x(t), 0, z). 

    The foot trajectories are periodic 3 point cubic splines in z(t),
    lines in x(t), and constant in y => q(t) = (x(t), y, z(t))

    Left and right foot trajectories are offset in time by "phase" timesteps.
    """
    # TODO: figure out formula for how com_vel relates to foot trajectories...
    # is SLIP model necessary?
    com_vel = 0.02 

    t = np.arange(phase)

    m = stride / phase

    ts = np.array([0, phase / 2, phase - 1])
    zs = np.array([0, peak, 0])
    spline = CubicSpline(ts, zs)

    x = m * t
    z = spline(t)

    # left foot steps first
    x_l = np.concatenate((x, np.ones(phase) * x[-1]))
    y_l = np.ones(phase * 2) * -stance
    z_l = np.concatenate((z, np.zeros(phase)))

    # then right foot
    x_r = np.concatenate((np.ones(phase) * x[0] + stride / 2, x + stride / 2))
    y_r = np.ones(phase * 2) * stance
    z_r = np.concatenate((np.zeros(phase), z))

    x_com = np.cumsum(np.ones(phase * 2) * com_vel)
    y_com = np.zeros(phase * 2)
    z_com = np.ones(phase * 2) * 1


    return np.column_stack([x_l, y_l, z_l, 
                       x_r, y_r, z_r, 
                       x_com, y_com, z_com])


class CassieTSEnv:
    def __init__(self, simrate=60):
        self.sim = CassieSim()
        self.vis = None

        self.qpos0 = np.copy(self.sim.qpos())
        self.qvel0 = np.copy(self.sim.qvel())

        self.observation_space = np.zeros(51)
        self.action_space      = np.zeros(10)

        self.phaselen = 14

        self.task_trajectory = get_trajectory(peak=0.2, stride=0.6, phase=self.phaselen)

        self.P = np.array([100,  100,  88,  96,  50]) 
        self.D = np.array([10.0, 10.0, 8.0, 9.6, 5.0])

        self.u = pd_in_t()

        self.simrate = simrate # simulate X mujoco steps with same pd target
                               # 60 brings simulation from 2000Hz to roughly 30Hz

        self.time    = 0 # number of time steps in current episode
        self.phase   = 0 # portion of the phase the robot is in
        self.counter = 0 # number of phase cycles completed in episode


        # see include/cassiemujoco.h for meaning of these indices
        self.pos_idx = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        self.vel_idx = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]
    
    def step_simulation(self, action):
        target = action

        self.u = pd_in_t()
        for i in range(5):
            # maybe write a wrapper for pd_in_t ?
            self.u.leftLeg.motorPd.pGain[i]  = self.P[i]
            self.u.rightLeg.motorPd.pGain[i] = self.P[i]

            self.u.leftLeg.motorPd.dGain[i]  = self.D[i]
            self.u.rightLeg.motorPd.dGain[i] = self.D[i]

            self.u.leftLeg.motorPd.torque[i]  = 0 # Feedforward torque
            self.u.rightLeg.motorPd.torque[i] = 0 

            self.u.leftLeg.motorPd.pTarget[i]  = target[i]
            self.u.rightLeg.motorPd.pTarget[i] = target[i + 5]

            self.u.leftLeg.motorPd.dTarget[i]  = 0
            self.u.rightLeg.motorPd.dTarget[i] = 0

        self.sim.step_pd(self.u)

    def step(self, action):
        #print(action)
        for _ in range(self.simrate):
            self.step_simulation(action)

        height = self.sim.qpos()[2]

        self.time  += 1
        self.phase += 1

        if self.phase > self.phaselen:
            self.phase = 0
            self.counter += 1

        # Early termination
        done = not(height > 0.5 and height < 3.0)

        reward = self.compute_reward()

        # TODO: make 0.3 a variable/more transparent
        if reward < 0.3:
            done = True

        return self.get_full_state(), reward, done, {}

    def reset(self):
        #print("Reset")
        self.phase = 0 #random.randint(0, self.phaselen)
        self.time = 0
        self.counter = 0

        self.sim.set_qpos(self.qpos0)
        self.sim.set_qvel(self.qvel0)
        # self.sim.set_qpos(np.array([0.0045, 0, 0.4973, 0.9785, -0.0164, 0.01787, -0.2049,
        #  -1.1997, 0, 1.4267, 0, -1.5244, 1.5244, -1.5968,
        #  -0.0045, 0, 0.4973, 0.9786, 0.00386, -0.01524, -0.2051,
        #  -1.1997, 0, 1.4267, 0, -1.5244, 1.5244, -1.5968]))

        return self.get_full_state()


    # NOTE: this reward is slightly different from the one in Xie et al
    # see notes for details
    def compute_reward(self):
        qpos = np.copy(self.sim.qpos())

        com_pos = qpos[0:2]
        foot_pos = np.zeros(6)
        self.sim.foot_pos(foot_pos)

        target = self.get_ref_state(self.phase)

        foot_target = target[0:6]
        com_target  = target[6:9]

        foot_error = 0
        for i in range(len(foot_pos)):
            foot_error += 3 * (foot_pos[i] - foot_target[i]) ** 2

        com_error = 0
        for i in range(len(com_pos)):
            com_error += 10 * (com_pos[i] - com_target[i]) ** 2

        # <1,0,0,0>
        # print(self.qpos0[3:7])
        # quaternion distance
        orientation_error = np.arccos(2 * np.inner(qpos[3:7], self.qpos0[3:7]) ** 2 - 1)

        print(".", 0.4 * np.exp(-foot_error), 0.3 * np.exp(-com_error), 0.3 * np.exp(-orientation_error))

        reward = 0.4 * np.exp(-foot_error) + \
                 0.3 * np.exp(-com_error)  + \
                 0.3 * np.exp(-orientation_error)

        #print(foot_pos, foot_target)

        return reward

    # get the corresponding state from the reference trajectory for the current phase
    def get_ref_state(self, phase=None):
        if phase is None:
            phase = self.phase

        if phase > self.phaselen:
            phase = 0

        return self.task_trajectory[phase, :]

    def get_full_state(self):
        qpos = np.copy(self.sim.qpos())
        qvel = np.copy(self.sim.qvel()) 

        ref = self.get_ref_state(self.phase + 1)

        # TODO: maybe convert to set subtraction for clarity
        # {i for i in range(35)} - 
        # {0, 10, 11, 12, 13, 17, 18, 19, 24, 25, 26, 27, 31, 32, 33}

        # this is everything except pelvis x and qw, achilles rod quaternions, 
        # and heel spring/foot crank/plantar rod angles
        # note: x is forward dist, y is lateral dist, z is height

        # makes sense to always exclude x because it is in global coordinates and
        # irrelevant to phase-based control. Z is inherently invariant to
        # trajectory despite being global coord. Y is only invariant to straight
        # line trajectories.

        # [ 0] Pelvis y
        # [ 1] Pelvis z
        # [ 2] Pelvis orientation qw
        # [ 3] Pelvis orientation qx
        # [ 4] Pelvis orientation qy
        # [ 5] Pelvis orientation qz
        # [ 6] Left hip roll         (Motor [0])
        # [ 7] Left hip yaw          (Motor [1])
        # [ 8] Left hip pitch        (Motor [2])
        # [ 9] Left knee             (Motor [3])
        # [10] Left shin                        (Joint [0])
        # [11] Left tarsus                      (Joint [1])
        # [12] Left foot             (Motor [4], Joint [2])
        # [13] Right hip roll        (Motor [5])
        # [14] Right hip yaw         (Motor [6])
        # [15] Right hip pitch       (Motor [7])
        # [16] Right knee            (Motor [8])
        # [17] Right shin                       (Joint [3])
        # [18] Right tarsus                     (Joint [4])
        # [19] Right foot            (Motor [9], Joint [5])
        pos_index = np.array([1,2,3,4,5,6,7,8,9,14,15,16,20,21,22,23,28,29,30,34])

        # [ 0] Pelvis x
        # [ 1] Pelvis y
        # [ 2] Pelvis z
        # [ 3] Pelvis orientation wx
        # [ 4] Pelvis orientation wy
        # [ 5] Pelvis orientation wz
        # [ 6] Left hip roll         (Motor [0])
        # [ 7] Left hip yaw          (Motor [1])
        # [ 8] Left hip pitch        (Motor [2])
        # [ 9] Left knee             (Motor [3])
        # [10] Left shin                        (Joint [0])
        # [11] Left tarsus                      (Joint [1])
        # [12] Left foot             (Motor [4], Joint [2])
        # [13] Right hip roll        (Motor [5])
        # [14] Right hip yaw         (Motor [6])
        # [15] Right hip pitch       (Motor [7])
        # [16] Right knee            (Motor [8])
        # [17] Right shin                       (Joint [3])
        # [18] Right tarsus                     (Joint [4])
        # [19] Right foot            (Motor [9], Joint [5])
        vel_index = np.array([0,1,2,3,4,5,6,7,8,12,13,14,18,19,20,21,25,26,27,31])

        clock = [np.sin(2 * np.pi *  self.phase / self.phaselen),
                 np.cos(2 * np.pi *  self.phase / self.phaselen)]

        return np.concatenate([qpos[pos_index], 
                               qvel[vel_index], 
                               ref,
                               clock])

    def render(self):
        if self.vis is None:
            self.vis = CassieVis()

        self.vis.draw(self.sim)
