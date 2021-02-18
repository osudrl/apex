# Consolidated Cassie environment.

from .cassiemujoco import pd_in_t, state_out_t, CassieSim, CassieVis

from .trajectory import *
from cassie.quaternion_function import *

from math import floor

import numpy as np 
import os
import random
import copy
import pickle

class CassieEnv_Min:
    def __init__(self):

        dirname = os.path.dirname(__file__)
        self.sim = CassieSim("./cassie/cassiemujoco/cassie.xml")
        self.simrate = 50

        traj_path = os.path.join(dirname, "trajectory", "stepdata.bin")
        self.trajectory = CassieTrajectory(traj_path)

        self.speed = 0
        self.phase_add = 1
        self.phaselen = floor(len(self.trajectory) / self.simrate) - 1 
        self.P = np.array([100,  100,  88,  96,  50]) 
        self.D = np.array([10.0, 10.0, 8.0, 9.6, 5.0])

        self.u = pd_in_t()

        self.phase   = 0 # portion of the phase the robot is in

        # see include/cassiemujoco.h for meaning of these indices
        self.pos_idx = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        self.vel_idx = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]

        self.pos_index = np.array([1,2,3,4,5,6,7,8,9,14,15,16,20,21,22,23,28,29,30,34])
        self.vel_index = np.array([0,1,2,3,4,5,6,7,8,12,13,14,18,19,20,21,25,26,27,31])

        self.offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])

    # Basic version of step_simulation, that only simulates forward in time, does not do any other
    # computation for reward, etc. Is faster and should be used for evaluation purposes
    def step_sim_basic(self, action):
        target = action + self.offset
        self.u = pd_in_t()
        for i in range(5):
            # TODO: move setting gains out of the loop?
            # maybe write a wrapper for pd_in_t ?
            self.u.leftLeg.motorPd.pGain[i]  = self.P[i]
            self.u.rightLeg.motorPd.pGain[i] = self.P[i]
            self.u.leftLeg.motorPd.dGain[i]  = self.D[i]
            self.u.rightLeg.motorPd.dGain[i] = self.D[i]

            self.u.leftLeg.motorPd.torque[i]  = 0  # Feedforward torque
            self.u.rightLeg.motorPd.torque[i] = 0

            self.u.leftLeg.motorPd.pTarget[i]  = target[i]
            self.u.rightLeg.motorPd.pTarget[i] = target[i + 5]

            self.u.leftLeg.motorPd.dTarget[i]  = 0
            self.u.rightLeg.motorPd.dTarget[i] = 0

        self.cassie_state = self.sim.step_pd(self.u)

    def reset_for_test(self):
        self.simsteps = 0
        self.phase = 0
        self.time = 0
        self.counter = 0
        self.orient_add = 0
        self.orient_time = np.inf
        self.y_offset = 0
        self.phase_add = 1
        self.speed = 0

        qpos, qvel = self.get_ref_state(self.phase)
        self.sim.set_qpos(qpos)
        self.sim.set_qvel(qvel)
        # Need to reset u? Or better way to reset cassie_state than taking step
        self.cassie_state = self.sim.step_pd(self.u)
    
        return self.get_full_state()

    # get the corresponding state from the reference trajectory for the current phase
    def get_ref_state(self, phase=None):
        if phase is None:
            phase = self.phase

        if phase > self.phaselen:
            phase = 0

        # TODO: make this not so hackish
        if phase > floor(len(self.trajectory) / self.simrate) - 1:
            phase = floor((phase / self.phaselen) * len(self.trajectory) / self.simrate)

        desired_ind = phase * self.simrate 
        pos = np.copy(self.trajectory.qpos[int(desired_ind)])
        vel = np.copy(self.trajectory.qvel[int(desired_ind)])

        # this is just setting the x to where it "should" be given the number
        # of cycles
        # pos[0] += (self.trajectory.qpos[-1, 0] - self.trajectory.qpos[0, 0]) * self.counter

        # ^ should only matter for COM error calculation,
        # gets dropped out of state variable for input reasons

        ### Setting variable speed
        pos[0] *= self.speed
        pos[0] += (self.trajectory.qpos[-1, 0] - self.trajectory.qpos[0, 0]) * self.counter * self.speed
 
        # setting lateral distance target to 0?
        # regardless of reference trajectory?
        pos[1] = 0

        vel[0] *= self.speed

        return pos, vel

    def get_full_state(self):
        qpos = np.copy(self.sim.qpos())
        qvel = np.copy(self.sim.qvel())

        ref_pos, ref_vel = self.get_ref_state(self.phase + self.phase_add)

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

        clock = [np.sin(2 * np.pi * self.phase / self.phaselen),
                np.cos(2 * np.pi * self.phase / self.phaselen)]
        ext_state = np.concatenate((clock, [self.speed]))

        # Update orientation
        new_orient = self.cassie_state.pelvis.orientation[:]
        new_translationalVelocity = self.cassie_state.pelvis.translationalVelocity[:]
        new_translationalAcceleleration = self.cassie_state.pelvis.translationalAcceleration[:]
        # new_translationalVelocity[0:2] += self.com_vel_offset
        quaternion = euler2quat(z=self.orient_add, y=0, x=0)
        iquaternion = inverse_quaternion(quaternion)
        new_orient = quaternion_product(iquaternion, self.cassie_state.pelvis.orientation[:])
        if new_orient[0] < 0:
            new_orient = -new_orient
        new_translationalVelocity = rotate_by_quaternion(self.cassie_state.pelvis.translationalVelocity[:], iquaternion)
        new_translationalAcceleleration = rotate_by_quaternion(self.cassie_state.pelvis.translationalAcceleration[:], iquaternion)
        motor_pos = self.cassie_state.motor.position[:]
        joint_pos = self.cassie_state.joint.position[:]

        # Use state estimator
        robot_state = np.concatenate([
            [self.cassie_state.pelvis.position[2] - self.cassie_state.terrain.height],  # pelvis height
            new_orient,                                 # pelvis orientation
            motor_pos,                                     # actuated joint positions

            new_translationalVelocity,                       # pelvis translational velocity
            self.cassie_state.pelvis.rotationalVelocity[:],                          # pelvis rotational velocity
            self.cassie_state.motor.velocity[:],                                     # actuated joint velocities

            new_translationalAcceleleration,                   # pelvis translational acceleration

            joint_pos,                                     # unactuated joint positions
            self.cassie_state.joint.velocity[:]                                      # unactuated joint velocities
        ])

        state = np.concatenate([robot_state, ext_state])
    
        return state

 
# Currently unused
# def get_omniscient_state(self):
#     full_state = self.get_full_state()
#     omniscient_state = np.hstack((full_state, self.sim.get_dof_damping(), self.sim.get_body_mass(), self.sim.get_body_ipos(), self.sim.get_ground_friction))
#     return omniscient_state

# nbody layout:
# 0:  worldbody (zero)
# 1:  pelvis

# 2:  left hip roll
# 3:  left hip yaw
# 4:  left hip pitch
# 5:  left achilles rod
# 6:  left knee
# 7:  left knee spring
# 8:  left shin
# 9:  left tarsus
# 10:  left heel spring
# 12:  left foot crank
# 12: left plantar rod
# 13: left foot

# 14: right hip roll
# 15: right hip yaw
# 16: right hip pitch
# 17: right achilles rod
# 18: right knee
# 19: right knee spring
# 20: right shin
# 21: right tarsus
# 22: right heel spring
# 23: right foot crank
# 24: right plantar rod
# 25: right foot


# qpos layout
# [ 0] Pelvis x
# [ 1] Pelvis y
# [ 2] Pelvis z
# [ 3] Pelvis orientation qw
# [ 4] Pelvis orientation qx
# [ 5] Pelvis orientation qy
# [ 6] Pelvis orientation qz
# [ 7] Left hip roll         (Motor [0])
# [ 8] Left hip yaw          (Motor [1])
# [ 9] Left hip pitch        (Motor [2])
# [10] Left achilles rod qw
# [11] Left achilles rod qx
# [12] Left achilles rod qy
# [13] Left achilles rod qz
# [14] Left knee             (Motor [3])
# [15] Left shin                        (Joint [0])
# [16] Left tarsus                      (Joint [1])
# [17] Left heel spring
# [18] Left foot crank
# [19] Left plantar rod
# [20] Left foot             (Motor [4], Joint [2])
# [21] Right hip roll        (Motor [5])
# [22] Right hip yaw         (Motor [6])
# [23] Right hip pitch       (Motor [7])
# [24] Right achilles rod qw
# [25] Right achilles rod qx
# [26] Right achilles rod qy
# [27] Right achilles rod qz
# [28] Right knee            (Motor [8])
# [29] Right shin                       (Joint [3])
# [30] Right tarsus                     (Joint [4])
# [31] Right heel spring
# [32] Right foot crank
# [33] Right plantar rod
# [34] Right foot            (Motor [9], Joint [5])

# qvel layout
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
