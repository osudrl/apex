# Consolidated Cassie environment.

from .cassiemujoco import pd_in_t, state_out_t, CassieSim, CassieVis

from .trajectory import CassieTrajectory, getAllTrajectories
from cassie.quaternion_function import *
from .rewards import *
from .cassie_clean import CassieEnv_clean

from math import floor, ceil

import numpy as np 
import os
import random
import copy


import pickle

class CassieEnv_accel_nopelvel(CassieEnv_clean):
    def __init__(self, simrate=60, dynamics_randomization=False, reward="empty_reward", history=0, model="cassie.xml", reinit=False):
        CassieEnv_clean.__init__(self, simrate, dynamics_randomization, reward, history)
        
        self.slope_rand = False
        self.joint_rand = False
        self.load_mass_rand = False
        self.getup = False
        self.zero_clock = False
        self.sprint = None
        self.train_turn = False
        self.train_push = False
        self.train_sprint = False
        self.train_stand = False
        self.train_mass = False
        self.step_in_place = False
        self.train_pole = False
        self.train_loadmass = False

        if self.train_mass and self.sim.nq != 42:
            print("Error: wrong model file")
            exit()

    def reset(self):

        self.phase = random.uniform(0, self.phaselen)
        self.time = 0
        self.counter = 0

        self.state_history = [np.zeros(self._obs) for _ in range(self.history+1)]

        if self.all_resets:        
            rand_ind = random.randint(0, self.reset_len-1)
            qpos = np.copy(self.reset_states["qpos"][rand_ind, :])
            qvel = np.copy(self.reset_states["qvel"][rand_ind, :])
        else:
            rand_ind = random.randint(0, len(self.reset_states)-1)
            qpos = np.copy(self.reset_states.qpos[rand_ind])
            qvel = np.copy(self.reset_states.qvel[rand_ind])

        x_size = 0.3
        y_size = 0.2
        qvel[0] += np.random.random() * 2 * x_size - x_size
        qvel[1] = np.random.random() * 2 * y_size - y_size
        orientation = random.randint(-10, 10) * np.pi / 25
        quaternion = euler2quat(z=orientation, y=0, x=0)
        qpos[3:7] = quaternion
        self.y_offset = 0#random.uniform(-3.5, 3.5)
        # qpos[1] = self.y_offset

        if self.train_loadmass:
            rand_mass = random.randint(0, len(self.load_list))
            self.sim = CassieSim("./cassie/cassiemujoco/" + self.load_list[rand_mass], reinit=True)
            if self.vis is not None:
                self.vis = CassieVis(self.sim, "./cassie/cassiemujoco/" + self.load_list[rand_mass])

        if self.curr_model == "cassie_tray_box.xml" or self.train_mass:
            shift = np.array([0.1, 0, 0.23])
            shift = rotate_by_quaternion(shift, qpos[3:7])
            qpos = np.concatenate((qpos, [qpos[0]+shift[0], qpos[1]+shift[1], qpos[2]+shift[2]], qpos[3:7]))
            qvel = np.concatenate((qvel, qvel[0:2], np.zeros(4)))             
        elif self.curr_model != "cassie.xml":
            qpos = np.concatenate((qpos, np.zeros(self.sim.nq - len(qpos))))
            qvel = np.concatenate((qvel, np.zeros(self.sim.nv - len(qvel))))        

        self.sim.set_qpos_full(qpos)
        self.sim.set_qvel_full(qvel)

        if self.getup:
            angle = np.random.uniform(0, 2*np.pi)
            force_size = np.random.uniform(100, 150)
            force_time = np.random.randint(10, 20)
            force_x = force_size * np.cos(angle)
            force_y = force_size * np.sin(angle)
            for i in range(force_time):
                self.sim.apply_force([force_x, force_y, 0, 0, 0, 0], "cassie-pelvis")
                self.step_basic(np.zeros(10))
            for i in range(10):
                self.step_basic(np.zeros(10))
            self.phase = random.uniform(0, self.phaselen)
            self.time = 0
            self.counter = 0

            self.state_history = [np.zeros(self._obs) for _ in range(self.history+1)]

        # Need to reset u? Or better way to reset cassie_state than taking step
        self.cassie_state = self.sim.step_pd(self.u)

        if self.train_stand:
            self.speed = 0
            # self.speed = (random.randint(-5, 5)) / 10
            # if bool(random.getrandbits(1)):
                # self.speed = 0
        else:
            # self.speed = (random.randint(0, 40)) / 10
            self.speed_time = np.inf
            if random.randint(0, 4) == 0:
                self.speed = 0
            #     self.speed_schedule = [0, 4]
            else:
                self.speed = (random.randint(0, 40)) / 10
            #     self.speed_schedule = [self.speed, np.clip(self.speed + (random.randint(5, 10)) / 10, 0, 4)]
            # self.speed_time = 150 + np.random.randint(-20, 20)

        # self.speed_schedule = np.random.randint(0, 30, size=3) / 10
        # self.speed_schedule = np.random.randint(-10, 10, size=3) / 10
        # self.speed_schedule = self.speed*np.ones(3)
       
        # self.speed = self.speed_schedule[0]
        # Make sure that if speed is above 2, freq is at least 1.2
        # if self.speed > 1.5 or np.any(self.speed_schedule > 1.5):
        #     self.phase_add = 1.3 + 0.7*random.random()
        # else:
        #     self.phase_add = 1 + random.random()
        # self.phase_add = 1 + 0.5*random.random()
        self.update_speed(self.speed)

        self.orient_add = 0#random.randint(-10, 10) * np.pi / 25
        if self.train_turn and bool(random.getrandbits(1)):
            tps = 1/4 * np.pi * (self.simrate*0.0005) # max radian change per second to command
            if self.speed >= 3:
                tps /= 2
            self.turn_command = random.uniform(-tps, tps)
            self.orient_time = random.randint(60, 125)
            self.orient_dur = random.randint(30, 50)
            self.turn_rate = 0
        else:
            self.turn_command = 0
            self.orient_time = np.inf
            self.orient_dur = 0
            self.turn_rate = 0

        if self.train_push and bool(random.getrandbits(1)):
            # print("doing push")
            self.push_ang = random.uniform(0, 2*np.pi)
            self.push_size = random.uniform(10, 50)
            self.push_time = random.randint(0, 125)
            self.push_dur = random.randint(5, 10)
        else:
            # print("no push")
            self.push_ang = 0
            self.push_size = 0
            self.push_time = np.inf
            self.push_dur = 0

        if self.train_sprint:
            self.sprint = 1#random.randint(0, 1)
            self.sprint_switch = np.inf#random.randint(60, 125)
            self.swing_ratio = 0.8
            self.phase_add = 1.5
            self.speed = 4
        else:
            self.sprint = None
            self.sprint_switch = np.inf

        self.com_vel_offset = 0#0.1*np.random.uniform(-0.1, 0.1, 2)
        self.max_foot_vel = 0.0

        if self.dynamics_randomization:
            #### Dynamics Randomization ####
            self.damp_noise = np.clip([np.random.uniform(a, b) for a, b in self.damp_range], 0, None)
            self.mass_noise = np.clip([np.random.uniform(a, b) for a, b in self.mass_range], 0, None)
            # com_noise = [0, 0, 0] + [np.random.uniform(self.delta_x_min, self.delta_x_min)] + [np.random.uniform(self.delta_y_min, self.delta_y_max)] + [0] + list(self.default_ipos[6:])
            # fric_noise = [np.random.uniform(0.95, 1.05)] + [np.random.uniform(5e-4, 5e-3)] + [np.random.uniform(5e-5, 5e-4)]#+ list(self.default_fric[2:])
            # fric_noise = []
            # translational = np.random.uniform(0.6, 1.2)
            # torsional = np.random.uniform(1e-4, 1e-2)
            # rolling = np.random.uniform(5e-5, 5e-4)
            # for _ in range(int(len(self.default_fric)/3)):
            #     fric_noise += [translational, torsional, rolling]
            self.fric_noise = np.clip([np.random.uniform(0.8, 1.2), np.random.uniform(1e-4, 1e-2), np.random.uniform(5e-5, 5e-4)], 0, None)
            self.sim.set_dof_damping(self.damp_noise)
            self.sim.set_body_mass(self.mass_noise)
            # self.sim.set_body_ipos(com_noise)
            self.sim.set_geom_friction(self.fric_noise, "floor")
            self.sim.set_const()

        if self.slope_rand:
            rand_angle = np.pi/180*np.random.uniform(-5, 5, 2)
            floor_quat = euler2quat(z=0, y=rand_angle[0], x=rand_angle[1])
            self.sim.set_geom_quat(floor_quat, "floor")
        if self.joint_rand:
            self.joint_offsets = np.random.uniform(-0.03, 0.03, 14)
        if self.load_mass_rand:
            self.sim.set_body_mass(np.random.uniform(1, 10), name="load_mass")

        # Reward terms
        self.l_foot_orient = 0
        self.r_foot_orient = 0
        self.l_foot_cost_var = 0
        self.r_foot_cost_var = 0
        self.l_foot_cost_speedpos = 0
        self.r_foot_cost_speedpos = 0
        self.l_foot_cost_pos = 0
        self.r_foot_cost_pos = 0
        self.l_foot_cost_forcevel = 0
        self.r_foot_cost_forcevel = 0
        self.l_foot_cost_hop = 0
        self.r_foot_cost_hop = 0
        self.torque_cost = 0
        self.hiproll_cost = 0
        self.hiproll_act = 0
        self.hipyaw_vel = 0
        self.hipyaw_act = 0
        self.pel_stable = 0
        self.act_cost                   = 0
        self.torque_penalty             = 0
        self.l_foot_cost_smooth_force   = 0
        self.r_foot_cost_smooth_force   = 0
        self.pel_transacc               = 0
        self.pel_rotacc                 = 0
        self.forward_cost               = 0
        self.orient_cost                = 0
        self.orient_rollpitch_cost        = 0
        self.straight_cost              = 0
        self.yvel_cost                  = 0
        self.com_height                 = 0
        self.face_up_cost               = 0
        self.max_speed_cost             = 0
        self.motor_vel_cost             = 0
        self.motor_acc_cost             = 0
        self.foot_vel_cost              = 0

        return self.get_full_state()

    def get_full_state(self):
        qpos = np.copy(self.sim.qpos())
        qvel = np.copy(self.sim.qvel()) 

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

        # CLOCK BASED (NO TRAJECTORY)
        clock = [np.sin(2 * np.pi *  self.phase / (self.phaselen)),
                np.cos(2 * np.pi *  self.phase / (self.phaselen))]
        if self.getup or self.zero_clock or (self.train_stand and self.speed == 0 and not self.step_in_place):
            clock = [0, 0]
        if self.sprint is not None:
            if self.sprint == 0:
                clock = [0, 0]
            elif self.sprint == 1:
                self.speed = 4
        ext_state = np.concatenate((clock, [self.speed]))

        # Update orientation
        new_orient = self.cassie_state.pelvis.orientation[:]
        new_translationalAcceleleration = self.cassie_state.pelvis.translationalAcceleration[:]
        quaternion = euler2quat(z=self.orient_add, y=0, x=0)
        iquaternion = inverse_quaternion(quaternion)
        new_orient = quaternion_product(iquaternion, self.cassie_state.pelvis.orientation[:])
        if new_orient[0] < 0:
            new_orient = -new_orient
        new_translationalAcceleleration = rotate_by_quaternion(self.cassie_state.pelvis.translationalAcceleration[:], iquaternion)
        motor_pos = self.cassie_state.motor.position[:]
        joint_pos = np.concatenate([self.cassie_state.joint.position[0:2], self.cassie_state.joint.position[3:5]])
        joint_vel = np.concatenate([self.cassie_state.joint.velocity[0:2], self.cassie_state.joint.velocity[3:5]])
        if self.joint_rand:
            motor_pos += self.joint_offsets[0:10]
            joint_pos += self.joint_offsets[10:14]

        # Use state estimator
        robot_state = np.concatenate([
            self.cassie_state.leftFoot.position[:],     # left foot position
            self.cassie_state.rightFoot.position[:],     # right foot position
            new_orient,                                 # pelvis orientation
            motor_pos,                                     # actuated joint positions

            new_translationalAcceleleration,                       # pelvis translational velocity
            self.cassie_state.pelvis.rotationalVelocity[:],                          # pelvis rotational velocity 
            self.cassie_state.motor.velocity[:],                                     # actuated joint velocities
            
            joint_pos,                                     # unactuated joint positions
            joint_vel                                      # unactuated joint velocities
        ])

        #TODO: Set up foot position for non state est
        state = np.concatenate([robot_state, ext_state])

        self.state_history.insert(0, state)
        self.state_history = self.state_history[:self.history+1]

        return np.concatenate(self.state_history)


#nbody layout:
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
