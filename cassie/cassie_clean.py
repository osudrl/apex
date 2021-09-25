# Consolidated Cassie environment.

from .cassiemujoco import pd_in_t, state_out_t, CassieSim, CassieVis

from .trajectory import CassieTrajectory, getAllTrajectories
from cassie.quaternion_function import *
from .rewards import *

from math import floor, ceil

import numpy as np 
import os
import random
import copy


import pickle

class CassieEnv_clean:
    def __init__(self, simrate=60, dynamics_randomization=False, reward="empty_reward", history=0, model="cassie.xml", reinit=False):
        # self.sim = CassieSim("./cassie/cassiemujoco/cassie.xml")
        self.sim = CassieSim(os.path.join("./cassie/cassiemujoco/", model), reinit)
        self.curr_model = model
        self.vis = None
        self.clock_based = True
        self.phase_based = False
        self.all_resets = False

        self.reward_func = reward
        self.dynamics_randomization = dynamics_randomization
        self.simrate = simrate # simulate X mujoco steps with same pd target
                                # 60 brings simulation from 2000Hz to roughly 30Hz

        # Load ref traj for reset states
        if self.all_resets:
            self.reset_states = np.load(os.path.join(dirname, "trajectory", "total_reset_states.npz"))
            self.reset_len = self.reset_states["qpos"].shape[0]
        else:
            dirname = os.path.dirname(__file__)
            traj_path = os.path.join(dirname, "trajectory", "stepdata.bin")
            self.reset_states = CassieTrajectory(traj_path)

        self.observation_space, self.clock_inds, self.mirrored_obs = self.set_up_state_space()
        self.action_space = np.zeros(10)

        # Adds option for state history for FF nets
        self._obs = len(self.observation_space)
        self.history = history
        self.state_history = [np.zeros(self._obs) for _ in range(self.history+1)]
        self.observation_space = np.zeros(self._obs + self._obs * self.history)

        self.P = np.array([100,  100,  88,  96,  50]) 
        self.D = np.array([10.0, 10.0, 8.0, 9.6, 5.0])
        self.P[0] *= 0.8
        self.D[0] *= 0.8
        self.P[1] *= 0.8
        self.D[1] *= 0.8

        self.u = pd_in_t()
        # TODO: should probably initialize this to current state
        
        self.cassie_state = state_out_t()


        self.time    = 0 # number of time steps in current episode
        self.phase   = 0 # portion of the phase the robot is in
        self.counter = 0 # number of phase cycles completed in episode
        self.speed = 0
        self.phase_add = 1

        # NOTE: a reference trajectory represents ONE phase cycle
        # Set cycle time/number of phases in the clock
        cycle_time = 0.8
        self.phaselen = cycle_time / (self.simrate * 0.0005)

        # see include/cassiemujoco.h for meaning of these indices
        self.pos_idx = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        self.vel_idx = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]

        self.pos_index = np.array([1,2,3,4,5,6,7,8,9,14,15,16,20,21,22,23,28,29,30,34]) # For ref traj matching
        self.vel_index = np.array([0,1,2,3,4,5,6,7,8,12,13,14,18,19,20,21,25,26,27,31]) # For ref traj matching

        self.offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])
        # global flat foot orientation, can be useful part of reward function:
        self.neutral_foot_orient = np.array([-0.24790886454547323, -0.24679713195445646, -0.6609396704367185, 0.663921021343526])
        self.face_up_orient = euler2quat(x=0, y=-np.pi/4, z=0)

        #### Dynamics Randomization ####
        # self.dynamics_randomization = False
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
        self.step_in_place = False
        self.train_mass = False
        self.train_cart = False
        self.train_carrypole = False
        self.train_loadmass = False
        self.train_pole = False
        self.load_list = ["cassie.xml", "cassie_tray_box.xml", "cassie_cart_soft.xml", "cassie_carry_pole.xml", "cassie_jug_spring.xml"]
        self.legacy_reward = False
        self.mass_reward = self.train_mass or self.train_pole
        self.stand_reward = self.train_stand
        if self.train_mass and self.sim.nq != 42:
            print("Error: wrong model file")
            exit()
        # Record default dynamics parameters
        if self.dynamics_randomization:
            self.default_damping = self.sim.get_dof_damping()
            self.default_mass = self.sim.get_body_mass()
            self.default_ipos = self.sim.get_body_ipos()
            self.default_fric = self.sim.get_geom_friction()

            weak_factor = 0.5
            strong_factor = 3.5

            pelvis_damp_range = [[self.default_damping[0], self.default_damping[0]], 
                                [self.default_damping[1], self.default_damping[1]], 
                                [self.default_damping[2], self.default_damping[2]], 
                                [self.default_damping[3], self.default_damping[3]], 
                                [self.default_damping[4], self.default_damping[4]], 
                                [self.default_damping[5], self.default_damping[5]]] 

            hip_damp_range = [[self.default_damping[6]*weak_factor, self.default_damping[6]*strong_factor],
                            [self.default_damping[7]*weak_factor, self.default_damping[7]*strong_factor],
                            [self.default_damping[8]*weak_factor, self.default_damping[8]*strong_factor]]  # 6->8 and 19->21

            achilles_damp_range = [[self.default_damping[9]*weak_factor,  self.default_damping[9]*strong_factor],
                                    [self.default_damping[10]*weak_factor, self.default_damping[10]*strong_factor], 
                                    [self.default_damping[11]*weak_factor, self.default_damping[11]*strong_factor]] # 9->11 and 22->24

            knee_damp_range     = [[self.default_damping[12]*weak_factor, self.default_damping[12]*strong_factor]]   # 12 and 25
            shin_damp_range     = [[self.default_damping[13]*weak_factor, self.default_damping[13]*strong_factor]]   # 13 and 26
            tarsus_damp_range   = [[self.default_damping[14], self.default_damping[14]]]             # 14 and 27
            heel_damp_range     = [[self.default_damping[15], self.default_damping[15]]]                           # 15 and 28
            fcrank_damp_range   = [[self.default_damping[16]*weak_factor, self.default_damping[16]*strong_factor]]   # 16 and 29
            prod_damp_range     = [[self.default_damping[17], self.default_damping[17]]]                           # 17 and 30
            foot_damp_range     = [[self.default_damping[18]*weak_factor, self.default_damping[18]*strong_factor]]   # 18 and 31

            side_damp = hip_damp_range + achilles_damp_range + knee_damp_range + shin_damp_range + tarsus_damp_range + heel_damp_range + fcrank_damp_range + prod_damp_range + foot_damp_range
            self.damp_range = pelvis_damp_range + side_damp + side_damp

            hi = 1.7
            lo = 0.5
            m = self.default_mass
            pelvis_mass_range      = [[lo*m[1],  hi*m[1]]]  # 1
            hip_mass_range         = [[lo*m[2],  hi*m[2]],  # 2->4 and 14->16
                                    [lo*m[3],  hi*m[3]], 
                                    [lo*m[4],  hi*m[4]]] 

            achilles_mass_range    = [[lo*m[5],  hi*m[5]]]  # 5 and 17
            knee_mass_range        = [[lo*m[6],  hi*m[6]]]  # 6 and 18
            knee_spring_mass_range = [[lo*m[7],  hi*m[7]]]  # 7 and 19
            shin_mass_range        = [[lo*m[8],  hi*m[8]]]  # 8 and 20
            tarsus_mass_range      = [[lo*m[9],  hi*m[9]]]  # 9 and 21
            heel_spring_mass_range = [[lo*m[10], hi*m[10]]] # 10 and 22
            fcrank_mass_range      = [[lo*m[11], hi*m[11]]] # 11 and 23
            prod_mass_range        = [[lo*m[12], hi*m[12]]] # 12 and 24
            foot_mass_range        = [[lo*m[13], hi*m[13]]] # 13 and 25

            side_mass = hip_mass_range + achilles_mass_range \
                        + knee_mass_range + knee_spring_mass_range \
                        + shin_mass_range + tarsus_mass_range \
                        + heel_spring_mass_range + fcrank_mass_range \
                        + prod_mass_range + foot_mass_range

            self.mass_range = [[0, 0]] + pelvis_mass_range + side_mass + side_mass

            self.damp_noise = np.zeros(len(self.damp_range))
            self.mass_noise = np.zeros(len(self.mass_range))
            self.fric_noise = np.zeros(3)

        # self.delta_x_min, self.delta_x_max = self.default_ipos[3] - 0.05, self.default_ipos[3] + 0.05
        # self.delta_y_min, self.delta_y_max = self.default_ipos[4] - 0.05, self.default_ipos[4] + 0.05

        ### Trims ###
        self.joint_offsets = np.zeros(14)
        self.com_vel_offset = 0
        self.y_offset = 0

        ### Random commands during training ###
        self.speed_schedule = np.zeros(3)
        self.orient_add = 0
        self.orient_command = 0
        self.orient_time = 1000 
        self.orient_dur = 40
        self.speed_time = 500
        self.turn_command = 0
        self.turn_rate = 0
        self.push_ang = 0
        self.push_size = 0
        self.push_time = np.inf
        self.push_dur = 0
        self.sprint_switch = np.inf#random.randint(60, 125)


        # Keep track of actions, torques
        self.prev_action = None 
        self.curr_action = None
        self.max_foot_vel = 0.0

        # for RNN policies
        self.critic_state = None

        # Reward terms
        self.des_foot_height = 0.10
        self.l_foot_orient = 0
        self.r_foot_orient = 0
        self.neutral_foot_orient = np.array([-0.24790886454547323, -0.24679713195445646, -0.6609396704367185, 0.663921021343526])
        self.lfoot_vel = np.zeros(3)
        self.rfoot_vel = np.zeros(3)
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
        self.orient_rollpitch_cost      = 0
        self.straight_cost              = 0
        self.yvel_cost                  = 0
        self.com_height                 = 0
        self.face_up_cost               = 0
        self.max_speed_cost             = 0
        self.motor_vel_cost             = 0
        self.motor_acc_cost             = 0
        self.foot_vel_cost              = 0
        self.ZMP_cost                   = 0
        self.tray_box_cost              = 0
        self.pole_pos_cost              = 0
        self.pole_vel_cost              = 0
        self.orient_pitch               = 0

        self.swing_ratio = 0.4

        self.debug = False

    def remake_cassie(self, model):
        self.sim = CassieSim(os.path.join("./cassie/cassiemujoco/", model), True)
        self.curr_model = model

    def set_up_state_space(self):

        mjstate_size   = 40
        state_est_size = 44

        speed_size     = 1

        clock_size    = 2
        
        
        base_mir_obs = np.array([ 3, -4, 5, 0.1, -1, 2, 6, -7, 8, -9, -15, -16, 17, 18, 19, -10, -11, 12, 13, 14, 20, -21, 22,
                                    -23, 24, -25, -31, -32, 33, 34, 35, -26, -27, 28, 29, 30, 
                                    38, 39, 36, 37, 42, 43, 40, 41])
        obs_size = state_est_size
        
        append_obs = np.array([len(base_mir_obs) + i for i in range(clock_size+speed_size)])
        mirrored_obs = np.concatenate([base_mir_obs, append_obs])
        clock_inds = append_obs[0:clock_size].tolist()
        obs_size += clock_size + speed_size
       

        # NOTE: mirror loss only set up for clock based with state estimation so far. 
        observation_space = np.zeros(obs_size)

        # check_arr = np.arange(obs_size, dtype=np.float64)
        # check_arr[0] = 0.1
        # print("mir obs check: ", np.all(np.sort(np.abs(mirrored_obs)) == check_arr))
        # print("mir obs check: ", np.sort(np.abs(mirrored_obs)))
        # print("mir obs check: ", np.sort(np.abs(mirrored_obs)) == check_arr)
        
        mirrored_obs = mirrored_obs.tolist()
        # print("mirrored_obs: ", mirrored_obs)
        # print("mir obs len: ", len(mirrored_obs))
        # print("obs_size: ", obs_size)
        
        return observation_space, clock_inds, mirrored_obs

    def step_simulation(self, action):

        target = action + self.offset

        if self.joint_rand:
            target -= self.joint_offsets[0:10]

        curr_fpos = np.zeros(6)
        self.sim.foot_pos(curr_fpos)
        prev_foot = copy.deepcopy(curr_fpos)
        self.u = pd_in_t()
        for i in range(5):
            # TODO: move setting gains out of the loop?
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

        self.cassie_state = self.sim.step_pd(self.u)
        self.sim.foot_pos(curr_fpos)
        self.lfoot_vel = (curr_fpos[0:3] - prev_foot[0:3]) / 0.0005
        self.rfoot_vel = (curr_fpos[3:6] - prev_foot[3:6]) / 0.0005

    def step_sim_basic(self, action):

        target = action + self.offset

        if self.joint_rand:
            target -= self.joint_offsets[0:10]

        self.u = pd_in_t()
        for i in range(5):
            # TODO: move setting gains out of the loop?
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

        self.cassie_state = self.sim.step_pd(self.u)

    def step(self, action, return_omniscient_state=False, f_term=0):

        foot_pos = np.zeros(6)
        self.l_foot_orient = 0
        self.r_foot_orient = 0
        self.l_foot_cost_pos = 0
        self.r_foot_cost_pos = 0
        self.l_foot_cost_forcevel = 0
        self.r_foot_cost_forcevel = 0
        self.torque_cost = 0
        self.hiproll_cost = 0
        self.hipyaw_vel = 0
        self.act_cost                   = 0
        self.torque_penalty             = 0
        self.pel_transacc               = 0
        self.pel_rotacc                 = 0
        self.forward_cost               = 0
        self.orient_cost                = 0
        self.straight_cost              = 0
        self.yvel_cost                  = 0
        if self.legacy_reward: 
            self.l_foot_cost_var = 0
            self.r_foot_cost_var = 0
            self.l_foot_cost_speedpos = 0
            self.r_foot_cost_speedpos = 0
            self.l_foot_cost_hop = 0
            self.r_foot_cost_hop = 0
            self.hiproll_act = 0
            self.hipyaw_act = 0
            self.l_foot_cost_smooth_force   = 0
            self.r_foot_cost_smooth_force   = 0
            self.orient_rollpitch_cost      = 0
            self.max_speed_cost             = 0
            self.motor_vel_cost             = 0
            self.motor_acc_cost             = 0
        if self.mass_reward:
            self.com_height                 = 0
            self.tray_bos_cost              = 0
            self.pole_pos_cost              = 0
            self.pole_vel_cost              = 0
            self.orient_pitch               = 0
        if self.stand_reward:
            self.com_height                 = 0
            self.foot_vel_cost              = 0
            self.ZMP_cost                   = 0

        # Reward clocks
        l_swing, r_swing = self.lin_clock5(self.swing_ratio, self.phase, percent_trans=.7)
        l_force, r_force = self.lin_clock5(self.swing_ratio, self.phase, percent_trans=.2)
        l_stance = -l_force + 1
        r_stance = -r_force + 1

        orient_target = np.array([1, 0, 0, 0])
        speed_target = np.array([self.speed, 0, 0])
        foot_orient_target = self.neutral_foot_orient
        
        des_height = self.des_foot_height
        if self.orient_add != 0:
            quaternion = euler2quat(z=self.orient_add, x=0, y=0)
            iquaternion = inverse_quaternion(quaternion)

            foot_orient_target = quaternion_product(iquaternion, foot_orient_target)
            speed_target = rotate_by_quaternion(speed_target, quaternion)

        for _ in range(self.simrate):
            if self.push_time <= self.time <= self.push_time + self.push_dur:
                force_x = self.push_size * np.cos(self.push_ang)
                force_y = self.push_size * np.sin(self.push_ang)
                self.sim.apply_force([force_x, force_y, 0, 0, 0, 0], "cassie-pelvis")
            self.step_simulation(action)
            qpos = np.copy(self.sim.qpos_full())
            qvel = np.copy(self.sim.qvel())

            # Foot Orientation Cost
            # lquat = self.cassie_state.leftFoot.orientation[:]
            # rquat = self.cassie_state.rightFoot.orientation[:]
            # ref = qpos[3:7]
            # lquat = quaternion_product(ref, lquat)
            # rquat = quaternion_product(ref, rquat)
            # leuler = quaternion2euler(lquat)
            # reuler = quaternion2euler(rquat)
            self.l_foot_orient += 20*(1 - np.inner(foot_orient_target, self.sim.xquat("left-foot")) ** 2)
            self.r_foot_orient += 20*(1 - np.inner(foot_orient_target, self.sim.xquat("right-foot")) ** 2)
            # self.l_foot_orient += leuler[1]
            # self.r_foot_orient += reuler[1]

            # Hip Yaw velocity cost
            self.hiproll_cost += (np.abs(qvel[6]) + np.abs(qvel[19])) / 3
            self.hipyaw_vel += (np.abs(qvel[7]) + np.abs(qvel[20]))                

            # Foot height cost
            self.sim.foot_pos(foot_pos)
            foot_forces = self.sim.get_foot_forces()
            if foot_forces[0] > self.l_max_foot_force:
                self.l_max_foot_force = foot_forces[0]
            if foot_forces[1] > self.r_max_foot_force:
                self.r_max_foot_force = foot_forces[1]
            # foot_forces = np.zeros(6)
            
            r_height_cost = 40*(des_height - foot_pos[5])**2
            r_vel_cost = np.sqrt(np.power(self.rfoot_vel[:], 2).sum())
            r_force_cost = np.abs(foot_forces[1]) / 75
            l_height_cost = 40*(des_height - foot_pos[2])**2
            l_vel_cost = np.sqrt(np.power(self.lfoot_vel[:], 2).sum())
            l_force_cost = np.abs(foot_forces[0]) / 75
            
            # Foot height cost smooth 
            # Left foot starts on ground and then lift up and then back on ground.
            # Right foot starts in air and then put on ground and then lift back up.
            self.l_foot_cost_pos += l_swing*(l_height_cost)
            self.r_foot_cost_pos += r_swing*(r_height_cost)
            self.l_foot_cost_forcevel += l_force*(l_force_cost) + l_stance*l_vel_cost
            self.r_foot_cost_forcevel += r_force*(r_force_cost) + r_stance*r_vel_cost                

            # Torque costs
            curr_torques = np.array(self.cassie_state.motor.torque[:])
            # self.torque_cost += 0.00006*np.linalg.norm(np.square(curr_torques))
            self.torque_penalty += 0.05 * sum(np.abs(curr_torques)/len(curr_torques))
            
            self.pel_transacc += np.linalg.norm(self.cassie_state.pelvis.translationalAcceleration[0:2])
            self.pel_rotacc += 2*np.linalg.norm(self.cassie_state.pelvis.rotationalVelocity[:])

            # Speedmatching costs
            forward_diff = np.abs(qvel[0] - speed_target[0])
            y_vel = np.abs(qvel[1] - speed_target[1])
            actual_orient = qpos[3:7]
            if self.orient_add != 0:
                actual_orient = quaternion_product(quaternion, actual_orient)
                if actual_orient[0] < 0:
                    actual_orient = -actual_orient
            orient_diff = 1 - np.inner(orient_target, actual_orient) ** 2

            straight_diff = 8*np.abs(qpos[1])
            # if forward_diff < 0.05:
            #     forward_diff = 0
            # if y_vel < 0.05:
            #     y_vel = 0
            if np.abs(qpos[1]) < 0.05:
                straight_diff = 0
            if orient_diff < 5e-3:
                orient_diff = 0
            else:
                orient_diff *= 30
            self.forward_cost += forward_diff
            self.orient_cost += orient_diff
            self.straight_cost += straight_diff
            self.yvel_cost += y_vel

            # Mass costs
            if self.train_mass:
                shift = np.array([0.1, 0, 0.23])
                shift = rotate_by_quaternion(shift, qpos[3:7])
                box_targ_pos = qpos[0:2] + shift[0:2]
                self.tray_box_cost += 2*np.linalg.norm(box_targ_pos - qpos[35:37])
                self.com_height += 4*(0.95 - qpos[2]) ** 2
            if self.train_pole:
                self.pole_pos_cost += np.abs(3*qpos[-1])
                self.pole_vel_cost += np.abs(qvel[-1])
                if np.abs(euler[1]) > 0.06:
                    self.orient_pitch += 2*np.abs(euler[1])
            if self.stand_reward:
                ZMP_mid = (foot_pos[0:2] + foot_pos[3:5]) / 2
                self.com_height += 4*(0.95 - qpos[2]) ** 2
                self.foot_vel_cost += self.max_foot_vel
                self.ZMP_cost += 4*np.linalg.norm(qpos[0:2] - ZMP_mid)
            if self.legacy_reward:
                if self.prev_action is not None:
                    self.hiproll_act += 2*np.linalg.norm(self.prev_action[[0, 5]] - action[[0, 5]])
                    self.hipyaw_act += 2*np.linalg.norm(self.prev_action[[1, 6]] - action[[1, 6]])
                else:
                    self.hiproll_act += 0
                    self.hipyaw_act += 0
                self.max_foot_vel = max(self.max_foot_vel, l_vel_cost, r_vel_cost)
                euler = quaternion2euler(qpos[3:7])
                orient_rollpitch = 2 * np.linalg.norm(euler[[0,1]])
                self.orient_rollpitch_cost += orient_rollpitch
                self.max_speed_cost += qvel[0]
                self.l_foot_cost_var += l_swing*(l_height_cost+l_force_cost) + l_stance*l_vel_cost
                self.r_foot_cost_var += r_swing*(r_height_cost+r_force_cost) + r_stance*r_vel_cost
                self.l_foot_cost_speedpos += l_swing*(l_height_cost) + l_stance*l_vel_cost
                self.r_foot_cost_speedpos += r_swing*(r_height_cost) + r_stance*r_vel_cost
                self.l_foot_cost_hop += l_force*(l_force_cost) + l_stance*l_vel_cost
                self.r_foot_cost_hop += l_force*(r_force_cost) + l_stance*r_vel_cost
                self.motor_vel_cost += 0.5*(np.abs(qvel[self.vel_idx]).sum() / 10)
                self.motor_acc_cost += 4*np.linalg.norm(qvel[self.vel_idx])

        self.l_foot_orient              /= self.simrate
        self.r_foot_orient              /= self.simrate
        self.l_foot_cost_pos            /= self.simrate
        self.r_foot_cost_pos            /= self.simrate
        self.l_foot_cost_forcevel       /= self.simrate
        self.r_foot_cost_forcevel       /= self.simrate
        self.torque_cost                /= self.simrate
        self.hiproll_cost               /= self.simrate
        self.hipyaw_vel                 /= self.simrate
        self.torque_penalty             /= self.simrate
        self.pel_transacc               /= self.simrate
        self.pel_rotacc                 /= self.simrate
        self.forward_cost               /= self.simrate
        self.orient_cost                /= self.simrate
        self.straight_cost              /= self.simrate
        self.yvel_cost                  /= self.simrate
        if self.legacy_reward: 
            self.l_foot_cost_var            /= self.simrate
            self.r_foot_cost_var            /= self.simrate
            self.l_foot_cost_speedpos       /= self.simrate
            self.r_foot_cost_speedpos       /= self.simrate
            self.l_foot_cost_hop            /= self.simrate
            self.r_foot_cost_hop            /= self.simrate
            self.hiproll_act                /= self.simrate
            self.hipyaw_act                 /= self.simrate
            self.l_foot_cost_smooth_force   /= self.simrate
            self.r_foot_cost_smooth_force   /= self.simrate
            self.orient_rollpitch_cost      /= self.simrate
            self.max_speed_cost             /= self.simrate
            self.motor_vel_cost             /= self.simrate
            self.motor_acc_cost             /= self.simrate
        if self.mass_reward:
            self.com_height                 /= self.simrate
            self.tray_bos_cost              /= self.simrate
            self.pole_pos_cost              /= self.simrate
            self.pole_vel_cost              /= self.simrate
            self.orient_pitch               /= self.simrate
        if self.stand_reward:
            self.com_height                 /= self.simrate
            self.foot_vel_cost              /= self.simrate
            self.ZMP_cost                   /= self.simrate
        if self.prev_action is not None:
            self.act_cost = 5 * sum(np.abs(self.prev_action - action)) / len(action)
        else:
            self.act_cost = 0
               
        height = self.sim.qpos()[2]
        self.curr_action = action

        self.time  += 1
        self.phase += self.phase_add
        self.orient_add += self.turn_rate

        # Early termination
        done = not(height > 0.4 and height < 3.0)

        reward = self.compute_reward(action)

        if self.phase >= self.phaselen:
            self.phase -= self.phaselen
            self.counter += 1

        if l_swing > 0.5:
            self.l_max_foot_force = 0
        if r_swing > 0.5:
            self.r_max_foot_force = 0

        # update previous action
        self.prev_action = action
        # self.update_speed(self.speed_schedule[int(np.floor(self.time/self.speed_time))])
        # if self.time != 0 and self.time % self.speed_time == 0:
        if self.time >= self.speed_time:
            self.update_speed(self.speed_schedule[1])
            # self.update_speed(max(0.0, min(3.0, self.speed + self.speed_schedule[min(int(np.floor(self.time/self.speed_time)), 2)])))
            # print("update speed: ", self.speed)
        if self.orient_time <= self.time <= self.orient_time + self.orient_dur:
            # self.orient_add = self.orient_command * (self.time - self.orient_time) / self.orient_dur
            self.turn_rate = self.turn_command
        else:
            self.turn_rate = 0
        if self.time == self.sprint_switch:
            self.sprint = 1 - self.sprint     # Swap 0 to 1 and 1 to 0

        # TODO: make 0.3 a variable/more transparent
        if reward < 0.4:# or np.exp(-self.l_foot_cost_smooth) < f_term or np.exp(-self.r_foot_cost_smooth) < f_term:
            done = True
        if self.getup:
            done = False
        # print("curr model:", self.curr_model)
        if self.train_mass or self.curr_model == "cassie_tray_box.xml":
            if self.sim.qpos_full()[37] < 0.5:
                done = True
        # if self.train_pole and (np.abs(qpos[-1]) > np.pi/4 or np.abs(qvel[4]) > 1.5):
        #     done = True

        if return_omniscient_state:
            return self.get_full_state(), self.get_omniscient_state(), reward, done, {}
        else:
            return self.get_full_state(), reward, done, {}

    def lin_clock5(self, swing_ratio, phase, percent_trans=.4):

        # percent_trans = .4  # percent of swing time to use as transition
        swing_time = self.phaselen * swing_ratio
        stance_time = self.phaselen * (1-swing_ratio)
        trans_time = swing_time * percent_trans
        phase_offset = (swing_time - stance_time) / 2
        swing_time -= trans_time
        r_phase = phase - phase_offset
        if r_phase < 0:
            r_phase += self.phaselen
        l_swing_linclock = 0
        if phase < trans_time / 2:
            l_swing_linclock = phase / (trans_time / 2)
        elif trans_time / 2 < phase <= swing_time + trans_time / 2:
            l_swing_linclock = 1
        elif swing_time + trans_time / 2 < phase <= swing_time + trans_time:
            l_swing_linclock = 1 - (phase-(swing_time+trans_time/2)) / (trans_time/2)
        elif swing_time+trans_time <= phase < self.phaselen:#swing_time+trans_time+stance_time:
            l_swing_linclock = 0
        r_swing_linclock = 0
        if r_phase < stance_time:
            r_swing_linclock = 0
        elif stance_time < r_phase <= stance_time + (trans_time)/2:
            r_swing_linclock = (r_phase-stance_time) / (trans_time/2)
        elif stance_time+trans_time/2 < r_phase <= stance_time+trans_time/2+swing_time:
            r_swing_linclock = 1
        elif stance_time+trans_time/2+swing_time < r_phase <= stance_time+swing_time+trans_time:
            r_swing_linclock = 1 - (r_phase-(stance_time+trans_time/2+swing_time)) / (trans_time/2)

        return l_swing_linclock, r_swing_linclock

    def step_basic(self, action, return_omniscient_state=False):

        for _ in range(self.simrate):
            self.step_sim_basic(action)

        self.time  += 1
        self.phase += self.phase_add

        if self.phase > self.phaselen:
            self.phase = 0
            self.counter += 1

        if return_omniscient_state:
            return self.get_full_state(), self.get_omniscient_state()
        else:
            return self.get_full_state()

    def reset(self):

        self.phase = random.uniform(0, self.phaselen)
        self.time = 0
        self.counter = 0

        self.state_history = [np.zeros(self._obs) for _ in range(self.history+1)]

        if self.dynamics_randomization:
            #### Dynamics Randomization ####
            self.damp_noise = np.clip([np.random.uniform(a, b) for a, b in self.damp_range], 0, None)
            self.mass_noise = np.clip([np.random.uniform(a, b) for a, b in self.mass_range], 0, None)
            if self.sim.nv > 32:
                self.damp_noise = np.concatenate((self.damp_noise, self.default_damping[32:]))
            if self.sim.nbody > 26:
                self.mass_noise = np.concatenate((self.mass_noise, self.default_mass[26:]))
            # com_noise = [0, 0, 0] + [np.random.uniform(self.delta_x_min, self.delta_x_min)] + [np.random.uniform(self.delta_y_min, self.delta_y_max)] + [0] + list(self.default_ipos[6:])
            # fric_noise = [np.random.uniform(0.95, 1.05)] + [np.random.uniform(5e-4, 5e-3)] + [np.random.uniform(5e-5, 5e-4)]#+ list(self.default_fric[2:])
            # fric_noise = []
            # translational = np.random.uniform(0.6, 1.2)
            # torsional = np.random.uniform(1e-4, 1e-2)
            # rolling = np.random.uniform(5e-5, 5e-4)
            # for _ in range(int(len(self.default_fric)/3)):
            #     fric_noise += [translational, torsional, rolling]
            self.fric_noise = np.clip([np.random.uniform(0.6, 1.2), np.random.uniform(1e-4, 1e-2), np.random.uniform(5e-5, 5e-4)], 0, None)
            self.sim.set_dof_damping(self.damp_noise)
            self.sim.set_body_mass(self.mass_noise)
            # self.sim.set_body_ipos(com_noise)
            self.sim.set_geom_friction(self.fric_noise, "floor")
            self.sim.set_const()

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

        if self.curr_model == "cassie_tray_box.xml":#self.train_mass or (self.train_loadmass and rand_mass == 1):
            shift = np.array([0.1, 0, 0.23])
            shift = rotate_by_quaternion(shift, qpos[3:7])
            qpos = np.concatenate((qpos, [qpos[0]+shift[0], qpos[1]+shift[1], qpos[2]+shift[2]], qpos[3:7]))
            qvel = np.concatenate((qvel, qvel[0:2], np.zeros(4)))    

        if self.train_loadmass and rand_mass > 1:
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
            self.speed = (random.randint(-5, 10)) / 10
            # if random.randint(0, 4) == 0:
            #     self.speed = 0
            #     self.speed_schedule = [0, 4]
            # else:
            #     self.speed = (random.randint(0, 40)) / 10
            #     self.speed_schedule = [self.speed, np.clip(self.speed + (random.randint(10, 20)) / 10, 0, 4)]
            self.speed_time = np.inf#100 + np.random.randint(-20, 20)

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
            self.push_size = random.uniform(50, 100)
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
        self.l_foot_cost_pos = 0
        self.r_foot_cost_pos = 0
        self.l_foot_cost_forcevel = 0
        self.r_foot_cost_forcevel = 0
        self.torque_cost = 0
        self.hiproll_cost = 0
        self.hipyaw_vel = 0
        self.act_cost                   = 0
        self.torque_penalty             = 0
        self.pel_transacc               = 0
        self.pel_rotacc                 = 0
        self.forward_cost               = 0
        self.orient_cost                = 0
        self.straight_cost              = 0
        self.yvel_cost                  = 0
        if self.legacy_reward: 
            self.l_foot_cost_var = 0
            self.r_foot_cost_var = 0
            self.l_foot_cost_speedpos = 0
            self.r_foot_cost_speedpos = 0
            self.l_foot_cost_hop = 0
            self.r_foot_cost_hop = 0
            self.hiproll_act = 0
            self.hipyaw_act = 0
            self.l_foot_cost_smooth_force   = 0
            self.r_foot_cost_smooth_force   = 0
            self.orient_rollpitch_cost      = 0
            self.max_speed_cost             = 0
            self.motor_vel_cost             = 0
            self.motor_acc_cost             = 0
        if self.mass_reward:
            self.com_height                 = 0
            self.tray_bos_cost              = 0
            self.pole_pos_cost              = 0
            self.pole_vel_cost              = 0
            self.orient_pitch               = 0
        if self.stand_reward:
            self.com_height                 = 0
            self.foot_vel_cost              = 0
            self.ZMP_cost                   = 0

        return self.get_full_state()

    def reset_for_test(self, full_reset=False):
        self.phase = 0
        self.time = 0
        self.counter = 0
        self.orient_add = 0
        self.y_offset = 0
        self.phase_add = 1

        self.state_history = [np.zeros(self._obs) for _ in range(self.history+1)]

        if self.dynamics_randomization:
            self.damp_noise = self.default_damping
            self.mass_noise = self.default_mass
            self.fric_noise = self.default_fric
            self.sim.set_dof_damping(self.default_damping)
            self.sim.set_body_mass(self.default_mass)
            # self.sim.set_body_ipos(self.default_ipos)
            self.sim.set_geom_friction(self.default_fric)
            self.sim.set_const()

        self.speed = 0
        self.update_speed(self.speed)

        self.speed_schedule = self.speed * np.ones(3)
        self.orient_command = 0
        self.orient_time = np.inf 
        self.turn_rate = 0
        self.speed_time = np.inf

        if not full_reset:
            qpos = np.copy(self.reset_states.qpos[0])
            qvel = np.copy(self.reset_states.qvel[0])

            # Need to reset u? Or better way to reset cassie_state than taking step
            self.cassie_state = self.sim.step_pd(self.u)
            if self.train_cart:
                qpos = np.concatenate((qpos, np.zeros(self.sim.nq - len(qpos))))
                qvel = np.concatenate((qvel, np.zeros(self.sim.nv - len(qvel))))
            if self.train_carrypole:
                qpos = np.concatenate((qpos, np.zeros(self.sim.nq - len(qpos))))
                qvel = np.concatenate((qvel, np.zeros(self.sim.nv - len(qvel))))
            if self.curr_model == "cassie_tray_box.xml":
                shift = np.array([0.1, 0, 0.23])
                shift = rotate_by_quaternion(shift, qpos[3:7])
                qpos = np.concatenate((qpos, [qpos[0]+shift[0], qpos[1]+shift[1], qpos[2]+shift[2]], qpos[3:7]))
                qvel = np.concatenate((qvel, qvel[0:2], np.zeros(4)))    
            if self.curr_model != "cassie.xml":
                qpos = np.concatenate((qpos, np.zeros(self.sim.nq - len(qpos))))
                qvel = np.concatenate((qvel, np.zeros(self.sim.nv - len(qvel))))        
            self.sim.set_qpos_full(qpos)
            self.sim.set_qvel_full(qvel)
        else:
            self.sim.full_reset()
            self.reset_cassie_state()

        if self.slope_rand:
            self.sim.set_geom_quat(np.array([1, 0, 0, 0]), "floor")
        # if self.load_mass_rand:
            # self.sim.set_body_mass(0, name="load_mass")

        # Reward terms
        self.l_foot_orient = 0
        self.r_foot_orient = 0
        self.l_foot_cost_pos = 0
        self.r_foot_cost_pos = 0
        self.l_foot_cost_forcevel = 0
        self.r_foot_cost_forcevel = 0
        self.torque_cost = 0
        self.hiproll_cost = 0
        self.hipyaw_vel = 0
        self.act_cost                   = 0
        self.torque_penalty             = 0
        self.pel_transacc               = 0
        self.pel_rotacc                 = 0
        self.forward_cost               = 0
        self.orient_cost                = 0
        self.straight_cost              = 0
        self.yvel_cost                  = 0
        if self.legacy_reward: 
            self.l_foot_cost_var = 0
            self.r_foot_cost_var = 0
            self.l_foot_cost_speedpos = 0
            self.r_foot_cost_speedpos = 0
            self.l_foot_cost_hop = 0
            self.r_foot_cost_hop = 0
            self.hiproll_act = 0
            self.hipyaw_act = 0
            self.l_foot_cost_smooth_force   = 0
            self.r_foot_cost_smooth_force   = 0
            self.orient_rollpitch_cost      = 0
            self.max_speed_cost             = 0
            self.motor_vel_cost             = 0
            self.motor_acc_cost             = 0
        if self.mass_reward:
            self.com_height                 = 0
            self.tray_bos_cost              = 0
            self.pole_pos_cost              = 0
            self.pole_vel_cost              = 0
            self.orient_pitch               = 0
        if self.stand_reward:
            self.com_height                 = 0
            self.foot_vel_cost              = 0
            self.ZMP_cost                   = 0

        return self.get_full_state()

    def reset_cassie_state(self):
        # Only reset parts of cassie_state that is used in get_full_state
        self.cassie_state.pelvis.position[:] = [0, 0, 1.01]
        self.cassie_state.pelvis.orientation[:] = [1, 0, 0, 0]
        self.cassie_state.pelvis.rotationalVelocity[:] = np.zeros(3)
        self.cassie_state.pelvis.translationalVelocity[:] = np.zeros(3)
        self.cassie_state.pelvis.translationalAcceleration[:] = np.zeros(3)
        self.cassie_state.terrain.height = 0
        self.cassie_state.motor.position[:] = [0.0045, 0, 0.4973, -1.1997, -1.5968, 0.0045, 0, 0.4973, -1.1997, -1.5968]
        self.cassie_state.motor.velocity[:] = np.zeros(10)
        self.cassie_state.joint.position[:] = [0, 1.4267, -1.5968, 0, 1.4267, -1.5968]
        self.cassie_state.joint.velocity[:] = np.zeros(6)

    # Helper function for updating the speed, used in visualization tests
    def update_speed(self, new_speed):
        self.speed = new_speed
        # total_duration = (0.9 - 0.2 / 3.0 * self.speed)
        # self.phaselen = floor((total_duration * 2000 / self.simrate) - 1)
        if new_speed > 1:
            self.swing_ratio = 0.4 + .4*(new_speed - 1)/3
            self.phase_add = 1 + 0.5*(new_speed - 1)/2
            self.des_foot_height = .1 + 0.2*(new_speed - 1)/2
        elif new_speed == 0 and self.train_stand:
            self.swing_ratio = 0.0
            self.phase_add = 1
        else:
            self.swing_ratio = 0.4 
            self.des_foot_height = .10
            self.phase_add = 1
        if new_speed > 3:
            self.phase_add = 1.5
            self.des_foot_height = 0.30
        if self.train_stand:
            self.des_foot_height = 0.10
        # print("speed: {}, swing: {}, phase: {}, foot_height: {}".format(new_speed, self.swing_ratio, self.phase_add, self.des_foot_height))
                

    def compute_reward(self, action):
        return globals()[self.reward_func](self)

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
        new_translationalVelocity = self.cassie_state.pelvis.translationalVelocity[:]
        # new_translationalVelocity[0:2] += self.com_vel_offset
        quaternion = euler2quat(z=self.orient_add, y=0, x=0)
        iquaternion = inverse_quaternion(quaternion)
        new_orient = quaternion_product(iquaternion, self.cassie_state.pelvis.orientation[:])
        if new_orient[0] < 0:
            new_orient = -new_orient
        new_translationalVelocity = rotate_by_quaternion(self.cassie_state.pelvis.translationalVelocity[:], iquaternion)
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

            new_translationalVelocity,                       # pelvis translational velocity
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

    def render(self):
        if self.vis is None:
            self.vis = CassieVis(self.sim, "./cassie/cassiemujoco/cassie.xml")

        return self.vis.draw(self.sim)
    
# Currently unused
# def get_omniscient_state(self):
#     full_state = self.get_full_state()
#     omniscient_state = np.hstack((full_state, self.sim.get_dof_damping(), self.sim.get_body_mass(), self.sim.get_body_ipos(), self.sim.get_ground_friction))
#     return omniscient_state

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
