# Consolidated Cassie environment.

from .cassiemujoco import pd_in_t, state_out_t, CassieSim, CassieVis

from cassie.quaternion_function import *
from .periodicfn.probabilistic.periodic_func import Phase, vonmises_func
from .rewards import *
from .nn_ik.ikNet import IK_solver, get_sample_pos

from math import floor

import numpy as np
import os
import random
import copy

import pickle

import torch

# Load clock based reward functions from file
def load_reward_clock_funcs(path):
    with open(path, "rb") as f:
        clock_funcs = pickle.load(f)
    return clock_funcs


class CassieEnv:
    def __init__(self, simrate=50, command_profile="clock", input_profile="full", dynamics_randomization=True,
                 learn_gains=False, reward="iros_paper",
                 config="./cassie/cassiemujoco/cassie.xml", history=0, **kwargs):

        dirname = os.path.dirname(__file__)
        self.config = config
        self.sim = CassieSim(self.config)
        # self.sim = CassieSim("./cassie/cassiemujoco/cassie_drop_step.xml")
        self.vis = None

        # Arguments for the simulation and state space
        self.command_profile = command_profile
        self.input_profile = input_profile
        self.dynamics_randomization = dynamics_randomization
        self.clock_based = True

        # kwargs
        self.debug = False
        self.has_side_speed = True
        self.training = True
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)
        # Set up the IK Solver for state resets
        self.IK_solver = IK_solver()

        # Arguments for reward function
        self.reward_func = reward
        self.early_term_cutoff = 0.3

        # State space
        self.observation_space, self.clock_inds, self.mirrored_obs, self.phase_shift_inds = self.set_up_state_space(self.command_profile, self.input_profile)

        # Adds option for state history for FF nets
        self._obs = len(self.observation_space)
        self.history = history

        self.observation_space = np.zeros(self._obs + self._obs * self.history)

        self.P = np.array([100,  100,  88,  96,  50])
        self.D = np.array([10.0, 10.0, 8.0, 9.6, 5.0])

        # learn gains means there is a delta on the default PD gains ***FOR EACH LEG***
        self.learn_gains = learn_gains
        if self.learn_gains:
            self.action_space = np.zeros(10 + 20)
            self.mirrored_acts = [-5, -6, 7, 8, 9, -0.1, -1, 2, 3, 4,
                                  -15, -16, 17, 18, 19, -10, -11, 12, 13, 14,
                                  -25, -26, 27, 28, 29, -20, -21, 22, 23, 24]
        else:
            self.action_space = np.zeros(10)
            self.mirrored_acts = [-5, -6, 7, 8, 9, -0.1, -1, 2, 3, 4]

        self.u = pd_in_t()

        # TODO: should probably initialize this to current state
        self.cassie_state = state_out_t()
        self.simrate = simrate  # simulate X mujoco steps with same pd target. 50 brings simulation from 2000Hz to exactly 40Hz
        self.time    = 0        # number of time steps in current episode
        self.phase   = 0        # portion of the phase the robot is in
        self.counter = 0        # number of phase cycles completed in episode

        # NOTE: phase_based modifies self.phaselen throughout training
        # Cycle duration, phaselength, phase_add
        self.cycle_duration = 0.9  # seconds a single walk cycle takes
        self.phaselen = (2000 / simrate) * self.cycle_duration  # this is default simrate
        self.phase_add = 1

        # Set up phase based / load in clock based reward func
        self.early_reward = True if "early" in self.reward_func else False
        self.have_incentive = False if "no_incentive" in self.reward_func else True
        
        # Initialize gait to learn (this will change during env.reset)
        self.coeff = [1, -1]
        self.ratio = [0.5, 0.5]
        self.period_shift = [0.0, 0.5]
        self.std = [0.2, 0.2]
        self.update_gait()

        ### Constant Attributes ###
        # see include/cassiemujoco.h for meaning of these indices
        self.pos_idx = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        self.vel_idx = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]

        self.pos_index = np.array([1,2,3,4,5,6,7,8,9,14,15,16,20,21,22,23,28,29,30,34])
        self.vel_index = np.array([0,1,2,3,4,5,6,7,8,12,13,14,18,19,20,21,25,26,27,31])

        # CONFIGURE OFFSET for No Delta Policies
        self.offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])

        # global flat foot orientation, can be useful part of reward function:
        self.neutral_foot_orient = np.array([-0.24790886454547323, -0.24679713195445646, -0.6609396704367185, 0.663921021343526])

        ### Command inputs and their ranges ###
        self.max_orient_change = 0.2

        self.max_speed = 3.0
        self.min_speed = -0.3

        self.max_side_speed  = 0.3
        self.min_side_speed  = -0.3

        self.min_swing_ratio = 0.3
        self.max_swing_ratio = 0.7

        self.speed = 0
        self.side_speed = 0
        self.orient_add = 0
        
        ### tracking variables ###
        self.stepcount = 0
        self.l_high = False  # only true if foot is above 0.2m 
        self.r_high = False
        self.l_swing = False  # these will be true even if foot is barely above ground
        self.r_swing = False
        self.l_foot_frc = 0
        self.r_foot_frc = 0
        self.l_foot_vel = np.zeros(3)
        self.r_foot_vel = np.zeros(3)
        self.l_foot_pos = np.zeros(3)
        self.r_foot_pos = np.zeros(3)
        self.l_foot_orient = 0
        self.r_foot_orient = 0
        self.hiproll_cost = 0
        self.hiproll_act = 0

        # TODO: should this be mujoco tracking var or use state estimator. real command interface will use state est
        # Track pelvis position as baseline for pelvis tracking command inputs
        self.last_pelvis_pos = self.sim.qpos()[0:3]
        
        # Keep track of actions, torques
        self.prev_action = None
        self.curr_action = None
        self.prev_torque = None

        #### Dynamics Randomization ####
        self.dynamics_randomization = dynamics_randomization
        self.slope_rand = dynamics_randomization
        self.simrate_rand = dynamics_randomization
        self.joint_rand = dynamics_randomization

        self.max_pitch_incline = 0.03
        self.max_roll_incline = 0.03
        
        self.encoder_noise = 0.08

        self.max_simrate = self.simrate + 15
        self.min_simrate = self.simrate - 20
        self.small_simrate_noise_mag = 2
        
        self.damping_low = 0.3
        self.damping_high = 5.0

        self.mass_low = 0.5
        self.mass_high = 1.5

        self.fric_low = 0.4
        self.fric_high = 1.1

        # Record default dynamics parameters
        self.default_simrate = self.simrate
        self.default_damping = self.sim.get_dof_damping()
        self.default_mass = self.sim.get_body_mass()
        self.default_ipos = self.sim.get_body_ipos()
        self.default_fric = self.sim.get_geom_friction()
        self.default_rgba = self.sim.get_geom_rgba()
        self.default_quat = self.sim.get_geom_quat()

        self.motor_encoder_noise = np.zeros(10)
        self.joint_encoder_noise = np.zeros(6)

    def set_up_state_space(self, command_profile, input_profile):

        full_state_est_size = 46
        min_joint_state_est_size = 35
        min_foot_state_est_size = 21
        speed_size     = 2      # x speed, y speed
        clock_size     = 2      # sin, cos
        phase_size     = 6      # ratio, coeff, shift for each phase (2 phases)

        phase_shift_inds = None

        # input --> FULL
        if input_profile == "full":
            base_mir_obs = np.array([0.1, 1, -2, 3, -4, -10, -11, 12, 13, 14, -5, -6, 7, 8, 9, 15, -16, 17, -18, 19, -20, -26, -27, 28, 29, 30, -21, -22, 23, 24, 25, 31, -32, 33, 37, 38, 39, 34, 35, 36, 43, 44, 45, 40, 41, 42])
            obs_size = full_state_est_size
        # input --> MIN (joint encoders)
        elif input_profile == "min_joint":
            base_mir_obs = np.array([
                0.01, 1, 2, 3,          # pelvis orientation
                -4, 5, -6,              # rotational vel
                -12, -13, 14, 15, 16,   # left motor pos
                -7, -8, 9, 10, 11,      # right motor pos
                -22, -23, 24, 25, 26,   # left motor vel
                -17, -18, 19, 20, 21,   # right motor vel
                29, 30, 27, 28,         # joint pos
                33, 34, 31, 32          # joint vel
            ])
            obs_size = min_joint_state_est_size
        # input --> MIN (measured foot placements)
        elif input_profile == "min_foot":
            base_mir_obs = np.array([
                3, 4, 5,            # L foot relative pos
                0.1, 1, 2,          # R foot relative pos
                6, -7, 8, -9,       # pelvis orient (quaternion)
                -10, 11, -12,       # pelvis rot Vel
                17, -18, 19, -20,   # L foot orient
                13, -14, 15, -16    # R foot orient
            ])
            obs_size = min_foot_state_est_size
        else:
            raise NotImplementedError

        # command --> CLOCK_BASED : clock, speed
        if command_profile == "clock":
            append_obs = np.array([len(base_mir_obs) + i for i in range(clock_size+speed_size)])
            mirrored_obs = np.concatenate([base_mir_obs, append_obs])
            clock_inds = append_obs[0:clock_size].tolist()
            obs_size += clock_size + speed_size
            self.phase_based = False
        # command --> PHASE_BASED : clock, phase info, speed
        elif command_profile == "phase":
            append_obs = np.array([len(base_mir_obs) + i for i in range(clock_size+phase_size+speed_size)])
            mirrored_obs = np.concatenate([base_mir_obs, append_obs])
            clock_inds = append_obs[0:clock_size].tolist()
            # NOTE: within phase info, we must flip the elements of period_shift
            phase_shift_inds = append_obs[-2:-4:-1].tolist()
            self.phase_based = True
            obs_size += clock_size + phase_size + speed_size
        else:
            raise NotImplementedError

        observation_space = np.zeros(obs_size)
        mirrored_obs = mirrored_obs.tolist()

        return observation_space, clock_inds, mirrored_obs, phase_shift_inds

    def update_gait(self):
        self.gait = [
            Phase(start=0.0, end=self.ratio[0], std=self.std[0], coeff=self.coeff[0]),
            Phase(start=self.ratio[0], end=self.ratio[0]+self.ratio[1], std=self.std[1], coeff=self.coeff[1]),
        ]

    def rotate_to_orient(self, vec):
        quaternion  = euler2quat(z=self.orient_add, y=0, x=0)
        iquaternion = inverse_quaternion(quaternion)

        if len(vec) == 3:
            return rotate_by_quaternion(vec, iquaternion)

        elif len(vec) == 4:
            new_orient = quaternion_product(iquaternion, vec)
            if new_orient[0] < 0:
                new_orient = -new_orient
            return new_orient

    def step_simulation(self, action, learned_gains=None):

        target = action + self.offset

        if self.joint_rand:
            target -= self.motor_encoder_noise

        foot_pos = np.zeros(6)
        self.sim.foot_pos(foot_pos)
        prev_foot = copy.deepcopy(foot_pos)
        self.u = pd_in_t()
        for i in range(5):

            # TODO: move setting gains out of the loop?
            # maybe write a wrapper for pd_in_t ?
            if not self.learn_gains:
                self.u.leftLeg.motorPd.pGain[i]  = self.P[i]
                self.u.rightLeg.motorPd.pGain[i] = self.P[i]
                self.u.leftLeg.motorPd.dGain[i]  = self.D[i]
                self.u.rightLeg.motorPd.dGain[i] = self.D[i]
            else:
                self.u.leftLeg.motorPd.pGain[i]  = self.P[i] + learned_gains[i]
                self.u.rightLeg.motorPd.pGain[i] = self.P[i] + learned_gains[5+i]
                self.u.leftLeg.motorPd.dGain[i]  = self.D[i] + learned_gains[10+i]
                self.u.rightLeg.motorPd.dGain[i] = self.D[i] + learned_gains[15+i]

            self.u.leftLeg.motorPd.torque[i]  = 0  # Feedforward torque
            self.u.rightLeg.motorPd.torque[i] = 0

            self.u.leftLeg.motorPd.pTarget[i]  = target[i]
            self.u.rightLeg.motorPd.pTarget[i] = target[i + 5]

            self.u.leftLeg.motorPd.dTarget[i]  = 0
            self.u.rightLeg.motorPd.dTarget[i] = 0

        self.cassie_state = self.sim.step_pd(self.u)
        self.sim.foot_pos(foot_pos)
        self.l_foot_vel = (foot_pos[0:3] - prev_foot[0:3]) / 0.0005
        self.r_foot_vel = (foot_pos[3:6] - prev_foot[3:6]) / 0.0005
        foot_forces = self.sim.get_foot_forces()
        if self.l_high and foot_forces[0] > 0:
            self.l_high = False
            self.stepcount += 1
        elif not self.l_high and foot_pos[2] >= 0.2:
            self.l_high = True
        if self.r_high and foot_forces[0] > 0:
            self.stepcount += 1
            self.r_high = False
        elif not self.r_high and foot_pos[5] >= 0.2:
            self.r_high = True

        if self.l_swing and foot_forces[0] > 0:
            self.l_swing = False
        elif not self.l_swing and foot_pos[2] >= 0:
            self.l_swing = True
        if self.r_swing and foot_forces[0] > 0:
            self.r_swing = False
        elif not self.r_swing and foot_pos[5] >= 0:
            self.r_swing = True

    # Basic version of step_simulation, that only simulates forward in time, does not do any other
    # computation for reward, etc. Is faster and should be used for evaluation purposes
    def step_sim_basic(self, action, learned_gains=None):

        target = action + self.offset

        if self.joint_rand:
            target -= self.motor_encoder_noise

        self.u = pd_in_t()
        for i in range(5):

            # TODO: move setting gains out of the loop?
            # maybe write a wrapper for pd_in_t ?
            if self.learn_gains is False:
                self.u.leftLeg.motorPd.pGain[i]  = self.P[i]
                self.u.rightLeg.motorPd.pGain[i] = self.P[i]
                self.u.leftLeg.motorPd.dGain[i]  = self.D[i]
                self.u.rightLeg.motorPd.dGain[i] = self.D[i]
            else:
                self.u.leftLeg.motorPd.pGain[i]  = self.P[i] + learned_gains[i]
                self.u.rightLeg.motorPd.pGain[i] = self.P[i] + learned_gains[5+i]
                self.u.leftLeg.motorPd.dGain[i]  = self.D[i] + learned_gains[10+i]
                self.u.rightLeg.motorPd.dGain[i] = self.D[i] + learned_gains[15+i]

            self.u.leftLeg.motorPd.torque[i]  = 0  # Feedforward torque
            self.u.rightLeg.motorPd.torque[i] = 0

            self.u.leftLeg.motorPd.pTarget[i]  = target[i]
            self.u.rightLeg.motorPd.pTarget[i] = target[i + 5]

            self.u.leftLeg.motorPd.dTarget[i]  = 0
            self.u.rightLeg.motorPd.dTarget[i] = 0

        self.cassie_state = self.sim.step_pd(self.u)

    def step(self, action, return_omniscient_state=False, f_term=0):
        
        if self.simrate_rand:
            self.simrate = int(floor(self.default_simrate + np.random.uniform(-self.small_simrate_noise_mag, self.small_simrate_noise_mag)))
        else:
            self.simrate = self.default_simrate

        # reset mujoco tracking variables
        self.l_foot_frc = 0
        self.r_foot_frc = 0
        foot_pos = np.zeros(6)
        self.l_foot_pos = np.zeros(3)
        self.r_foot_pos = np.zeros(3)
        self.l_foot_orient_cost = 0
        self.r_foot_orient_cost = 0
        self.hiproll_cost = 0
        self.hiproll_act = 0

        if self.learn_gains:
            action, learned_gains = action[0:10], action[10:]

        for i in range(self.simrate):
            if self.learn_gains:
                self.step_simulation(action, learned_gains)
            else:
                self.step_simulation(action)
            qpos = np.copy(self.sim.qpos())
            qvel = np.copy(self.sim.qvel())
            # Foot Force Tracking
            foot_forces = self.sim.get_foot_forces()
            self.l_foot_frc += foot_forces[0]
            self.r_foot_frc += foot_forces[1]
            # Relative Foot Position tracking
            self.sim.foot_pos(foot_pos)
            self.l_foot_pos += foot_pos[0:3]
            self.r_foot_pos += foot_pos[3:6]
            # Foot Orientation Cost
            self.l_foot_orient_cost += (1 - np.inner(self.neutral_foot_orient, self.sim.xquat("left-foot")) ** 2)
            self.r_foot_orient_cost += (1 - np.inner(self.neutral_foot_orient, self.sim.xquat("right-foot")) ** 2)
            # Hip Yaw velocity cost
            self.hiproll_cost += (np.abs(qvel[6]) + np.abs(qvel[19])) / 3
            if self.prev_action is not None:
                self.hiproll_act += 2*np.linalg.norm(self.prev_action[[0, 5]] - action[[0, 5]])
            else:
                self.hiproll_act += 0
        
        self.l_foot_frc              /= self.simrate
        self.r_foot_frc              /= self.simrate
        self.l_foot_pos              /= self.simrate
        self.r_foot_pos              /= self.simrate
        self.l_foot_orient_cost      /= self.simrate
        self.r_foot_orient_cost      /= self.simrate
        self.hiproll_cost            /= self.simrate
        self.hiproll_act             /= self.simrate

        height = self.sim.qpos()[2]
        self.curr_action = action

        self.time  += 1
        self.phase += self.phase_add

        if self.phase > self.phaselen:
            self.last_pelvis_pos = self.sim.qpos()[0:3]
            self.phase = 0
            self.counter += 1

        # no more knee walking
        # if self.sim.xpos("left-tarsus")[2] < 0.1 or self.sim.xpos("right-tarsus")[2] < 0.1:
        #     done = True
            # print("left tarsus: {:.2f}\tleft foot: {:.2f}".format(self.sim.xpos("left-tarsus")[2], self.sim.xpos("left-foot")[2]))
            # print("right tarsus: {:.2f}\tright foot: {:.2f}".format(self.sim.xpos("right-tarsus")[2], self.sim.xpos("right-foot")[2]))
            # while(1):
            #     self.vis.draw(self.sim)
        if height < 0.4 or height > 3.0:
            done = True
        else:
            done = False

        # make sure trackers aren't None and calculate reward
        if self.prev_action is None:
            self.prev_action = action
        if self.prev_torque is None:
            self.prev_torque = np.asarray(self.cassie_state.motor.torque[:])
        reward = self.compute_reward(action)

        # update previous action
        self.prev_action = action
        # update previous torque
        self.prev_torque = np.asarray(self.cassie_state.motor.torque[:])

        # TODO: make 0.3 a variable/more transparent
        if reward < self.early_term_cutoff:
            done = True

        if self.training:

            # random changes to orientation
            if np.random.randint(300) == 0:
                self.orient_add += np.random.uniform(-self.max_orient_change, self.max_orient_change)

            # random changes to speed
            if np.random.randint(100) == 0:
                self.speed = np.random.uniform(self.min_speed, self.max_speed)
                self.update_speed(self.speed)

            # random changes to sidespeed
            if np.random.randint(300) == 0:
                self.side_speed = np.random.uniform(self.min_side_speed, self.max_side_speed)

            # if in phase cmd mode then random changes to commanded gait
            if (np.random.randint(300) == 0) and (self.command_profile == "phase"):
                swing_ratio = np.random.uniform(self.min_swing_ratio, self.max_swing_ratio)
                # self.coeff = random.choice([[1, -1], [-1, 1]])
                self.coeff = [1, -1]  # 1: swing  -1: stance
                self.ratio = [swing_ratio, 1-swing_ratio]
                self.period_shift = random.choice([[0.0, 0.5], [0.5, 0.0], [0.0, 0.0]])
                self.update_gait()

        if return_omniscient_state:
            return self.get_full_state(), self.get_omniscient_state(), reward, done, {}
        else:
            return self.get_full_state(), reward, done, {}

    # More basic, faster version of step
    def step_basic(self, action, return_omniscient_state=False):

        if self.simrate_rand:
            self.simrate = int(floor(self.default_simrate + np.random.uniform(-self.small_simrate_noise_mag, self.small_simrate_noise_mag)))
        else:
            self.simrate = self.default_simrate

        if self.learn_gains:
            action, learned_gains = action[0:10], action[10:]

        for i in range(self.simrate):
            if self.learn_gains:
                self.step_sim_basic(action, learned_gains)
            else:
                self.step_sim_basic(action)

        self.time  += 1
        self.phase += self.phase_add

        if self.phase > self.phaselen:
            self.last_pelvis_pos = self.sim.qpos()[0:3]
            self.phase = 0
            self.counter += 1

        if return_omniscient_state:
            return self.get_full_state(), self.get_omniscient_state()
        else:
            return self.get_full_state()

    def reset(self):

        self.speed = np.random.uniform(self.min_speed, self.max_speed)
        self.side_speed = np.random.uniform(self.min_side_speed, self.max_side_speed)

        # Set up phase based
        if self.command_profile == "phase":
            # randomizing coeff, FIXING ratio/phaselen (for now), period_shift
            self.cycle_duration = 0.85
            swing_ratio = np.random.uniform(self.min_swing_ratio, self.max_swing_ratio)
            self.phaselen = self.cycle_duration * (2000 / self.default_simrate)
            # self.coeff = random.choice([[1, -1], [-1, 1]])
            self.coeff = [1, -1]  # 1: swing  -1: stance
            self.ratio = [0.5, 0.5]
            self.period_shift = random.choice([[0.0, 0.5], [0.5, 0.0], [0.0, 0.0]])
            self.update_gait()

        # ELSE load in clock based reward func
        elif self.command_profile == "clock":
            # choose ratio/phaselen based on speed, everything else constant
            self.cycle_duration = (0.9 - 0.2 / self.max_speed * abs(self.speed))  # 0.9s at 0.0 m/s, 0.7s at max abs(speed)
            swing_ratio = (0.30 + ((0.70 - 0.30) / self.max_speed) * abs(self.speed))  # 0.3 at 0.0 m/s, 0.7 at max abs(speed)
            print(swing_ratio)
            print(self.cycle_duration)
            self.phaselen = self.cycle_duration * (2000 / self.default_simrate)
            print(self.phaselen)
            self.coeff = [1, -1]  # 1: swing  -1: stance
            self.ratio = [swing_ratio, 1-swing_ratio]
            self.update_gait()

        self.clock1 = vonmises_func(np.arange(0, self.phaselen+self.phase_add) / self.phaselen, self.gait, shift=self.period_shift[0])
        self.clock2 = vonmises_func(np.arange(0, self.phaselen+self.phase_add) / self.phaselen, self.gait, shift=self.period_shift[1])
        # self.clock1 = vonmises_func(np.linspace(0, 1, num=floor(self.phaselen)+1), self.gait, shift=self.period_shift[0])
        # self.clock2 = vonmises_func(np.linspace(0, 1, num=floor(self.phaselen)+1), self.gait, shift=self.period_shift[1])

        self.phase = random.randint(0, floor(self.phaselen))
        self.time = 0
        self.counter = 0

        self.state_history = [np.zeros(self._obs) for _ in range(self.history+1)]

        # Randomize dynamics:
        if self.dynamics_randomization:
            damp = self.default_damping

            pelvis_damp_range = [[damp[0], damp[0]],
                                [damp[1], damp[1]],
                                [damp[2], damp[2]],
                                [damp[3], damp[3]],
                                [damp[4], damp[4]],
                                [damp[5], damp[5]]]  # 0->5

            hip_damp_range = [[damp[6]*self.damping_low, damp[6]*self.damping_high],
                              [damp[7]*self.damping_low, damp[7]*self.damping_high],
                              [damp[8]*self.damping_low, damp[8]*self.damping_high]]          # 6->8 and 19->21

            achilles_damp_range = [[damp[9]*self.damping_low, damp[9]*self.damping_high],
                                   [damp[10]*self.damping_low, damp[10]*self.damping_high],
                                   [damp[11]*self.damping_low, damp[11]*self.damping_high]]   # 9->11 and 22->24

            knee_damp_range     = [[damp[12]*self.damping_low, damp[12]*self.damping_high]]   # 12 and 25
            shin_damp_range     = [[damp[13]*self.damping_low, damp[13]*self.damping_high]]   # 13 and 26
            tarsus_damp_range   = [[damp[14]*self.damping_low, damp[14]*self.damping_high]]   # 14 and 27

            heel_damp_range     = [[damp[15], damp[15]]]                                      # 15 and 28
            fcrank_damp_range   = [[damp[16]*self.damping_low, damp[16]*self.damping_high]]   # 16 and 29
            prod_damp_range     = [[damp[17], damp[17]]]                                      # 17 and 30
            foot_damp_range     = [[damp[18]*self.damping_low, damp[18]*self.damping_high]]   # 18 and 31

            side_damp = hip_damp_range + achilles_damp_range + knee_damp_range + shin_damp_range + tarsus_damp_range + heel_damp_range + fcrank_damp_range + prod_damp_range + foot_damp_range
            damp_range = pelvis_damp_range + side_damp + side_damp
            damp_noise = [np.random.uniform(a, b) for a, b in damp_range]

            m = self.default_mass
            pelvis_mass_range      = [[self.mass_low*m[1], self.mass_high*m[1]]]   # 1
            hip_mass_range         = [[self.mass_low*m[2], self.mass_high*m[2]],   # 2->4 and 14->16
                                    [self.mass_low*m[3], self.mass_high*m[3]],
                                    [self.mass_low*m[4], self.mass_high*m[4]]]

            achilles_mass_range    = [[self.mass_low*m[5], self.mass_high*m[5]]]    # 5 and 17
            knee_mass_range        = [[self.mass_low*m[6], self.mass_high*m[6]]]    # 6 and 18
            knee_spring_mass_range = [[self.mass_low*m[7], self.mass_high*m[7]]]    # 7 and 19
            shin_mass_range        = [[self.mass_low*m[8], self.mass_high*m[8]]]    # 8 and 20
            tarsus_mass_range      = [[self.mass_low*m[9], self.mass_high*m[9]]]    # 9 and 21
            heel_spring_mass_range = [[self.mass_low*m[10], self.mass_high*m[10]]]  # 10 and 22
            fcrank_mass_range      = [[self.mass_low*m[11], self.mass_high*m[11]]]  # 11 and 23
            prod_mass_range        = [[self.mass_low*m[12], self.mass_high*m[12]]]  # 12 and 24
            foot_mass_range        = [[self.mass_low*m[13], self.mass_high*m[13]]]  # 13 and 25

            side_mass = hip_mass_range + achilles_mass_range \
                        + knee_mass_range + knee_spring_mass_range \
                        + shin_mass_range + tarsus_mass_range \
                        + heel_spring_mass_range + fcrank_mass_range \
                        + prod_mass_range + foot_mass_range

            mass_range = [[0, 0]] + pelvis_mass_range + side_mass + side_mass
            mass_noise = [np.random.uniform(a, b) for a, b in mass_range]

            delta = 0.0
            com_noise = [0, 0, 0] + [np.random.uniform(val - delta, val + delta) for val in self.default_ipos[3:]]

            fric_noise = []
            translational = np.random.uniform(self.fric_low, self.fric_high)
            torsional = np.random.uniform(1e-4, 5e-4)
            rolling = np.random.uniform(1e-4, 2e-4)
            for _ in range(int(len(self.default_fric)/3)):
                fric_noise += [translational, torsional, rolling]

            self.sim.set_dof_damping(np.clip(damp_noise, 0, None))
            self.sim.set_body_mass(np.clip(mass_noise, 0, None))
            self.sim.set_body_ipos(com_noise)
            self.sim.set_geom_friction(np.clip(fric_noise, 0, None))
        else:
            self.sim.set_body_mass(self.default_mass)
            self.sim.set_body_ipos(self.default_ipos)
            self.sim.set_dof_damping(self.default_damping)
            self.sim.set_geom_friction(self.default_fric)

        if self.slope_rand:
            geom_plane = [np.random.uniform(-self.max_roll_incline, self.max_roll_incline), np.random.uniform(-self.max_pitch_incline, self.max_pitch_incline), 0]
            quat_plane   = euler2quat(z=geom_plane[2], y=geom_plane[1], x=geom_plane[0])
            geom_quat  = list(quat_plane) + list(self.default_quat[4:])
            self.sim.set_geom_quat(geom_quat)
        else:
            self.sim.set_geom_quat(self.default_quat)

        if self.simrate_rand:
            self.simrate = np.random.uniform(self.min_simrate, self.max_simrate)

        if self.joint_rand:
            self.motor_encoder_noise = np.random.uniform(-self.encoder_noise, self.encoder_noise, size=10)
            self.joint_encoder_noise = np.random.uniform(-self.encoder_noise, self.encoder_noise, size=6)
        else:
            self.motor_encoder_noise = np.zeros(10)
            self.joint_encoder_noise = np.zeros(6)

        # apply dynamics
        self.sim.set_const()

        ## Reset to starting position via IK
        self.sim.set_qpos(self.IK_solver(*get_sample_pos()))

        self.last_pelvis_pos = self.sim.qpos()[0:3]

        # Need to reset u? Or better way to reset cassie_state than taking step
        self.cassie_state = self.sim.step_pd(self.u)

        # reset commands
        self.orient_add = 0  # random.randint(-10, 10) * np.pi / 25
        self.speed = np.random.uniform(self.min_speed, self.max_speed)
        self.side_speed = np.random.uniform(self.min_side_speed, self.max_side_speed)

        # reset mujoco tracking variables
        self.l_foot_frc = 0
        self.r_foot_frc = 0
        self.l_foot_orient_cost = 0
        self.r_foot_orient_cost = 0
        self.hiproll_cost = 0
        self.hiproll_act = 0

        return self.get_full_state()

    def reset_for_test(self, full_reset=False):
        self.phase = 0
        self.time = 0
        self.counter = 0
        self.orient_add = 0
        self.phase_add = 1

        self.state_history = [np.zeros(self._obs) for _ in range(self.history+1)]

        self.speed = 0
        self.coeff = [1, -1]
        self.ratio = [0.5, 0.5]
        self.period_shift = [0.0, 0.5]
        self.std = [0.2, 0.2]
        self.update_gait()

        self.clock1 = vonmises_func(np.arange(0, self.phaselen+self.phase_add) / self.phaselen, self.gait, shift=self.period_shift[0])
        self.clock2 = vonmises_func(np.arange(0, self.phaselen+self.phase_add) / self.phaselen, self.gait, shift=self.period_shift[1])
        # self.clock1 = vonmises_func(np.linspace(0, 1, num=floor(self.phaselen)+1), self.gait, shift=self.period_shift[0])
        # self.clock2 = vonmises_func(np.linspace(0, 1, num=floor(self.phaselen)+1), self.gait, shift=self.period_shift[1])

        if not full_reset:

            # reset mujoco tracking variables
            self.last_pelvis_pos = self.sim.qpos()[0:3]
            self.l_foot_frc = 0
            self.r_foot_frc = 0
            self.l_foot_orient = 0
            self.r_foot_orient = 0

            # Need to reset u? Or better way to reset cassie_state than taking step
            self.cassie_state = self.sim.step_pd(self.u)
        else:
            self.sim.full_reset()
            self.reset_cassie_state()

        if self.dynamics_randomization:
            self.sim.set_dof_damping(self.default_damping)
            self.sim.set_body_mass(self.default_mass)
            self.sim.set_body_ipos(self.default_ipos)
            self.sim.set_geom_friction(self.default_fric)
            self.sim.set_const()

        if self.slope_rand:
            self.sim.set_geom_quat(np.array([1, 0, 0, 0]), "floor")

        if self.joint_rand:
            self.motor_encoder_noise = np.zeros(10)
            self.joint_encoder_noise = np.zeros(6)

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
    # not needed in training cause we don't change speeds in middle of rollout, and
    # we randomize the starting phase of each rollout
    def update_speed(self, new_speed, new_side_speed=None):

        self.speed = np.clip(new_speed, self.min_speed, self.max_speed)
        if new_side_speed is not None:
            self.side_speed = np.clip(new_side_speed, self.min_side_speed, self.max_side_speed)
        
        if self.command_profile == "clock":
            # choose ratio/phaselen based on speed
            self.cycle_duration = (0.9 - 0.2 / self.max_speed * abs(self.speed))  # 0.9s at 0.0 m/s, 0.7s at max abs(speed)
            swing_ratio = (0.30 + ((0.70 - 0.30) / self.max_speed) * abs(self.speed))  # 0.3 at 0.0 m/s, 0.7 at max abs(speed)
            self.phaselen = self.cycle_duration * (2000 / self.default_simrate)
            self.ratio = [swing_ratio, 1-swing_ratio]
            self.update_gait()
        else:
            pass

    def compute_reward(self, action):

        if self.reward_func == "clock":
            self.early_term_cutoff = -99.
            if self.early_reward:
                return early_clock_reward(self, action)
            else:
                return clock_reward(self, action)
        elif self.reward_func == "no_speed_clock":
            self.early_term_cutoff = -99.
            return no_speed_clock_reward(self, action)
        elif self.reward_func == "max_vel_clock":
            self.early_term_cutoff = -99.
            return max_vel_clock_reward(self, action)
        else:
            raise NotImplementedError

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

        speed_input = [self.speed] if not self.has_side_speed else [self.speed, self.side_speed]

        # command --> PHASE_BASED : clock, phase info, speed
        if self.command_profile == "phase":
            clock = [np.sin(2 * np.pi * self.phase / self.phaselen),
                    np.cos(2 * np.pi * self.phase / self.phaselen)]
            ext_state = np.concatenate((clock, [*self.coeff, *self.ratio, *self.period_shift, *speed_input]))
        # command --> CLOCK_BASED : clock, speed
        elif self.command_profile == "clock":
            clock = [np.sin(2 * np.pi * self.phase / self.phaselen),
                    np.cos(2 * np.pi * self.phase / self.phaselen)]
            ext_state = np.concatenate((clock, speed_input))
        else:
            raise NotImplementedError

        # Update orientation
        new_orient = self.rotate_to_orient(self.cassie_state.pelvis.orientation[:])
        pelvis_height = [self.cassie_state.pelvis.position[2] - self.cassie_state.terrain.height]
        pelvis_vel = self.rotate_to_orient(self.cassie_state.pelvis.translationalVelocity[:])
        pelvis_rvel = self.cassie_state.pelvis.rotationalVelocity[:]
        pelvis_accel = self.rotate_to_orient(self.cassie_state.pelvis.translationalAcceleration[:])

        # motor and joint poses
        if self.joint_rand:
            motor_pos = self.cassie_state.motor.position[:] + self.motor_encoder_noise
            joint_pos = self.cassie_state.joint.position[:] + self.joint_encoder_noise
        else:
            motor_pos = self.cassie_state.motor.position[:]
            joint_pos = self.cassie_state.joint.position[:]

        motor_vel = self.cassie_state.motor.velocity[:]
        joint_vel = self.cassie_state.joint.velocity[:]

        if self.input_profile == "min_foot":
            robot_state = np.concatenate([
                self.cassie_state.leftFoot.position[:],             # left foot position
                self.cassie_state.rightFoot.position[:],            # right foot position
                new_orient[:],                                      # pelvis orientation
                pelvis_rvel,                                        # pelvis rotational velocity
                self.cassie_state.leftFoot.orientation[:],          # left foot orientation
                self.cassie_state.rightFoot.orientation[:]          # right foot orientation
            ])
        elif self.input_profile == "min_joint":
            # remove double-counted joint/motor positions
            joint_pos = np.concatenate([joint_pos[:2], joint_pos[3:5]])
            joint_vel = np.concatenate([joint_vel[:2], joint_vel[3:5]])
            robot_state = np.concatenate([
                new_orient[:],   # pelvis orientation
                pelvis_rvel,     # pelvis rotational velocity
                motor_pos,       # actuated joint positions
                motor_vel,       # actuated joint velocities
                joint_pos,       # unactuated joint positions
                joint_vel        # unactuated joint velocities
            ])
        elif self.input_profile == "full":
            robot_state = np.concatenate([
                pelvis_height,  # pelvis height
                new_orient[:],   # pelvis orientation
                motor_pos,       # actuated joint positions
                pelvis_vel,      # pelvis translational velocity
                pelvis_rvel,     # pelvis rotational velocity
                motor_vel,       # actuated joint velocities
                pelvis_accel,    # pelvis translational acceleration
                joint_pos,       # unactuated joint positions
                joint_vel        # unactuated joint velocities
            ])
        else:
            raise NotImplementedError

        state = np.concatenate([robot_state, ext_state])

        self.state_history.insert(0, state)
        self.state_history = self.state_history[:self.history+1]

        return np.concatenate(self.state_history)

    def render(self):
        if self.vis is None:
            self.vis = CassieVis(self.sim, self.config)

        return self.vis.draw(self.sim)


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
