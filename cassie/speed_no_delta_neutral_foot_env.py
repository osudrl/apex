from .cassiemujoco import pd_in_t, state_out_t, CassieSim, CassieVis

from .trajectory import CassieTrajectory
from cassie.quaternion_function import *
from .rewards import *

from math import floor

import numpy as np 
import os
import random
import copy

import pickle

class CassieIKTrajectory:
    def __init__(self, filepath):
        with open(filepath, "rb") as f:
            trajectory = pickle.load(f)

        self.qpos = np.copy(trajectory["qpos"])
        self.qvel = np.copy(trajectory["qvel"])
        #self.foot =
    
    def __len__(self):
        return len(self.qpos)

class CassieFootTrajectory:
    def __init__(self, filepath):
        with open(filepath, "rb") as f:
            trajectory = pickle.load(f)

        self.rfoot = trajectory["rfoot"]
        self.lfoot = trajectory["lfoot"]
        self.rfoot_vel = trajectory["rfoot_vel"]
        self.lfoot_vel = trajectory["lfoot_vel"]

class CassieTraj:
    def __init__(self, filepath):
        data = np.load(filepath)

        # states
        # self.time = data[:, 0]
        self.qpos = data#[:, 1:36]
        
        self.len = data.shape[0]
        self.qvel = np.zeros((self.len, 32))
        # self.qvel = data[:, 36:68]
    
    def __len__(self):
        return self.len

class CassieEnv_speed_no_delta_neutral_foot:
    def __init__(self, traj, simrate=60, clock_based=False, state_est=False):
        self.sim = CassieSim("./cassie/cassiemujoco/cassie.xml")
        self.vis = None

        self.clock_based = clock_based
        self.state_est = state_est

        if clock_based:
            self.observation_space = np.zeros(42 + 1)
            if self.state_est:
                self.observation_space = np.zeros(48 + 1)       # Size for use with state est
            self.ext_size = 3   # Size of ext_state input, used when constructing mirror obs vector
        else:
            self.observation_space = np.zeros(80)
            if self.state_est:
                self.observation_space = np.zeros(86)       # Size for use with state est
            self.ext_size = 1   # Size of ext_state input, used when constructing mirror obs vector
        self.action_space      = np.zeros(10)

        dirname = os.path.dirname(__file__)
        if traj == "walking":
            traj_path = os.path.join(dirname, "trajectory", "stepdata.bin")

        elif traj == "stepping":
            # traj_path = os.path.join(dirname, "trajectory", "spline_stepping_traj.pkl")
            traj_path = os.path.join(dirname, "trajectory", "more-poses-trial.bin")

        # self.trajectory = CassieIKTrajectory(traj_path)
        # self.trajectory = CassieTrajectory(traj_path)
        self.trajectory = CassieTraj(os.path.join(dirname, "trajectory", "iktraj_land0.4_speed1.0_fixedheightfreq_fixedtdvel_fixedfoot.npy"))
        # self.foot_traj = CassieFootTrajectory(os.path.join(dirname, "trajectory", "foottraj_doublestance_time0.4_land1.0_h0.2.pkl"))
        self.foot_traj = CassieFootTrajectory(os.path.join(dirname, "trajectory", "foottraj_land0.4_speed1.0_fixedheightfreq_fixedtdvel.pkl"))

        self.P = np.array([100,  100,  88,  96,  50]) 
        self.D = np.array([10.0, 10.0, 8.0, 9.6, 5.0])

        self.u = pd_in_t()

        # TODO: should probably initialize this to current state
        self.cassie_state = state_out_t()

        self.simrate = simrate # simulate X mujoco steps with same pd target
                               # 60 brings simulation from 2000Hz to roughly 30Hz

        self.time    = 0 # number of time steps in current episode
        self.phase   = 0 # portion of the phase the robot is in
        self.counter = 0 # number of phase cycles completed in episode

        # NOTE: a reference trajectory represents ONE phase cycle

        # should be floor(len(traj) / simrate) - 1
        # should be VERY cautious here because wrapping around trajectory
        # badly can cause assymetrical/bad gaits
        self.phaselen = floor(len(self.trajectory) / self.simrate) - 1

        # see include/cassiemujoco.h for meaning of these indices
        self.pos_idx = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        self.vel_idx = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]

        self.base_mirror_obs = [0.1, 1, 2, 3, 4, -10, -11, 12, 13, 14, -5, -6, 7, 8, 9, 15,
                            16, 17, 18, 19, 20, -26, -27, 28, 29, 30, -21, -22, 23, 24,
                            25, 31, 32, 33, 37, 38, 39, 34, 35, 36, 43, 44, 45, 40, 41, 42]

        self.speed = 1
        # maybe make ref traj only send relevant idxs?
        ref_pos, ref_vel = self.get_ref_state(self.phase)
        self.phase_add = 1
        self.l_high = False
        self.r_high = False
        self.lfoot_vel = np.zeros(3)
        self.rfoot_vel = np.zeros(3)
        self.l_foot_diff = 0
        self.r_foot_diff = 0
        self.l_footvel_diff = 0
        self.r_footvel_diff = 0
        self.l_foot_orient = 0
        self.r_foot_orient = 0
        self.com_error         = 0
        self.com_vel_error     = 0
        self.orientation_error = 0
        self.joint_error = 0
        self.torque_cost = 0
        self.smooth_cost = 0

        #### Dynamics Randomization ####
        self.dynamics_rand = False
        self.slope_rand = False
        self.joint_rand = True
        # Record default dynamics parameters
        if self.dynamics_rand:
            self.default_damping = self.sim.get_dof_damping()
            self.default_mass = self.sim.get_body_mass()
            self.default_ipos = self.sim.get_body_ipos()
            self.default_fric = self.sim.get_geom_friction()

            weak_factor = 0.5
            strong_factor = 0.5

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

            hi = 1.1
            lo = 0.9
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

        # self.delta_x_min, self.delta_x_max = self.default_ipos[3] - 0.05, self.default_ipos[3] + 0.05
        # self.delta_y_min, self.delta_y_max = self.default_ipos[4] - 0.05, self.default_ipos[4] + 0.05
        self.joint_offsets = np.zeros(16)
        self.com_vel_offset = 0
        
        self.speed_schedule = np.zeros(4)
        self.orient_add = 0
        self.orient_time = 500
        self.prev_action = None
        self.curr_action = None
        self.prev_torque = None
        self.y_offset = 0
        self.sim.set_geom_friction([0.6, 1e-4, 5e-5], "floor")

        if self.state_est:
            self.clock_inds = [46, 47]
        else:
            self.clock_inds = [40, 41]
    

    def step_simulation(self, action):

        real_action = action
        offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])
        real_action = real_action + offset
        # real_action[4] += -1.5968
        # real_action[9] += -1.5968
        
        if self.joint_rand:
            real_action -= self.joint_offsets[0:10]

        # target = action + ref_pos[self.pos_idx]
        foot_pos = np.zeros(6)
        self.sim.foot_pos(foot_pos)
        prev_foot = copy.deepcopy(foot_pos)
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

            self.u.leftLeg.motorPd.pTarget[i]  = real_action[i]
            self.u.rightLeg.motorPd.pTarget[i] = real_action[i + 5]

            self.u.leftLeg.motorPd.dTarget[i]  = 0
            self.u.rightLeg.motorPd.dTarget[i] = 0

        self.cassie_state = self.sim.step_pd(self.u)
        self.sim.foot_pos(foot_pos)
        self.lfoot_vel = (foot_pos[0:3] - prev_foot[0:3]) / 0.0005
        self.rfoot_vel = (foot_pos[3:6] - prev_foot[3:6]) / 0.0005
        foot_forces = self.sim.get_foot_forces()
        if self.l_high and foot_forces[0] > 0:
            self.l_high = False
        elif not self.l_high and foot_pos[2] >= 0.2:
            self.l_high = True
        if self.r_high and foot_forces[0] > 0:
            self.r_high = False
        elif not self.r_high and foot_pos[5] >= 0.2:
            self.r_high = True

    def step(self, action):    
        foot_pos = np.zeros(6)
        self.l_foot_diff = 0
        self.r_foot_diff = 0
        self.l_footvel_diff = 0
        self.r_footvel_diff = 0
        self.l_foot_orient = 0
        self.r_foot_orient = 0
        self.com_error         = 0
        self.com_vel_error     = 0
        self.orientation_error = 0
        self.joint_error = 0
        self.torque_cost = 0
        self.smooth_cost = 0

        ref_pos, ref_vel = self.get_ref_state(self.phase+self.phase_add)
        weight = [0.15, 0.15, 0.1, 0.05, 0.05, 0.15, 0.15, 0.1, 0.05, 0.05]
        neutral_foot_orient = np.array([-0.24790886454547323, -0.24679713195445646, -0.6609396704367185, 0.663921021343526])
        for i in range(self.simrate):
            self.step_simulation(action)
            qpos = np.copy(self.sim.qpos())
            qvel = np.copy(self.sim.qvel())
            # ref_lpos, ref_rpos, ref_lvel, ref_rvel = self.get_ref_foot(self.phase, i+1)
            # Calculate foot pos and vel diff
            # self.sim.foot_pos(foot_pos)
            # lfoot = np.copy(self.foot_traj.lfoot[int(self.phase * self.simrate) + i+1, :])
            # rfoot = np.copy(self.foot_traj.rfoot[int(self.phase * self.simrate) + i+1, :])
            # lfoot_vel = np.copy(self.foot_traj.lfoot_vel[int(self.phase * self.simrate) + i+1, :])
            # rfoot_vel = np.copy(self.foot_traj.rfoot_vel[int(self.phase * self.simrate) + i+1, :])
            # self.l_foot_diff += (np.linalg.norm(foot_pos[0:3] - lfoot) - self.l_foot_diff) / (i+1)
            # self.r_foot_diff += (np.linalg.norm(foot_pos[3:6] - rfoot) - self.r_foot_diff) / (i+1)
            # self.l_footvel_diff += (np.linalg.norm(self.lfoot_vel - lfoot_vel) - self.l_footvel_diff) / (i+1)
            # self.r_footvel_diff += (np.linalg.norm(self.rfoot_vel - rfoot_vel) - self.r_footvel_diff) / (i+1)
            # self.l_foot_diff += np.linalg.norm(foot_pos[0:3] - ref_lpos)
            # self.r_foot_diff += np.linalg.norm(foot_pos[3:6] - ref_rpos)
            # self.l_footvel_diff += np.linalg.norm(self.lfoot_vel - ref_lvel)
            # self.r_footvel_diff += np.linalg.norm(self.rfoot_vel - ref_rvel)

            # Calculate qpos diffs
            # ref_pos = np.copy(self.trajectory.qpos[int(self.phase*self.simrate)+i+1])
            # ref_pos[0] *= self.speed
            # ref_pos[0] += (self.trajectory.qpos[-1, 0]- self.trajectory.qpos[0, 0])* self.counter * self.speed

            ##### Ref traj errors #####
            # ref_pos = np.copy(self.trajectory.qpos[int(self.phase*self.simrate)+i+1])
            # ref_pos[0] = self.speed * self.counter + (self.speed / self.phaselen)*self.phase
            # ref_pos[1] = self.side_speed * self.counter + (self.side_speed / self.phaselen)*self.phase
            #####                 #####
            # each joint pos
            for i, j in enumerate(self.pos_idx):
                self.joint_error += 30 * weight[i] * (ref_pos[j] - qpos[j]) ** 2
            # left and right shin springs
            # for i in [15, 29]:
            #     # NOTE: in Xie et al spring target is 0
            #     self.spring_error += 1000 * (ref_pos[i] - qpos[i]) ** 2  

            # center of mass: x, y, z
            # self.com_error += np.inner(ref_pos[0:3] - qpos[0:3], ref_pos[0:3] - qpos[0:3])
            # self.com_vel_error += np.abs(qvel[0] - self.speed)
            # self.orientation_error += np.inner(ref_pos[3:7] - qpos[3:7], ref_pos[3:7] - qpos[3:7])

            # curr_l_yfoot = np.abs(foot_pos[2] - self.foot_traj.lfoot[int(self.phase*self.simrate) + i+1, 2])
            # curr_r_yfoot = np.abs(foot_pos[5] - self.foot_traj.rfoot[int(self.phase*self.simrate) + i+1, 2])
            # self.l_foot_diff += (curr_l_yfoot - self.l_foot_diff) / (i + 1)
            # self.r_foot_diff += (curr_r_yfoot - self.r_foot_diff) / (i + 1)
            # curr_l_yfoot_vel = np.abs(self.lfoot_vel - self.foot_traj.lfoot_vel[int(self.phase*self.simrate) + i+1])
            # curr_r_yfoot_vel = np.abs(self.rfoot_vel - self.foot_traj.rfoot_vel[int(self.phase*self.simrate) + i+1])
            # self.l_footvel_diff += (curr_l_yfoot_vel - self.l_footvel_diff) / (i + 1)
            # self.r_footvel_diff += (curr_r_yfoot_vel - self.r_footvel_diff) / (i + 1)

            # Foot Orientation Cost
            self.l_foot_orient += 100*(1 - np.inner(neutral_foot_orient, self.sim.xquat("left-foot")) ** 2)
            self.r_foot_orient += 100*(1 - np.inner(neutral_foot_orient, self.sim.xquat("right-foot")) ** 2)

            # Torque costs
            curr_torques = np.array(self.cassie_state.motor.torque[:])
            self.torque_cost += 0.00004*np.linalg.norm(np.square(curr_torques))
            if self.prev_torque is not None:
                self.smooth_cost += 0.0001*np.linalg.norm(np.square(curr_torques - self.prev_torque))
            else:
                self.smooth_cost += 0
            self.prev_torque = curr_torques


        # self.com_error          /= self.simrate 
        # self.com_vel_error      /= self.simrate
        # self.orientation_error  /= self.simrate 
        # self.l_foot_diff        /= self.simrate
        # self.r_foot_diff        /= self.simrate
        # self.l_footvel_diff     /= self.simrate
        # self.r_footvel_diff     /= self.simrate
        self.l_foot_orient      /= self.simrate
        self.r_foot_orient      /= self.simrate
        self.torque_cost        /= self.simrate
        self.smooth_cost        /= self.simrate
        self.joint_error        /= self.simrate

        height = self.sim.qpos()[2]
        self.curr_action = action
        
        self.time  += 1
        self.phase += self.phase_add
        # Assuming max traj len when sampling is 400
        # self.speed = self.speed_schedule[min(int(np.floor(self.time/100)), 2)]

        if self.phase > self.phaselen:
            self.phase = 0
            self.counter += 1

        # Early termination
        done = not(height > 0.4 and height < 3.0)

        reward = self.compute_reward()
        self.prev_action = action

        # TODO: make 0.3 a variable/more transparent
        # if reward < 0.3:
            # done = True
            # print("reward too low")

        return self.get_full_state(), reward, done, {}

    def reset(self):
        self.phase = random.randint(0, self.phaselen)
        self.time = 0
        self.counter = 0

        qpos, qvel = self.get_ref_state(self.phase)
        orientation = random.randint(-10, 10) * np.pi / 25
        quaternion = euler2quat(z=orientation, y=0, x=0)
        qpos[3:7] = quaternion
        self.y_offset = 0#random.uniform(-3.5, 3.5)
        qpos[1] = self.y_offset

        self.sim.set_qpos(qpos)
        self.sim.set_qvel(qvel)

        # Need to reset u? Or better way to reset cassie_state than taking step
        self.cassie_state = self.sim.step_pd(self.u)

        self.speed = (random.randint(0, 30)) / 10
        # self.speed_schedule = np.random.randint(0, 30, size=3) / 10
        # self.speed = self.speed_schedule[0]
        # Make sure that if speed is above 2, freq is at least 1.2
        if self.speed > 1.3:# or np.any(self.speed_schedule > 1.6):
            self.phase_add = 1.3 + 0.7*random.random()
        else:
            self.phase_add = 1 + random.random()
        # self.phase_add = 1
        self.orient_add = 0#random.randint(-10, 10) * np.pi / 25
        self.orient_time = 500#random.randint(50, 200) 
        self.com_vel_offset = 0#0.1*np.random.uniform(-0.1, 0.1, 2)
        self.lfoot_vel = np.zeros(3)
        self.rfoot_vel = np.zeros(3)
        self.l_foot_diff = 0
        self.r_foot_diff = 0
        self.l_footvel_diff = 0
        self.r_footvel_diff = 0
        self.l_foot_orient = 0
        self.r_foot_orient = 0
        self.torque_cost = 0
        self.smooth_cost = 0
        self.joint_error = 0
        self.prev_action = None
        self.prev_torque = None

        if self.dynamics_rand:
            #### Dynamics Randomization ####
            damp_noise = [np.random.uniform(a, b) for a, b in self.damp_range]
            # mass_noise = [np.random.uniform(a, b) for a, b in self.mass_range]
            # com_noise = [0, 0, 0] + [np.random.uniform(self.delta_x_min, self.delta_x_min)] + [np.random.uniform(self.delta_y_min, self.delta_y_max)] + [0] + list(self.default_ipos[6:])
            # fric_noise = [np.random.uniform(0.95, 1.05)] + [np.random.uniform(5e-4, 5e-3)] + [np.random.uniform(5e-5, 5e-4)]#+ list(self.default_fric[2:])
            # fric_noise = []
            # translational = np.random.uniform(0.6, 1.2)
            # torsional = np.random.uniform(1e-4, 1e-2)
            # rolling = np.random.uniform(5e-5, 5e-4)
            # for _ in range(int(len(self.default_fric)/3)):
            #     fric_noise += [translational, torsional, rolling]
            # fric_noise = [np.random.uniform(0.6, 1.2), np.random.uniform(1e-4, 1e-2), np.random.uniform(5e-5, 5e-4)]
            self.sim.set_dof_damping(np.clip(damp_noise, 0, None))
            # self.sim.set_body_mass(np.clip(mass_noise, 0, None))
            # self.sim.set_body_ipos(com_noise)
            # self.sim.set_geom_friction(np.clip(fric_noise, 0, None), "floor")

            self.sim.set_const()
        if self.slope_rand:
            rand_angle = np.pi/180*np.random.uniform(-5, 5, 2)
            floor_quat = euler2quat(z=0, y=rand_angle[0], x=rand_angle[1])
            self.sim.set_geom_quat(floor_quat, "floor")
        if self.joint_rand:
            self.joint_offsets = np.random.uniform(-0.03, 0.03, 16)


        return self.get_full_state()

    # used for plotting against the reference trajectory
    def reset_for_test(self):
        self.phase = 0
        self.time = 0
        self.counter = 0
        self.speed = 1
        self.speed_schedule = self.speed * np.ones(3)
        self.orient_add = 0
        self.orient_time = 500
        self.y_offset = 0
        self.phase_add = 1

        qpos, qvel = self.get_ref_state(self.phase)

        self.sim.set_qpos(qpos)
        self.sim.set_qvel(qvel)

        # self.sim.reset()

        # Need to reset u? Or better way to reset cassie_state than taking step
        self.cassie_state = self.sim.step_pd(self.u)
        self.lfoot_vel = np.zeros(3)
        self.rfoot_vel = np.zeros(3)
        self.l_foot_diff = 0
        self.r_foot_diff = 0
        self.l_footvel_diff = 0
        self.r_footvel_diff = 0
        self.l_foot_orient = 0
        self.r_foot_orient = 0
        self.torque_cost = 0
        self.smooth_cost = 0
        self.joint_error = 0
        self.prev_action = None
        self.prev_torque = None

        if self.dynamics_rand:
            self.sim.set_dof_damping(self.default_damping)
            # self.sim.set_body_mass(self.default_mass)
            # self.sim.set_body_ipos(self.default_ipos)
            # self.sim.set_ground_friction(self.default_fric)
            self.sim.set_const()
        if self.slope_rand:
            self.sim.set_geom_quat(np.array([1, 0, 0, 0]), "floor")


        return self.get_full_state()
    
    def set_joint_pos(self, jpos, fbpos=None, iters=5000):
        """
        Kind of hackish. 
        This takes a floating base position and some joint positions
        and abuses the MuJoCo solver to get the constrained forward
        kinematics. 

        There might be a better way to do this, e.g. using mj_kinematics
        """

        # actuated joint indices
        joint_idx = [7, 8, 9, 14, 20,
                     21, 22, 23, 28, 34]

        # floating base indices
        fb_idx = [0, 1, 2, 3, 4, 5, 6]

        for _ in range(iters):
            qpos = np.copy(self.sim.qpos())
            qvel = np.copy(self.sim.qvel())

            qpos[joint_idx] = jpos

            if fbpos is not None:
                qpos[fb_idx] = fbpos

            self.sim.set_qpos(qpos)
            self.sim.set_qvel(0 * qvel)

            self.sim.step_pd(pd_in_t())


    # NOTE: this reward is slightly different from the one in Xie et al
    # see notes for details
    def compute_reward(self):
    
        # reward = trajmatch_reward(self)
        # reward = speedmatch_reward(self)
        reward = speedmatch_footorient_joint_smooth_reward(self)

        return reward

    def get_ref_foot(self, phase, cycle_ind):
        if phase is None: 
            phase = self.phase
        if phase > self.phaselen:
            phase = 0

        # Copy data from foot traj
        l_pos = np.copy(self.foot_traj.lfoot[int(phase*self.simrate) + cycle_ind, :])
        r_pos = np.copy(self.foot_traj.rfoot[int(phase*self.simrate) + cycle_ind, :])
        l_vel = np.copy(self.foot_traj.lfoot_vel[int(phase*self.simrate) + cycle_ind, :])
        r_vel = np.copy(self.foot_traj.rfoot_vel[int(phase*self.simrate) + cycle_ind, :])

        # Setting variable speed
        l_pos[0] *= self.speed
        l_pos[0] += (self.foot_traj.lfoot[-1, 0] - self.foot_traj.lfoot[0, 0]) * self.counter * self.speed
        r_pos[0] *= self.speed
        r_pos[0] += (self.foot_traj.rfoot[-1, 0] - self.foot_traj.rfoot[0, 0]) * self.counter * self.speed

        l_vel[0] *= self.speed
        r_vel[0] *= self.speed

        return l_pos, r_pos, l_vel, r_vel

    # get the corresponding state from the reference trajectory for the current phase
    def get_ref_state(self, phase=None):
        if phase is None:
            phase = self.phase

        if phase > self.phaselen:
            phase = 0

        desired_ind = phase * self.simrate
        # phase_diff = desired_ind - math.floor(desired_ind)
        # if phase_diff != 0:       # desired ind is an int
        #     pos_prev = np.copy(self.trajectory.qpos[math.floor(desired_ind)])
        #     vel_prev = np.copy(self.trajectory.qvel[math.floor(desired_ind)])
        #     pos_next = np.copy(self.trajectory.qpos[math.ceil(desired_ind)])
        #     vel_next = np.copy(self.trajectory.qvel[math.ceil(desired_ind)])
        #     pos = pos_prev + phase_diff * (pos_next - pos_prev)
        #     vel = vel_prev + phase_diff * (vel_next - vel_prev)
        # else:
        # print("desired ind: ", desired_ind)
        pos = np.copy(self.trajectory.qpos[int(desired_ind)])
        vel = np.copy(self.trajectory.qvel[int(desired_ind)])

        # this is just setting the x to where it "should" be given the number
        # of cycles
        # pos[0] += (self.trajectory.qpos[-1, 0] - self.trajectory.qpos[0, 0]) * self.counter
        
        # ^ should only matter for COM error calculation,
        # gets dropped out of state variable for input reasons

        ###### Setting variable speed  #########
        pos[0] *= self.speed
        pos[0] += (self.trajectory.qpos[-1, 0]- self.trajectory.qpos[0, 0])* self.counter * self.speed
        vel[0] *= self.speed
        ######                          ########

        # setting lateral distance target to 0?
        # regardless of reference trajectory?
        # pos[1] = 0
        # Zero out hip roll
        # pos[7] = 0
        # pos[21] = 0
        # pos[8] = 0
        # pos[22] = 0
        # pos[0:2] = np.zeros(2)
        # pos[2] = 1.3

        return pos, vel

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
        
        ext_state = np.concatenate((clock, [self.speed]))
        # Update orientation
        new_orient = self.cassie_state.pelvis.orientation[:]
        new_translationalVelocity = self.cassie_state.pelvis.translationalVelocity[:]
        # new_translationalVelocity[0:2] += self.com_vel_offset
        if self.time >= self.orient_time:
            quaternion = euler2quat(z=self.orient_add, y=0, x=0)
            iquaternion = inverse_quaternion(quaternion)
            new_orient = quaternion_product(iquaternion, self.cassie_state.pelvis.orientation[:])
            if new_orient[0] < 0:
                new_orient = -new_orient
            new_translationalVelocity = rotate_by_quaternion(self.cassie_state.pelvis.translationalVelocity[:], iquaternion)
        motor_pos = self.cassie_state.motor.position[:]
        joint_pos = self.cassie_state.joint.position[:]
        if self.joint_rand:
            motor_pos += self.joint_offsets[0:10]
            joint_pos += self.joint_offsets[10:16]

        # Use state estimator
        robot_state = np.concatenate([
            [self.cassie_state.pelvis.position[2] - self.cassie_state.terrain.height], # pelvis height
            new_orient,                                 # pelvis orientation
            motor_pos,                                     # actuated joint positions

            new_translationalVelocity,                       # pelvis translational velocity
            self.cassie_state.pelvis.rotationalVelocity[:],                          # pelvis rotational velocity 
            self.cassie_state.motor.velocity[:],                                     # actuated joint velocities

            self.cassie_state.pelvis.translationalAcceleration[:],                   # pelvis translational acceleration
            
            joint_pos,                                     # unactuated joint positions
            self.cassie_state.joint.velocity[:]                                      # unactuated joint velocities
        ])

        if self.state_est:
            return np.concatenate([robot_state,  
                               ext_state])
        else:
            return np.concatenate([qpos[pos_index], 
                               qvel[vel_index], 
                               ext_state])

    def render(self):
        if self.vis is None:
            self.vis = CassieVis(self.sim, "./cassie/cassiemujoco/cassie.xml")

        return self.vis.draw(self.sim)
