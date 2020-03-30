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
        self.length = self.qpos.shape[0]
        self.rfoot = np.copy(trajectory["rfoot"])
        self.lfoot = np.copy(trajectory["lfoot"])
    
    def __len__(self):
        return len(self.qpos)

class CassieFootTrajectory:
    def __init__(self, filepath):
        with open(filepath, "rb") as f:
            trajectory = pickle.load(f)

        # com_xyz = np.copy(trajectory["qpos"][:, 0:3])
        # rfoot_relative = np.copy(trajectory["rfoot"])
        # lfoot_relative = np.copy(trajectory["lfoot"])
        # self.rfoot = com_xyz + rfoot_relative
        # self.lfoot = com_xyz + lfoot_relative
        self.rfoot = trajectory["rfoot"][:]
        self.lfoot = trajectory["lfoot"][:]
        self.rfoot_vel = trajectory["rfoot_vel"][:]
        self.lfoot_vel = trajectory["lfoot_vel"][:]
        # print("rfoot shape: ", self.rfoot.shape)
        # print("rfoot vel shape: ", self.rfoot_vel.shape)
    
    def __len__(self):
        return len(self.rfoot)

class CassieTraj:
    def __init__(self, filepath):
        data = np.load(filepath)

        # states
        # self.time = data[:, 0]
        self.qpos = data#[:, 1:36]
        self.len = data.shape[0]
        # self.qvel = data[:, 36:68]
    
    def __len__(self):
        return self.len

class CassieEnv_speed_sidestep:
    def __init__(self, traj, simrate=60, clock_based=False, state_est=False):
        self.sim = CassieSim("./cassie/cassiemujoco/cassie.xml")
        self.vis = None

        self.clock_based = clock_based
        self.state_est = state_est

        if clock_based:
            self.observation_space = np.zeros(42 + 2)
            if self.state_est:
                self.observation_space = np.zeros(48 + 2)       # Size for use with state est
            self.ext_size = 4   # Size of ext_state input, used when constructing mirror obs vector
        else:
            self.observation_space = np.zeros(80)
            if self.state_est:
                self.observation_space = np.zeros(86)       # Size for use with state est
            self.ext_size = 2   # Size of ext_state input, used when constructing mirror obs vector
        self.action_space      = np.zeros(10)

        dirname = os.path.dirname(__file__)
        if traj == "walking":
            traj_path = os.path.join(dirname, "trajectory", "stepdata.bin")

        elif traj == "stepping":
            # traj_path = os.path.join(dirname, "trajectory", "spline_stepping_traj.pkl")
            traj_path = os.path.join(dirname, "trajectory", "more-poses-trial.bin")
                        
        # self.trajectory = CassieTrajectory(traj_path)
        self.trajectory = CassieTraj(os.path.join(dirname, "trajectory", "iktraj_land0.4_speed1.0_fixedheightfreq_fixedtdvel_fixedfoot.npy"))
        # self.foot_traj = CassieFootTrajectory(os.path.join(dirname, "trajectory", "foottraj_doublestance_time0.4_land0.4_h0.2_vels.pkl"))
        # self.foot_traj = CassieFootTrajectory(os.path.join(dirname, "trajectory", "foottraj_doublestance_time0.4_land0.2_vels.pkl"))
        self.foot_traj = CassieFootTrajectory(os.path.join(dirname, "trajectory", "foottraj_land0.4_speed1.0_fixedheightfreq_fixedtdvel.pkl"))

        self.P = np.array([100,  100,  88,  96,  50]) 
        self.D = np.array([10.0, 10.0, 8.0, 9.6, 5.0])

        self.u = pd_in_t()

        self.base_mirror_obs = [0.1, 1, 2, 3, 4, -10, -11, 12, 13, 14, -5, -6, 7, 8, 9, 15,
                            16, 17, 18, 19, 20, -26, -27, 28, 29, 30, -21, -22, 23, 24,
                            25, 31, 32, 33, 37, 38, 39, 34, 35, 36, 43, 44, 45, 40, 41, 42]

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

        self.speed = 0
        self.side_speed = 0     # Positive is left direction, negative is right direction
        self.max_force = 0
        # maybe make ref traj only send relevant idxs?
        ref_pos, ref_vel = self.get_ref_state(self.phase)
        self.prev_action = None
        self.prev_torque = None
        self.curr_action = None
        self.lfoot_vel = 0
        self.rfoot_vel = 0
        self.l_foot_diff = 0
        self.r_foot_diff = 0
        self.l_footvel_diff = 0
        self.r_footvel_diff = 0
        self.joint_error       = 0
        self.com_error         = 0
        self.orientation_error = 0
        self.spring_error      = 0
        self.torque_cost = 0
        self.smooth_cost = 0
        self.lf_heightvel = 0
        self.rf_heightvel = 0
        self.ltdvel_cost = 0
        self.rtdvel_cost = 0
        self.lfoot_orient_cost = 0
        self.rfoot_orient_cost = 0

        self.phase_add = 1
        if self.state_est:
            self.clock_inds = [46, 47]
        else:
            self.clock_inds = [40, 41] 

    def step_simulation(self, action):
        
        offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])
        real_action = action + offset
        #target = action + offset

        #h = 0.005
        #Tf = 1.0 / 300.0
        #alpha = h / (Tf + h)
        
        #real_action = (1-alpha)*self.prev_action + alpha*target
        #self.prev_action = real_action      # Save previous action

        ######## Remove foot offset ########
        # real_action[4] -= -1.5968
        # real_action[9] -= -1.5968
                
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
        # if (np.abs(self.lfoot_vel) < 0.05):
        #     print("l foot height: ", foot_pos[2])
        # if (np.abs(self.rfoot_vel) < 0.05):
        #     print("r foot height: ", foot_pos[5])
        # print("foot vel: ", self.lfoot_vel, self.rfoot_vel)
        # foot_forces = self.sim.get_foot_forces()
        # self.max_force = max(foot_forces[0], foot_forces[1])

    def step(self, action):
        foot_pos = np.zeros(6)
        self.l_foot_diff = 0
        self.r_foot_diff = 0
        self.l_footvel_diff = 0
        self.r_footvel_diff = 0
        self.joint_error       = 0
        self.com_error         = 0
        self.orientation_error = 0
        self.spring_error      = 0
        self.torque_cost = 0
        self.smooth_cost = 0
        self.lf_heightvel = 0
        self.rf_heightvel = 0
        self.ltdvel_cost = 0
        self.rtdvel_cost = 0
        self.lfoot_orient_cost = 0
        self.rfoot_orient_cost = 0

        weight = [0.15, 0.15, 0.1, 0.05, 0.05, 0.15, 0.15, 0.1, 0.05, 0.05]
        neutral_foot_orient = np.array([-0.24790886454547323, -0.24679713195445646, -0.6609396704367185, 0.663921021343526])
        ######## Linear Interpolate ########
        # num_steps = int(self.simrate * 3/4)  # Number of steps to interpolate over. Should be between 0 and self.simrate
        # alpha = 1 / num_steps
        # if self.prev_action is not None:
        #     for _ in range(self.simrate):            
        #         self.step_simulation((1-alpha)*self.prev_action + alpha*action)
        #         if alpha < 1:
        #             alpha += 1 / num_steps
        #         else:
        #             alpha = 1
        # else:
        ######## Regular action (no interpolate) ######## 
        for i in range(self.simrate):
            self.step_simulation(action)
            qpos = np.copy(self.sim.qpos())
            ref_lpos, ref_rpos, ref_lvel, ref_rvel = self.get_ref_foot(self.phase, i+1)
            # Calculate foot pos and vel diff
            self.sim.foot_pos(foot_pos)
            # curr_l_yfoot = np.abs(foot_pos[2] - self.foot_traj.lfoot[int(self.phase*self.simrate) + i+1, 2])
            # curr_r_yfoot = np.abs(foot_pos[5] - self.foot_traj.rfoot[int(self.phase*self.simrate) + i+1, 2])
            # curr_l_yfoot = np.linalg.norm(foot_pos[[0, 2]] - ref_lpos[[0,2]])
            # curr_r_yfoot = np.linalg.norm(foot_pos[[3, 5]] - ref_rpos[[0,2]])
            curr_l_yfoot = np.linalg.norm(foot_pos[2] - ref_lpos[2])
            curr_r_yfoot = np.linalg.norm(foot_pos[5] - ref_rpos[2])
            # if curr_l_yfoot > self.l_foot_diff:
            #     self.l_foot_diff = curr_l_yfoot
            # if curr_r_yfoot > self.r_foot_diff:
            #     self.r_foot_diff = curr_r_yfoot
            self.l_foot_diff += curr_l_yfoot# - self.l_foot_diff) / (i + 1)
            self.r_foot_diff += curr_r_yfoot# - self.r_foot_diff) / (i + 1)
            # curr_l_yfoot_vel = np.abs(self.lfoot_vel - self.foot_traj.lfoot_vel[int(self.phase*self.simrate) + i+1, 2])
            # curr_r_yfoot_vel = np.abs(self.rfoot_vel - self.foot_traj.rfoot_vel[int(self.phase*self.simrate) + i+1, 2])
            # curr_l_yfoot_vel = np.linalg.norm(self.lfoot_vel[[0, 2]] - ref_lvel[[0, 2]])
            # curr_r_yfoot_vel = np.linalg.norm(self.rfoot_vel[[0, 2]] - ref_rvel[[0, 2]])
            curr_l_yfoot_vel = np.linalg.norm(self.lfoot_vel[2] - ref_lvel[2])
            curr_r_yfoot_vel = np.linalg.norm(self.rfoot_vel[2] - ref_rvel[2])
            # if curr_l_yfoot_vel > self.l_footvel_diff:
            #     self.l_footvel_diff = curr_l_yfoot_vel
            # if curr_r_yfoot_vel > self.r_footvel_diff:
            #     self.r_footvel_diff = curr_r_yfoot_vel
            self.l_footvel_diff += curr_l_yfoot_vel# - self.l_footvel_diff) / (i + 1)
            self.r_footvel_diff += curr_r_yfoot_vel# - self.r_footvel_diff) / (i + 1)
            
            ##### Ref traj errors #####
            # ref_pos = np.copy(self.trajectory.qpos[int(self.phase*self.simrate)+i+1])
            # ref_pos[0] = self.speed * self.counter + (self.speed / self.phaselen)*self.phase
            # ref_pos[1] = self.side_speed * self.counter + (self.side_speed / self.phaselen)*self.phase
            # #####                 #####
            # # each joint pos
            # for i, j in enumerate(self.pos_idx):
            #     self.joint_error += 30 * weight[i] * (ref_pos[j] - qpos[j]) ** 2
            # # center of mass: x, y, z
            # for j in [0, 1, 2]:
            #     # NOTE: in Xie et al y target is 0
            #     self.com_error += (ref_pos[j] - qpos[j]) ** 2
            # # COM orientation: qw, qx, qy, qz
            # for j in [3, 4, 5, 6]:
            #     # NOTE: in Xie et al orientation target is 0
            #     self.orientation_error += (ref_pos[j] - qpos[j]) ** 2
            # # left and right shin springs
            # for i in [15, 29]:
            #     # NOTE: in Xie et al spring target is 0
            #     self.spring_error += 1000 * (ref_pos[i] - qpos[i]) ** 2    

            ######### Torque costs ########
            curr_torques = np.array(self.cassie_state.motor.torque[:])
            self.torque_cost += 0.0001*np.linalg.norm(np.square(curr_torques))
            if self.prev_torque is not None:
                self.smooth_cost += 0.0001*np.linalg.norm(np.square(curr_torques - self.prev_torque))
            else:
                self.smooth_cost += 0
            self.prev_torque = curr_torques 
            ######## Foot height vel cost ########
            self.lf_heightvel += (0.1 - foot_pos[2])**2 * np.linalg.norm(self.lfoot_vel[0:2]) * 20
            self.rf_heightvel += (0.1 - foot_pos[5])**2 * np.linalg.norm(self.rfoot_vel[0:2]) * 20
            ######## Tdvel cost ########
            # When foot is close to ground, i.e. (0.2-foot_height) is large, then want foot z velocity to be low
            # When foot is away from ground, don't care how high foot vel is, so weight foot z vel by how close
            # foot is to target height
            self.ltdvel_cost += 0.9 * min(0, self.lfoot_vel[2])**2#(0.2 - foot_pos[2])**2 * -min(0, self.lfoot_vel[2]) * 40
            self.rtdvel_cost += 0.9 * min(0, self.rfoot_vel[2])**2#(0.2 - foot_pos[5])**2 * -min(0, self.rfoot_vel[2]) * 40
            # self.ltdvel_cost += 10*(0.3 - foot_pos[2])**2 * min(0, self.lfoot_vel[2])**2
            # self.rtdvel_cost += 10*(0.3 - foot_pos[5])**2 * min(0, self.rfoot_vel[2])**2
            ######## Foot Orientation ########
            self.lfoot_orient_cost += 1 - np.inner(np.array([1, 0, 0, 0]), self.cassie_state.leftFoot.orientation[:]) ** 2
            self.rfoot_orient_cost += 1 - np.inner(np.array([1, 0, 0, 0]), self.cassie_state.rightFoot.orientation[:]) ** 2
            # self.lfoot_orient_cost += 100*(1 - np.inner(neutral_foot_orient, self.sim.xquat("left-foot")) ** 2)
            # self.rfoot_orient_cost += 100*(1 - np.inner(neutral_foot_orient, self.sim.xquat("right-foot")) ** 2)

            
        self.joint_error        /= self.simrate 
        self.com_error          /= self.simrate 
        self.orientation_error  /= self.simrate 
        self.spring_error       /= self.simrate
        self.l_foot_diff        /= self.simrate
        self.r_foot_diff        /= self.simrate
        self.l_footvel_diff     /= self.simrate
        self.r_footvel_diff     /= self.simrate
        self.torque_cost        /= self.simrate
        self.smooth_cost        /= self.simrate
        self.lf_heightvel       /= self.simrate
        self.rf_heightvel       /= self.simrate
        self.ltdvel_cost        /= self.simrate
        self.rtdvel_cost        /= self.simrate
        self.lfoot_orient_cost  /= self.simrate
        self.rfoot_orient_cost  /= self.simrate

        height = self.sim.qpos()[2]

        self.time  += 1
        self.phase += self.phase_add
        self.curr_action = action

        if self.phase > self.phaselen:
            self.phase = 0
            self.counter += 1

        # Early termination
        done = not(height > 0.4 and height < 3.0)

        reward = self.compute_reward()
        self.prev_action = action

        # TODO: make 0.3 a variable/more transparent
        if reward < 0.3:
            done = True

        return self.get_full_state(), reward, done, {}

    def reset(self):
        self.phase = random.randint(0, self.phaselen)
        self.time = 0
        self.counter = 0
        self.max_force = 0

        orientation = random.randint(-10, 10) * np.pi / 25
        quaternion = euler2quat(z=orientation, y=0, x=0)
        qpos, qvel = self.get_ref_state(self.phase)
        qpos[3:7] = quaternion
        # qpos[2] -= .1

        self.sim.set_qpos(qpos)
        self.sim.set_qvel(qvel)

        # Need to reset u? Or better way to reset cassie_state than taking step
        self.cassie_state = self.sim.step_pd(self.u)

        
        self.speed = (random.randint(-5, 30)) / 10
        self.side_speed = 0.6*random.random() - 0.3
        if self.speed > 1.5:
            self.phase_add = 1.3 + 0.7*random.random()
        else:
            self.phase_add = 1 + random.random()
        ref_pos, ref_vel = self.get_ref_state(self.phase)
        self.prev_action = None
        self.prev_torque = None
        self.lfoot_vel = 0
        self.rfoot_vel = 0
        self.l_foot_diff = 0
        self.r_foot_diff = 0
        self.l_footvel_diff = 0
        self.r_footvel_diff = 0
        self.torque_cost = 0
        self.smooth_cost = 0
        self.lf_heightvel = 0
        self.rf_heightvel = 0
        self.ltdvel_cost = 0
        self.rtdvel_cost = 0
        self.lfoot_orient_cost = 0
        self.rfoot_orient_cost = 0

        return self.get_full_state()

    # used for plotting against the reference trajectory
    def reset_for_test(self):
        self.phase = 0
        self.time = 0
        self.counter = 0
        self.speed = 0.5
        self.side_speed = 0
        self.max_force = 0

        qpos, qvel = self.get_ref_state(self.phase)

        self.sim.set_qpos(qpos)
        self.sim.set_qvel(qvel)

        # Need to reset u? Or better way to reset cassie_state than taking step
        self.cassie_state = self.sim.step_pd(self.u)
        self.phase_add = 1
        ref_pos, ref_vel = self.get_ref_state(self.phase)
        self.prev_action = None
        self.prev_torque = None
        self.prev_foot = None
        self.lfoot_vel = 0
        self.rfoot_vel = 0
        self.l_foot_diff = 0
        self.r_foot_diff = 0
        self.l_footvel_diff = 0
        self.r_footvel_diff = 0
        self.torque_cost = 0
        self.smooth_cost = 0
        self.lf_heightvel = 0
        self.rf_heightvel = 0
        self.ltdvel_cost = 0
        self.rtdvel_cost = 0
        self.lfoot_orient_cost = 0
        self.rfoot_orient_cost = 0

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

    def compute_reward(self):

        # reward = side_speedmatch_reward(self)
        # reward = side_speedmatch_foottraj_reward(self)
        # reward = side_speedmatch_heightvel_reward(self)
        # reward = side_speedmatch_heuristic_reward(self)
        # reward = side_speedmatch_torquesmooth_reward(self)
        reward = side_speedmatch_footorient_footheightvel_actpenalty_reward(self)

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

        pos = np.copy(self.trajectory.qpos[int(phase * self.simrate)])

        # this is just setting the x to where it "should" be given the number
        # of cycles
        # pos[0] += (self.trajectory.qpos[-1, 0] - self.trajectory.qpos[0, 0]) * self.counter
        
        # ^ should only matter for COM error calculation,
        # gets dropped out of state variable for input reasons

        ###### Setting variable speed  #########
        pos[0] *= self.speed
        pos[0] += (self.trajectory.qpos[-1, 0]- self.trajectory.qpos[0, 0])* self.counter * self.speed
        # pos[0] = self.speed * self.counter + (self.speed / self.phaselen)*phase
        ######                          ########
        ###### Setting variable side speed  #########
        # pos[1] *= self.side_speed
        # pos[1] += (self.trajectory.qpos[-1, 1]- self.trajectory.qpos[0, 1])* self.counter * self.side_speed
        pos[1] = self.side_speed * self.counter + (self.side_speed / self.phaselen)*phase
        ######                          ########

        # setting lateral distance target to 0?
        # regardless of reference trajectory?
        # pos[1] = 0

        # vel = np.copy(self.trajectory.qvel[int(phase * self.simrate)])
        vel = np.zeros(32)
        vel[0] *= self.speed
        # vel[1] *= self.side_speed
        # vel[0] = self.speed
        vel[1] = self.side_speed

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
        
        ext_state = np.concatenate((clock, [self.speed, self.side_speed]))

        # Use state estimator
        robot_state = np.concatenate([
            [self.cassie_state.pelvis.position[2] - self.cassie_state.terrain.height], # pelvis height
            self.cassie_state.pelvis.orientation[:],                                 # pelvis orientation
            self.cassie_state.motor.position[:],                                     # actuated joint positions

            self.cassie_state.pelvis.translationalVelocity[:],                       # pelvis translational velocity
            self.cassie_state.pelvis.rotationalVelocity[:],                          # pelvis rotational velocity 
            self.cassie_state.motor.velocity[:],                                     # actuated joint velocities

            self.cassie_state.pelvis.translationalAcceleration[:],                   # pelvis translational acceleration
            
            self.cassie_state.joint.position[:],                                     # unactuated joint positions
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
