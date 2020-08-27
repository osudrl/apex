from .cassiemujoco import pd_in_t, state_out_t, CassieSim, CassieVis

from .trajectory import CassieTrajectory

from math import floor

import numpy as np 
import os
import random

import pickle

# Creating the Standing Environment
class CassieStandingEnv:

    def __init__(self, traj="stepping", simrate=60, state_est=True):

        # Using CassieSim
        self.sim = CassieSim('./cassie/cassiemujoco/cassie.xml')
        self.vis = None

        self.state_est = state_est
        
        # Observation and Action Spaces
        self.observation_space = np.zeros(40)
        if self.state_est:
            self.observation_space = np.zeros(46)       # Size for use with state est
        
        self.action_space = np.zeros(10)

        # Initial Standing States
        dirname = os.path.dirname(__file__)
        if traj == "walking":
            traj_path = os.path.join(dirname, "trajectory", "stepdata.bin")

        elif traj == "stepping":
            traj_path = os.path.join(dirname, "trajectory", "more-poses-trial.bin")
        
        self.trajectory = CassieTrajectory(traj_path)
        
        self.init_qpos = np.copy(self.sim.qpos())
        self.init_qvel = np.copy(self.sim.qvel())

        self.goal_qpos = 0

        self.P = np.array([100,  100,  88,  96,  50]) 
        self.D = np.array([10.0, 10.0, 8.0, 9.6, 5.0])

        self.u = pd_in_t()

        self.cassie_state = state_out_t()

        self.simrate = simrate

        self.phase = 0
        self.phase_add = 1
        self.phaselen = floor(len(self.trajectory) / self.simrate) - 1
        self.speed = 0
        self.counter = 0
        self.time = 0

        # See include/cassiemujoco.h for meaning of these indices
        self.pos_idx = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        self.vel_idx = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]

    @property
    def dt(self):
        return 1 / 2000 * self.simrate

    def close(self):
        if self.vis is not None:
            del self.vis
            self.vis = None
    
    def step_simulation(self, action):
        # Create Target Action
        offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])
        target = action + offset

        self.u = pd_in_t()

        # Forces?
        # self.sim.apply_force([np.random.uniform(-30, 30), np.random.uniform(-30, 30), 0, 0, 0])

        # Apply Action
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

    def step(self, action):
        for _ in range(self.simrate):
            self.step_simulation(action)

        # Current State
        state = self.get_full_state()
        
        # Current Reward
        reward = self.compute_reward()

        self.time  += 1
        self.phase += 1

        if self.phase > self.phaselen:
            self.phase = 0
            self.counter += 1

        # Early termination
        height = self.sim.qpos()[2]
        done = not(height > 0.4 and height < 3.0)
        
        return state, reward, done, {}

    def reset(self):
        self.phase = random.randint(0, self.phaselen)
        qpos0, qvel0 = self.get_ref_state(self.phase)

        self.sim.set_qpos(np.ndarray.flatten(qpos0))
        self.sim.set_qvel(np.ndarray.flatten(qvel0))

        self.goal_qpos = np.ndarray.flatten(self.init_qpos)
        self.goal_qvel = np.ndarray.flatten(self.init_qvel)

        return self.get_full_state()


    def compute_reward(self):
        qpos = np.copy(self.sim.qpos())
        qvel = np.copy(self.sim.qvel())
        left_foot_pos = self.cassie_state.leftFoot.position[:]
        right_foot_pos = self.cassie_state.rightFoot.position[:]
        foot_pos = np.concatenate([left_foot_pos, right_foot_pos])

        # # Pelvis Height
        # height_diff = np.linalg.norm(qpos[2] - self.goal_qpos[2])
        # height_diff = np.exp(-height_diff)

        # # Pelvis Velocity
        # vel_diff = np.linalg.norm(qvel[0:3])
        # vel_diff = np.exp(-vel_diff)

        # # Quaternion Orientation
        # orient_diff = (np.abs(np.arccos(2 * self.goal_qpos[3] ** 2 * qpos[3] ** 2 - 1))) ** 2
        # orient_diff = np.exp(-orient_diff)

        # # Loss and Reward
        # reward = 0.2 * height_diff + 0.6 * vel_diff + 0.2 * orient_diff

        # Upper Body Pose Modulation
        left_roll = np.exp(-qpos[6]**2)
        left_pitch = np.exp(-qpos[8]**2)
        right_roll = np.exp(-qpos[13]**2)
        right_pitch = np.exp(-qpos[15]**2)
        r_pose = 0.25 * left_roll + 0.25 * left_pitch + 0.25 * right_roll + 0.25 * right_pitch

        # COM Position Modulation
        capture_point_pos = np.sqrt(0.5 * (np.abs(foot_pos[0]) + np.abs(foot_pos[3]))**2 + 0.5 * (np.abs(foot_pos[1]) + np.abs(foot_pos[4]))**2)

        xy_com_pos = np.exp(-(capture_point_pos)**2)
        z_com_pos = np.exp(-(qpos[1] - 0.9)**2)
        r_com_pos = 0.5 * xy_com_pos + 0.5 * z_com_pos

        # COM Velocity Modulation
        capture_point_vel = capture_point_pos * np.sqrt(9.8/np.abs(qpos[1]))

        xy_com_vel = np.exp(-((capture_point_vel - np.sqrt(qvel[0]**2 + qvel[1]**2))**2))
        z_com_vel = np.exp(-(qvel[2]**2))

        if np.linalg.norm(self.cassie_state.leftFoot.heelForce) < 5 or np.linalg.norm(self.cassie_state.leftFoot.toeForce) < 5 or np.linalg.norm(self.cassie_state.rightFoot.heelForce) < 5 or np.linalg.norm(self.cassie_state.rightFoot.heelForce) < 5:
            r_com_vel = z_com_vel
        else:
            r_com_vel = 0.5 * xy_com_vel + 0.5 * z_com_vel

        # Total Reward
        reward = 0.33 * r_pose + 0.33 * r_com_pos + 0.34 * r_com_vel

        # Ground Contact
        if np.linalg.norm(self.cassie_state.leftFoot.heelForce) < 5 and np.linalg.norm(self.cassie_state.leftFoot.toeForce) < 5 and np.linalg.norm(self.cassie_state.rightFoot.heelForce) < 5 and np.linalg.norm(self.cassie_state.rightFoot.heelForce) < 5:
            reward = reward - 0.5 

        return reward

    def get_ref_state(self, phase=None):
        if phase is None:
            phase = self.phase

        if phase > self.phaselen:
            phase = 0

        pos = np.copy(self.trajectory.qpos[phase * self.simrate])
        pos[1] = 0

        vel = np.copy(self.trajectory.qvel[phase * self.simrate])

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
            return robot_state  
        else:
            return np.concatenate([qpos[pos_index], 
                               qvel[vel_index]])
    def render(self):
        if self.vis is None:
            self.vis = CassieVis(self.sim, "./cassie/cassiemujoco/cassie.xml")

        self.vis.draw(self.sim)