# Importing Libraries

from .cassiemujoco import pd_in_t, state_out_t, CassieSim, CassieVis

from .trajectory import CassieTrajectory

from math import floor

import numpy as np 
import os
import random

import gym
from gym import spaces
import pickle

# Creating the Standing Environment
class CassieStandingEnv(gym.Env):

    def __init__(self, simrate=60):

        # Using CassieSim
        self.sim = CassieSim('./cassie/cassiemujoco/cassie.xml')
        self.vis = None

        # Observation and Action Spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(46,))
        self.action_space      = spaces.Box(low=-np.inf, high=np.inf, shape=(10,))

        # Initial Standing States
        self.trajectory = CassieTrajectory('./cassie/trajectory/stepdata.bin')
        
        self.init_qpos = np.copy(self.sim.qpos())
        self.init_qvel = np.copy(self.sim.qvel())
        
        self.goal_qpos = 0
        self.goal_qvel = 0

        # PD Controller
        self.P = np.array([100,  100,  88,  96,  50]) 
        self.D = np.array([10.0, 10.0, 8.0, 9.6, 5.0])        

        self.u = pd_in_t()

        self.cassie_state = state_out_t()

        self.simrate = simrate

        self.phase = 0
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
        offset = np.array([0.0045, 0.0, 0.4973, -1.1997, 0.8 * -1.5968, 0.0045, 0.0, 0.4973, -1.1997, 0.8 * -1.5968])
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
        # x = np.random.choice(self.trajectory.sample().shape[0], size = 1, replace = False)
        # qpos0, qvel0 = self.trajectory['qpos'][x], np.zeros(32)

        self.phase = random.randint(0, self.phaselen)
        self.speed = random.uniform(-0.3, 0.3)
        qpos0, qvel0 = self.get_ref_state(self.phase)

        c = 0.01 

        # qpos0 = self.init_qpos + np.random.uniform(low=-c, high=c, size=self.init_qpos.size)
        # qvel0 = self.init_qvel + np.random.uniform(low=-c, high=c, size=self.init_qvel.size)

        self.sim.set_qpos(np.ndarray.flatten(qpos0))
        self.sim.set_qvel(np.ndarray.flatten(qvel0))

        self.goal_qpos = np.ndarray.flatten(self.init_qpos)
        self.goal_qvel = np.ndarray.flatten(self.init_qvel)

        return self.get_full_state()


    def compute_reward(self):
        qpos = np.copy(self.sim.qpos())
        qvel = np.copy(self.sim.qvel())

        # Pelvis Height
        height_diff = np.linalg.norm(qpos[2] - self.goal_qpos[2])
        height_diff = np.exp(-height_diff)

        # Pelvis Velocity
        vel_diff = np.linalg.norm(qvel[0:3])
        vel_diff = np.exp(-vel_diff)

        # Quaternion Orientation
        orient_diff = (np.abs(np.arccos(2 * self.goal_qpos[3] ** 2 * qpos[3] ** 2 - 1))) ** 2
        orient_diff = np.exp(-orient_diff)

        # Loss and Reward
        reward = 0.05 * height_diff + 0.9 * vel_diff + 0.05 * orient_diff

        return reward

    def get_ref_state(self, phase=None):
        if phase is None:
            phase = self.phase

        if phase > self.phaselen:
            phase = 0

        pos = np.copy(self.trajectory.qpos[phase * self.simrate])

        # this is just setting the x to where it "should" be given the number
        # of cycles
        # pos[0] += (self.trajectory.qpos[-1, 0] - self.trajectory.qpos[0, 0]) * self.counter
        
        # ^ should only matter for COM error calculation,
        # gets dropped out of state variable for input reasons

        ###### Setting variable speed  #########
        pos[0] *= self.speed
        pos[0] += (self.trajectory.qpos[-1, 0]- self.trajectory.qpos[0, 0])* self.counter * self.speed
        ######                          ########

        # setting lateral distance target to 0?
        # regardless of reference trajectory?
        pos[1] = 0

        vel = np.copy(self.trajectory.qvel[phase * self.simrate])
        vel[0] *= self.speed

        return pos, vel

    def get_full_state(self):
        # this is everything except pelvis x and qw, achilles rod quaternions, 
        # and heel spring/foot crank/plantar rod angles
        # NOTE: x is forward dist, y is lateral dist, z is height

        # makes sense to always exclude x because it is in global coordinates and
        # irrelevant to phase-based control. Z is inherently invariant to (flat)
        # trajectories despite being global coord. Y is only invariant to straight
        # line trajectories.

        # [ 0] Pelvis y
        # [ 1] Pelvis z
        # [ 2] Pelvis or/ientation qw
        # [ 3] Pelvis orientation qx
        # [ 4] Pelvis orientation qy
        # [ 5] Pelvis orientation qz
        # [ 6] Left hip roll         (Motor [0])
        # [ 7] Left hip yaw          (Motor [1])
        # [ 8] Left hip pitch        (Motor [2])
        # [ 9] Left knee             (Motor [3])
        # [10] Left shin                        (Joint [0])
        # [11] Left tarsus                      (Joint [1])
        # [12] Left foot              (Motor [4], Joint [2])
        # [13] Right hip roll        (Motor [5])
        # [14] Right hip yaw         (Motor [6])
        # [15] Right hip pitch       (Motor [7])
        # [16] Right knee            (Motor [8])
        # [17] Right shin                       (Joint [3])
        # [18] Right tarsus                     (JCassieIKTrajectoryoint [4])
        # [19] Right foot            (Motor [9], JCassieIKTrajectoryoint [5])
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
        # [10] Left shin                        
        # [11] Left tarsus                      
        # [12] Left foot             (Motor [4],
        # [13] Right hip roll        (Motor [5])
        # [14] Right hip yaw         (Motor [6])
        # [15] Right hip pitch       (Motor [7])
        # [16] Right knee            (Motor [8])
        # [17] Right shin                       (Joint [3])
        # [18] Right tarsus                     (Joint [4])
        # [19] Right foot            (Motor [9], Joint [5])
        vel_index = np.array([0,1,2,3,4,5,6,7,8,12,13,14,18,19,20,21,25,26,27,31])

        robot_state = np.concatenate([
            [self.cassie_state.pelvis.position[2] - self.cassie_state.terrain.height], # pelvis height 1
            self.cassie_state.pelvis.orientation[:],                                 # pelvis orientation 4 
            self.cassie_state.motor.position[:],                                     # actuated joint positions 10

            self.cassie_state.pelvis.translationalVelocity[:],                       # pelvis translational velocity 3
            self.cassie_state.pelvis.rotationalVelocity[:],                          # pelvis rotational velocity 3
            self.cassie_state.motor.velocity[:],                                     # actuated joint velocities 10

            self.cassie_state.pelvis.translationalAcceleration[:],                   # pelvis translational acceleration 3
            
            self.cassie_state.joint.position[:],                                     # unactuated joint positions 6
            self.cassie_state.joint.velocity[:]                                      # unactuated joint velocities 6
        ])

        return robot_state
    def render(self):
        if self.vis is None:
            self.vis = CassieVis(self.sim, "./cassie/cassiemujoco/cassie.xml")

        self.vis.draw(self.sim)