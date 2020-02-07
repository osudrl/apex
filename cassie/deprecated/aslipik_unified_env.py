from .cassiemujoco import pd_in_t, state_out_t, CassieSim, CassieVis

from .trajectory import CassieTrajectory

from math import floor

import numpy as np 
import os
import random

import pickle

def getAllTrajectories(speeds):
    trajectories = []

    for i, speed in enumerate(speeds):
        dirname = os.path.dirname(__file__)
        traj_path = os.path.join(dirname, "trajectory", "aslipTrajsSweep", "walkCycle_{}.pkl".format(speed))
        trajectories.append(CassieIKTrajectory(traj_path))

    # print("Got all trajectories")
    return trajectories

class CassieIKTrajectory:
    def __init__(self, filepath):
        with open(filepath, "rb") as f:
            trajectory = pickle.load(f)

        self.qpos = np.copy(trajectory["qpos"])
        self.length = self.qpos.shape[0]
        self.qvel = np.copy(trajectory["qvel"])
        self.rfoot = np.copy(trajectory["rfoot"])
        self.lfoot = np.copy(trajectory["lfoot"])

# simrate used to be 60
class UnifiedCassieIKEnv:
    def __init__(self, traj="stepping", simrate=60, clock_based=True, state_est=True, training=True):
        self.sim = CassieSim("./cassiemujoco/cassie.xml")
        self.vis = None

        self.clock_based = clock_based
        self.state_est = state_est

        if clock_based:
            self.observation_space = np.zeros(42 + 1)
            self.clock_inds = [40, 41]
            if self.state_est:
                self.observation_space = np.zeros(48 + 1)       # Size for use with state est
                self.clock_inds = [46, 47]
        else:
            self.observation_space = np.zeros(80)
            if self.state_est:
                self.observation_space = np.zeros(86)       # Size for use without state est
        self.action_space      = np.zeros(10)

        self.speeds = [x / 10 for x in range(0, 31)]
        self.trajectories = getAllTrajectories(self.speeds)
        self.num_speeds = len(self.trajectories)

        self.P = np.array([100,  100,  88,  96,  50]) 
        self.D = np.array([10.0, 10.0, 8.0, 9.6, 5.0])

        self.u = pd_in_t()

        self.simrate = simrate # simulate X mujoco steps with same pd target
                               # 60 brings simulation from 2000Hz to roughly 30Hz

        self.time    = 0 # number of time steps in current episode
        self.phase   = 0 # portion of the phase the robot is in
        self.counter = 0 # number of phase cycles completed in episode

        # NOTE: each trajectory in trajectories should have the same length
        self.speed = self.speeds[0]
        self.trajectory = self.trajectories[0]

        self.training = training

        # NOTE: a reference trajectory represents ONE phase cycle

        # should be floor(len(traj) / simrate) - 1
        # should be VERY cautious here because wrapping around trajectory
        # badly can cause assymetrical/bad gaits
        # self.phaselen = floor(self.trajectory.length / self.simrate) - 1
        # self.phaselen = self.trajectory.length - 1
        self.phaselen = self.trajectory.length - 1

        # see include/cassiemujoco.h for meaning of these indices
        self.pos_idx = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        self.vel_idx = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]

        # params for changing the trajectory
        # self.speed = 2 # how fast (m/s) do we go
        # self.gait = [1,0,0,0]   # one-hot vector of gaits:
                                # [1, 0, 0, 0] -> walking/running (left single stance, right single stance)
                                # [0, 1, 0, 0] -> hopping (double stance, flight phase)
                                # [0, 0, 1, 0] -> skipping (double stance, right single stance, flight phase, right single stance)
                                # [0, 0, 0, 1] -> galloping (double stance, right single stance, flight, left single stance)


        # maybe make ref traj only send relevant idxs?
        ref_pos, ref_vel = self.get_ref_state(self.phase)
        self.prev_action = ref_pos[self.pos_idx]
        self.phase_add = 1

        # Output of Cassie's state estimation code
        self.cassie_state = state_out_t()

        # for print statements
        self.debug = False

    def step_simulation(self, action):

        # maybe make ref traj only send relevant idxs?
        if(self.phase == self.phaselen - 1):
            ref_pos, ref_vel = self.get_ref_state(0)
        else:
            ref_pos, ref_vel = self.get_ref_state(self.phase + 1)

        target = action + ref_pos[self.pos_idx]

        # h = 0.0005
        # Tf = 1.0 / 300.0
        # alpha = h / (Tf + h)
        # real_action = (1-alpha)*self.prev_action + alpha*target

        real_action = target

        # diff = real_action - self.prev_action
        # max_diff = np.ones(10)*0.1
        # for i in range(10):
        #     if diff[i] < -max_diff[i]:
        #         target[i] = self.prev_action[i] - max_diff[i]
        #     elif diff[i] > max_diff[i]:
        #         target[i] = self.prev_action[i] + max_diff[i]

        self.prev_action = real_action
        # real_action = target

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

    def step(self, action):
        for _ in range(self.simrate):
            self.step_simulation(action)

        height = self.sim.qpos()[2]

        self.time  += 1
        self.phase += 1

        # if self.phase > self.phaselen:
        if self.phase >= self.phaselen:
            self.phase = 0
            self.counter += 1

        # Early termination
        done = not(height > 0.4 and height < 3.0)

        reward = self.compute_reward()

        # TODO: make 0.3 a variable/more transparent
        if reward < 0.5:
            done = True

        return self.get_full_state(), reward, done, {}

    def reset(self):
        random_speed_idx = random.randint(0, self.num_speeds-1)
        self.speed = self.speeds[random_speed_idx]
        # print("current speed: {}".format(self.speed))
        self.trajectory = self.trajectories[random_speed_idx] # switch the current trajectory
        self.phase = random.randint(0, self.phaselen - 1)
        # self.phase = 0
        self.time = 0
        self.counter = 0

        qpos, qvel = self.get_ref_state(self.phase)
        # qpos[2] -= .1

        self.sim.set_qpos(qpos)
        # self.sim.set_qvel(qvel)
        self.sim.set_qvel(np.zeros(qvel.shape))

        # Need to reset u? Or better way to reset cassie_state than taking step
        self.cassie_state = self.sim.step_pd(self.u)

        # maybe make ref traj only send relevant idxs?
        ref_pos, ref_vel = self.get_ref_state(self.phase)
        self.prev_action = ref_pos[self.pos_idx]

        return self.get_full_state()

    # used for plotting against the reference trajectory
    def reset_for_test(self):
        random_speed_idx = random.randint(0, self.num_speeds)
        self.speed = self.speeds[random_speed_idx]
        self.trajectory = self.trajectories[random_speed_idx] # switch the current trajectory
        self.phase = 0
        self.time = 0
        self.counter = 0

        qpos, qvel = self.get_ref_state(self.phase)

        self.sim.set_qpos(qpos)
        self.sim.set_qvel(qvel)

        # maybe make ref traj only send relevant idxs?
        ref_pos, ref_vel = self.get_ref_state(self.phase)
        self.prev_action = ref_pos[self.pos_idx]

        # Need to reset u? Or better way to reset cassie_state than taking step
        self.cassie_state = self.sim.step_pd(self.u)

        return self.get_full_state()

    # NOTE: this reward is slightly different from the one in Xie et al
    # see notes for details
    def compute_reward(self):
        qpos = np.copy(self.sim.qpos())
        qvel = np.copy(self.sim.qvel())

        ref_pos, ref_vel = self.get_ref_state(self.phase)

        # TODO: should be variable; where do these come from?
        # TODO: see magnitude of state variables to gauge contribution to reward
        weight = [0.15, 0.15, 0.1, 0.05, 0.05, 0.15, 0.15, 0.1, 0.05, 0.05]

        # weight = [0.05, 0.05, 0.25, 0.25, 0.05, 
        #           0.05, 0.05, 0.25, 0.25, 0.05]

        #weight = [.1] * 10

        footpos_error     = 0
        joint_error       = 0
        com_error         = 0
        orientation_error = 0
        spring_error      = 0

        # enforce distance between feet and com
        ref_rfoot, ref_lfoot  = self.get_ref_footdist(self.phase + 1)

        # left foot
        lfoot = self.cassie_state.leftFoot.position[:]
        rfoot = self.cassie_state.rightFoot.position[:]
        for j in [0, 1, 2]:
            footpos_error += np.linalg.norm(lfoot[j] - ref_lfoot[j]) +  np.linalg.norm(rfoot[j] - ref_rfoot[j])
        
        if self.debug:
            print("ref_rfoot: {}  rfoot: {}".format(ref_rfoot, rfoot))
            print("ref_lfoot: {}  lfoot: {}".format(ref_lfoot, lfoot))
            print(footpos_error)

        # speed reward component
        speed_diff = np.abs(qvel[0] - self.speed)

        # each joint pos, skipping feet
        for i, j in enumerate(self.pos_idx):
            target = ref_pos[j]
            actual = qpos[j]

            if j == 20 or j == 34:
                joint_error += 0
            else:
                joint_error += (target - actual) ** 2

        # center of mass: x, y, z
        for j in [0, 1, 2]:
            target = ref_pos[j]
            actual = qpos[j]

            # NOTE: in Xie et al y target is 0

            com_error += (target - actual) ** 2
        
        # COM orientation: qx, qy, qz
        for j in [4, 5, 6]:
            target = ref_pos[j] # NOTE: in Xie et al orientation target is 0
            actual = qpos[j]

            orientation_error += (target - actual) ** 2

        # left and right shin springs
        for i in [15, 29]:
            target = ref_pos[i] # NOTE: in Xie et al spring target is 0
            actual = qpos[i]

            spring_error += (target - actual) ** 2      
        
        # reward = 0.1 * np.exp(-footpos_error) +       \
        #          0.5 * np.exp(-joint_error) +       \
        #          0.3 * np.exp(-com_error) +         \
        #          0.1 * np.exp(-orientation_error) + \
        #          0.0 * np.exp(-spring_error)
        reward = 0.1 * np.exp(-footpos_error) +       \
                 0.5 * np.exp(-joint_error) +       \
                 0.2 * np.exp(-com_error) +         \
                 0.1 * np.exp(-orientation_error) + \
                 0.1 * np.exp(-speed_diff)
        #reward = np.exp(-joint_error)

        # print("{}\t{}\t{}\t{}".format(self.speed, self.sim.qvel()[0], 0.2 * np.exp(-com_error), reward))

        # orientation error does not look informative
        # maybe because it's comparing euclidean distance on quaternions
        if self.debug:
            print("reward: {10}\nfoot:\t{0:.2f}, % = {1:.2f}\njoint:\t{2:.2f}, % = {3:.2f}\ncom:\t{4:.2f}, % = {5:.2f}\norient:\t{6:.2f}, % = {7:.2f}\nspeed:\t{8:.2f}, % = {9:.2f}\n\n".format(
            0.1 * np.exp(-footpos_error),     0.1 * np.exp(-footpos_error) / reward * 100,
            0.5 * np.exp(-joint_error),       0.5 * np.exp(-joint_error) / reward * 100,
            0.2 * np.exp(-com_error),         0.2 * np.exp(-com_error) / reward * 100,
            0.1 * np.exp(-orientation_error), 0.1 * np.exp(-orientation_error) / reward * 100,
            0.1 * np.exp(-speed_diff),      0.1 * np.exp(-speed_diff) / reward * 100,
            reward
            )
            )
            print("desired_speed: {}".format(self.speed))
        return reward

    # get the corresponding state from the reference trajectory for the current phase
    def get_ref_state(self, phase=None):

        if phase is None:
            phase = self.phase

        if phase > self.phaselen:
            phase = 0

        # pos = np.copy(self.trajectory.qpos[phase * self.simrate])
        pos = np.copy(self.trajectory.qpos[phase])

        # this is just setting the x to where it "should" be given the number
        # of cycles
        #pos[0] += (self.trajectory.qpos[-1, 0] - self.trajectory.qpos[0, 0]) * self.counter
        pos[0] += (self.trajectory.qpos[-1, 0] - self.trajectory.qpos[0, 0]) * self.counter
        
        # ^ should only matter for COM error calculation,
        # gets dropped out of state variable for input reasons

        # setting lateral distance target to 0?
        # regardless of reference trajectory?
        pos[1] = 0

        # vel = np.copy(self.trajectory.qvel[phase * self.simrate])
        vel = np.copy(self.trajectory.qvel[phase])

        return pos, vel

    # get the corresponding state from the reference trajectory for the current phase
    def get_ref_footdist(self, phase=None):
        if phase is None:
            phase = self.phase

        if phase > self.phaselen:
            phase = 0

        # rfoot = np.copy(self.trajectory.rfoot[phase * self.simrate])
        # lfoot = np.copy(self.trajectory.lfoot[phase * self.simrate])
        rfoot = np.copy(self.trajectory.rfoot[phase])
        lfoot = np.copy(self.trajectory.lfoot[phase])

        return rfoot, lfoot

    def get_full_state(self):
        qpos = np.copy(self.sim.qpos())
        qvel = np.copy(self.sim.qvel()) 

        if(self.phase == 0):
            ref_pos, ref_vel = self.get_ref_state(self.phaselen - 1)
        else:
            ref_pos, ref_vel = self.get_ref_state(self.phase)

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

        if self.clock_based:
            #qpos[self.pos_idx] -= ref_pos[self.pos_idx]
            #qvel[self.vel_idx] -= ref_vel[self.vel_idx]

            clock = [np.sin(2 * np.pi *  self.phase / self.phaselen),
                     np.cos(2 * np.pi *  self.phase / self.phaselen)]

            ext_state = ext_state = np.concatenate((clock, [self.speed]))

        else:
            ext_state = np.concatenate([ref_pos[pos_index], ref_vel[vel_index]])


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

    def reset_for_normalization(self):
        return self.reset()

    def render(self):
        if self.vis is None:
            self.vis = CassieVis(self.sim, "./cassiemujoco/cassie.xml")

        return self.vis.draw(self.sim)
