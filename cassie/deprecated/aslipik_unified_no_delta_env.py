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
        traj_path = os.path.join(dirname, "trajectory", "aslipTrajsTaskSpace", "walkCycle_{}.pkl".format(speed))
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
        self.rpos = np.copy(trajectory["rpos"])
        self.rvel = np.copy(trajectory["rvel"])
        self.lpos = np.copy(trajectory["lpos"])
        self.lvel = np.copy(trajectory["lvel"])
        self.cpos = np.copy(trajectory["cpos"])
        self.cvel = np.copy(trajectory["cvel"])

# simrate used to be 60
class UnifiedCassieIKEnvNoDelta:
    def __init__(self, traj="stepping", simrate=60, clock_based=True, state_est=True, training=True, debug=False):
        self.sim = CassieSim("./cassiemujoco/cassie.xml")
        self.vis = None

        # robot state estimation included here
        self.observation_space = np.zeros(46 + 18)

        # motor PD targets
        self.action_space      = np.zeros(10)

        self.speeds = np.array([x / 10 for x in range(0, 21)])
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
        self.reward_pos_idx = [3,4,5,6,7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        self.vel_idx = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]

        # for enforcing the correct foot orientation
        self.global_initial_foot_orient = np.array([-0.24135469773826795, -0.24244324494623198, -0.6659363823866352, 0.6629463911006771])
        self.avg_lfoot_quat = np.zeros(4)
        self.avg_rfoot_quat = np.zeros(4)

        # offset for no delta policy
        self.offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])

        ## THIS IS JUST AN IDEA, instead of doing this we are having one ref trajectory per speed
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
        self.debug = debug

    def step_simulation(self, action):

        # # maybe make ref traj only send relevant idxs?
        # if(self.phase == self.phaselen - 1):
        #     ref_pos, ref_vel = self.get_ref_state(0)
        # else:
        #     ref_pos, ref_vel = self.get_ref_state(self.phase + 1)

        # target = action + ref_pos[self.pos_idx]

        # # h = 0.0005
        # # Tf = 1.0 / 300.0
        # # alpha = h / (Tf + h)
        # # real_action = (1-alpha)*self.prev_action + alpha*target

        # real_action = target

        # # diff = real_action - self.prev_action
        # # max_diff = np.ones(10)*0.1
        # # for i in range(10):
        # #     if diff[i] < -max_diff[i]:
        # #         target[i] = self.prev_action[i] - max_diff[i]
        # #     elif diff[i] > max_diff[i]:
        # #         target[i] = self.prev_action[i] + max_diff[i]

        # self.prev_action = real_action

        real_action = action + self.offset

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

    def step(self, action):

        for i in range(self.simrate):
            self.step_simulation(action)
            # calculate running average of foot quaternion
            self.avg_lfoot_quat += self.sim.xquat("left-foot")
            self.avg_rfoot_quat += self.sim.xquat("right-foot")
        self.avg_lfoot_quat /= self.simrate
        self.avg_rfoot_quat /= self.simrate

        height = self.sim.qpos()[2]

        self.time  += 1
        self.phase += 1

        # if self.phase > self.phaselen:
        if self.phase >= self.phaselen:
            self.phase = 0
            self.counter += 1

        # Early termination
        done = not(height > 0.4 and height < 3.0)

        reward = self.compute_reward(action)

        # reset avg foot quaternion
        self.avg_lfoot_quat = np.zeros(4)
        self.avg_rfoot_quat = np.zeros(4)

        # update previous action
        self.prev_action = action

        # TODO: make 0.3 a variable/more transparent
        if reward < 0.3:
            done = True

        return self.get_full_state(), reward, done, {}

    def reset(self):
        random_speed_idx = random.randint(0, self.num_speeds-1)
        self.speed = self.speeds[random_speed_idx]
        # print("current speed: {}".format(self.speed))
        self.trajectory = self.trajectories[random_speed_idx] # switch the current trajectory
        self.phaselen = self.trajectory.length - 1
        self.phase = random.randint(0, self.phaselen)
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
        self.speed = 0
        self.trajectory = self.trajectories[0] # switch the current trajectory
        self.phaselen = self.trajectory.length - 1
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

    def update_selected_trajectory(self, new_speed):
        self.speed = new_speed
        self.trajectory = self.trajectories[(np.abs(self.speeds - self.speed)).argmin()]
        self.phaselen = self.trajectory.length - 1

    # NOTE: this reward is slightly different from the one in Xie et al
    # see notes for details
    def compute_reward(self, action):
        qpos = np.copy(self.sim.qpos())
        qvel = np.copy(self.sim.qvel())

        ref_pos, ref_vel = self.get_ref_state(self.phase)

        # TODO: should be variable; where do these come from?
        # TODO: see magnitude of state variables to gauge contribution to reward
        # weight = [0.15, 0.15, 0.1, 0.05, 0.05, 0.15, 0.15, 0.1, 0.05, 0.05]

        # weight = [0.05, 0.05, 0.25, 0.25, 0.05, 
        #           0.05, 0.05, 0.25, 0.25, 0.05]

        #weight = [.1] * 10

        footpos_error     = 0
        com_vel_error     = 0
        action_penalty    = 0
        foot_orient_penalty = 0
        straight_diff = 0

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

        # try to match com velocity
        ref_cvel = self.get_ref_com_vel(self.phase + 1)

        # center of mass vel: x, y, z
        cvel = self.cassie_state.pelvis.translationalVelocity
        for j in [0, 1, 2]:
            com_vel_error += np.linalg.norm(cvel[j] - ref_cvel[j])

        # # each joint pos, skipping feet
        # for i, j in enumerate(self.reward_pos_idx):
        #     target = ref_pos[j]
        #     actual = qpos[j]

        #     if j == 20 or j == 34:
        #         joint_error += 0
        #     else:
        #         joint_error += (target - actual) ** 2

        # action penalty
        action_penalty = np.linalg.norm(action - self.prev_action)

        # foot orientation penalty
        foot_orient_penalty = np.linalg.norm(self.avg_rfoot_quat - self.global_initial_foot_orient) + np.linalg.norm(self.avg_lfoot_quat - self.global_initial_foot_orient)

        # straight difference penalty
        straight_diff = np.abs(qpos[1])
        if straight_diff < 0.05:
            straight_diff = 0

        reward = 0.3 * np.exp(-footpos_error) +    \
                 0.3 * np.exp(-com_vel_error) +    \
                 0.1 * np.exp(-action_penalty) +     \
                 0.2 * np.exp(-foot_orient_penalty) + \
                 0.1 * np.exp(-straight_diff)

        if self.debug:
            print("reward: {10}\nfoot:\t{0:.2f}, % = {1:.2f}\ncom_vel:\t{2:.2f}, % = {3:.2f}\naction_penalty:\t{4:.2f}, % = {5:.2f}\nfoot_orient_penalty:\t{6:.2f}, % = {7:.2f}\nstraight_diff:\t{8:.2f}, % = {9:.2f}\n\n".format(
            0.3 * np.exp(-footpos_error),          0.3 * np.exp(-footpos_error) / reward * 100,
            0.3 * np.exp(-com_vel_error),          0.3 * np.exp(-com_vel_error) / reward * 100,
            0.1 * np.exp(-action_penalty),         0.1 * np.exp(-action_penalty) / reward * 100,
            0.2 * np.exp(-foot_orient_penalty),    0.2 * np.exp(-foot_orient_penalty) / reward * 100,
            0.1  * np.exp(-straight_diff),         0.1  * np.exp(-straight_diff) / reward * 100,
            reward
            )
            )
            print("actual speed: {}\tdesired_speed: {}".format(qvel[0], self.speed))
        return reward

    # get the corresponding state from the reference trajectory for the current phase
    def get_ref_state(self, phase=None):

        # print("phase: {}\t (phaselen = {})".format(phase, self.phaselen))

        if phase is None:
            phase = self.phase

        if phase > self.phaselen:
            phase = 0

        # print("looped phase: {}".format(phase))
        # print()

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

        rpos = np.copy(self.trajectory.rpos[phase])
        lpos = np.copy(self.trajectory.lpos[phase])

        return rpos, lpos

    def get_ref_com_vel(self, phase=None):

        if phase is None:
            phase = self.phase

        if phase > self.phaselen:
            phase = 0

        cvel = np.copy(self.trajectory.cvel[phase])

        return cvel

    def get_ref_ext_state(self, phase=None):

        if phase is None:
            phase = self.phase

        if phase > self.phaselen:
            phase = 0

        rpos = np.copy(self.trajectory.rpos[phase])
        rvel = np.copy(self.trajectory.rvel[phase])
        lpos = np.copy(self.trajectory.lpos[phase])
        lvel = np.copy(self.trajectory.lvel[phase])
        cpos = np.copy(self.trajectory.cpos[phase])
        cvel = np.copy(self.trajectory.cvel[phase])

        return rpos, rvel, lpos, lvel, cpos, cvel

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

        if(self.phase == 0):
            ext_state = np.concatenate(self.get_ref_ext_state(self.phaselen - 1))
        else:
            ext_state = np.concatenate(self.get_ref_ext_state(self.phase))

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

        return np.concatenate([robot_state, ext_state])

    def reset_for_normalization(self):
        return self.reset()

    def render(self):
        if self.vis is None:
            self.vis = CassieVis(self.sim, "./cassiemujoco/cassie.xml")

        return self.vis.draw(self.sim)
