import numpy as np
import pickle
from cassie.trajectory.aslip_trajectory import get_ref_aslip_ext_state, get_ref_aslip_unaltered_state, get_ref_aslip_global_state


def clock_reward(self, action):

    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())

    # These used for normalizing the foot forces and velocities
    desired_max_foot_frc = 400
    desired_max_foot_vel = 3.0
    orient_targ = np.array([1, 0, 0, 0])

    # state info
    com_vel = qvel[0] # only care about x velocity
    # put a cap on the frc and vel so as to prevent the policy from learning to maximize them during phase.
    normed_left_frc = min(self.l_foot_frc, desired_max_foot_frc) / desired_max_foot_frc
    normed_right_frc = min(self.r_foot_frc, desired_max_foot_frc) / desired_max_foot_frc
    normed_left_vel = min(np.linalg.norm(self.l_foot_vel), desired_max_foot_vel) / desired_max_foot_vel
    normed_right_vel = min(np.linalg.norm(self.r_foot_vel), desired_max_foot_vel) / desired_max_foot_vel

    com_orient_error  = 0
    foot_orient_error  = 0
    com_vel_error     = 0
    straight_diff     = 0
    foot_vel_error    = 0
    foot_frc_error    = 0

    # com orient error
    com_orient_error += 10 * (1 - np.inner(orient_targ, qpos[3:7]) ** 2)

    # foot orient error
    foot_orient_error += 10 * (self.l_foot_orient_cost + self.r_foot_orient_cost)

    # com vel error
    com_vel_error += np.linalg.norm(com_vel - self.speed)
    
    # straight difference penalty
    straight_diff = np.abs(qpos[1])
    if straight_diff < 0.05:
        straight_diff = 0
    # height deadzone is +- 0.2 meters
    height_diff = np.abs(qpos[2] - 1.0)
    if height_diff < 0.2:
        height_diff = 0
    straight_diff += height_diff

    # force/vel clock errors

    # These values represent if we want to allow foot foot forces / vels (1 -> good if forces/vels exist), or (-1 -> bad if forces/vels exist)
    # left_frc_clock = self.left_clock(self.phase)
    # right_frc_clock = self.right_clock(self.phase)
    # left_vel_clock = -self.left_clock(self.phase)
    # right_vel_clock = -self.right_clock(self.phase)
    left_frc_clock = self.left_clock[0](self.phase)
    right_frc_clock = self.right_clock[0](self.phase)
    left_vel_clock = self.left_clock[1](self.phase)
    right_vel_clock = self.right_clock[1](self.phase)

    left_frc_score = np.tanh(left_frc_clock * normed_left_frc)
    left_vel_score = np.tanh(left_vel_clock * normed_left_vel)
    right_frc_score = np.tanh(right_frc_clock * normed_right_frc)
    right_vel_score = np.tanh(right_vel_clock * normed_right_vel)

    foot_frc_score = left_frc_score + right_frc_score
    foot_vel_score = left_vel_score + right_vel_score

    reward = 0.1 * np.exp(-com_orient_error) +    \
                0.1 * np.exp(-foot_orient_error) +    \
                0.2 * np.exp(-com_vel_error) +    \
                0.1 * np.exp(-straight_diff) +      \
                0.25 * foot_frc_score +     \
                0.25 * foot_vel_score

    if self.debug:
        print("l_frc phase : {:.2f}\t l_frc applied : {:.2f}\t l_frc_score: {:.2f}\t t_frc_score: {:.2f}".format(left_frc_clock, normed_left_frc, left_frc_score, foot_frc_score))
        print("l_vel phase : {:.2f}\t l_vel applied : {:.2f}\t l_vel_score: {:.2f}\t t_vel_score: {:.2f}".format(left_vel_clock, normed_left_vel, left_vel_score, foot_vel_score))
        # print("r_frc phase : {:.2f}\t r_frc applied : {:.2f}\t r_penalty: {:.2f}\t t_penalty: {:.2f}".format(right_frc_clock, normed_right_frc, right_frc_penalty, foot_frc_penalty))
        # print("r_vel phase : {:.2f}\t r_vel applied : {:.2f}\t r_penalty: {:.2f}\t t_penalty: {:.2f}".format(right_vel_clock, normed_right_vel, right_vel_penalty, foot_vel_penalty))
        print("reward: {12}\nfoot_orient:\t{0:.2f}, % = {1:.2f}\ncom_vel:\t{2:.2f}, % = {3:.2f}\nfoot_frc_score:\t{4:.2f}, % = {5:.2f}\nfoot_vel_score:\t{6:.2f}, % = {7:.2f}\nstraight_diff:\t{8:.2f}, % = {9:.2f}\ncom_orient:\t{10:.2f}, % = {11:.2f}".format(
        0.1 * np.exp(-foot_orient_error),         0.1 * np.exp(-foot_orient_error) / reward * 100,
        0.2 * np.exp(-com_vel_error),             0.2 * np.exp(-com_vel_error) / reward * 100,
        0.25 * foot_frc_score,                  0.25 * foot_frc_score / reward * 100,
        0.25 * foot_vel_score,                  0.25 * foot_vel_score / reward * 100,
        0.1  * np.exp(-straight_diff),            0.1 * np.exp(-straight_diff) / reward * 100,
        0.1  * np.exp(-com_orient_error),         0.1 * np.exp(-com_orient_error) / reward * 100,
        reward
        )
        )
        print("actual speed: {}\tcommanded speed: {}\n\n".format(np.linalg.norm(qvel[0:3]), self.speed))
    return reward

def load_clock_reward(self, action):

    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())

    # These used for normalizing the foot forces and velocities
    desired_max_foot_frc = 400
    desired_max_foot_vel = 3.0
    orient_targ = np.array([1, 0, 0, 0])

    # state info
    com_vel = qvel[0] # only care about x velocity
    # put a cap on the frc and vel so as to prevent the policy from learning to maximize them during phase.
    normed_left_frc = min(self.l_foot_frc, desired_max_foot_frc) / desired_max_foot_frc
    normed_right_frc = min(self.r_foot_frc, desired_max_foot_frc) / desired_max_foot_frc
    normed_left_vel = min(np.linalg.norm(self.l_foot_vel), desired_max_foot_vel) / desired_max_foot_vel
    normed_right_vel = min(np.linalg.norm(self.r_foot_vel), desired_max_foot_vel) / desired_max_foot_vel

    com_orient_error  = 0
    foot_orient_error  = 0
    com_vel_error     = 0
    straight_diff     = 0
    foot_vel_error    = 0
    foot_frc_error    = 0

    # com orient error
    com_orient_error += 10 * (1 - np.inner(orient_targ, qpos[3:7]) ** 2)

    # foot orient error
    foot_orient_error += 10 * (self.l_foot_orient_cost + self.r_foot_orient_cost)

    # com vel error
    com_vel_error += np.linalg.norm(com_vel - self.speed)
    
    # straight difference penalty
    straight_diff = np.abs(qpos[1])
    if straight_diff < 0.05:
        straight_diff = 0
    # height deadzone is +- 0.2 meters
    height_diff = np.abs(qpos[2] - 1.0)
    if height_diff < 0.2:
        height_diff = 0
    straight_diff += height_diff

    # force/vel clock errors

    # These values represent if we want to allow foot foot forces / vels (1 -> good if forces/vels exist), or (-1 -> bad if forces/vels exist)
    # left_frc_clock = self.left_clock(self.phase)
    # right_frc_clock = self.right_clock(self.phase)
    # left_vel_clock = -self.left_clock(self.phase)
    # right_vel_clock = -self.right_clock(self.phase)
    left_frc_clock = self.left_clock(self.phase)
    right_frc_clock = self.right_clock(self.phase)
    left_vel_clock = self.left_clock(self.phase)
    right_vel_clock = self.right_clock(self.phase)

    left_frc_score = np.tanh(left_frc_clock * normed_left_frc)
    left_vel_score = np.tanh(left_vel_clock * normed_left_vel)
    right_frc_score = np.tanh(right_frc_clock * normed_right_frc)
    right_vel_score = np.tanh(right_vel_clock * normed_right_vel)

    foot_frc_score = left_frc_score + right_frc_score
    foot_vel_score = left_vel_score + right_vel_score

    reward = 0.1 * np.exp(-com_orient_error) +    \
                0.1 * np.exp(-foot_orient_error) +    \
                0.2 * np.exp(-com_vel_error) +    \
                0.1 * np.exp(-straight_diff) +      \
                0.25 * foot_frc_score +     \
                0.25 * foot_vel_score

    if self.debug:
        print("l_frc phase : {:.2f}\t l_frc applied : {:.2f}\t l_frc_score: {:.2f}\t t_frc_score: {:.2f}".format(left_frc_clock, normed_left_frc, left_frc_score, foot_frc_score))
        print("l_vel phase : {:.2f}\t l_vel applied : {:.2f}\t l_vel_score: {:.2f}\t t_vel_score: {:.2f}".format(left_vel_clock, normed_left_vel, left_vel_score, foot_vel_score))
        # print("r_frc phase : {:.2f}\t r_frc applied : {:.2f}\t r_penalty: {:.2f}\t t_penalty: {:.2f}".format(right_frc_clock, normed_right_frc, right_frc_penalty, foot_frc_penalty))
        # print("r_vel phase : {:.2f}\t r_vel applied : {:.2f}\t r_penalty: {:.2f}\t t_penalty: {:.2f}".format(right_vel_clock, normed_right_vel, right_vel_penalty, foot_vel_penalty))
        print("reward: {12}\nfoot_orient:\t{0:.2f}, % = {1:.2f}\ncom_vel:\t{2:.2f}, % = {3:.2f}\nfoot_frc_score:\t{4:.2f}, % = {5:.2f}\nfoot_vel_score:\t{6:.2f}, % = {7:.2f}\nstraight_diff:\t{8:.2f}, % = {9:.2f}\ncom_orient:\t{10:.2f}, % = {11:.2f}".format(
        0.1 * np.exp(-foot_orient_error),         0.1 * np.exp(-foot_orient_error) / reward * 100,
        0.2 * np.exp(-com_vel_error),             0.2 * np.exp(-com_vel_error) / reward * 100,
        0.25 * foot_frc_score,                  0.25 * foot_frc_score / reward * 100,
        0.25 * foot_vel_score,                  0.25 * foot_vel_score / reward * 100,
        0.1  * np.exp(-straight_diff),            0.1 * np.exp(-straight_diff) / reward * 100,
        0.1  * np.exp(-com_orient_error),         0.1 * np.exp(-com_orient_error) / reward * 100,
        reward
        )
        )
        print("actual speed: {}\tcommanded speed: {}\n\n".format(np.linalg.norm(qvel[0:3]), self.speed))
    return reward

def low_speed_clock_reward(self, action):

    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())

    # These used for normalizing the foot forces and velocities
    desired_max_foot_frc = 400
    desired_max_foot_vel = 3.0
    orient_targ = np.array([1, 0, 0, 0])

    # state info
    com_vel = qvel[0] # only care about x velocity
    # put a cap on the frc and vel so as to prevent the policy from learning to maximize them during phase.
    normed_left_frc = min(self.l_foot_frc, desired_max_foot_frc) / desired_max_foot_frc
    normed_right_frc = min(self.r_foot_frc, desired_max_foot_frc) / desired_max_foot_frc
    normed_left_vel = min(np.linalg.norm(self.l_foot_vel), desired_max_foot_vel) / desired_max_foot_vel
    normed_right_vel = min(np.linalg.norm(self.r_foot_vel), desired_max_foot_vel) / desired_max_foot_vel

    com_orient_error  = 0
    foot_orient_error  = 0
    com_vel_error     = 0
    straight_diff     = 0
    foot_vel_error    = 0
    foot_frc_error    = 0

    # com orient error
    com_orient_error += 10 * (1 - np.inner(orient_targ, qpos[3:7]) ** 2)

    # foot orient error
    foot_orient_error += 10 * (self.l_foot_orient_cost + self.r_foot_orient_cost)

    # com vel error
    com_vel_error += np.linalg.norm(com_vel - 0.5)
    
    # straight difference penalty
    straight_diff = np.abs(qpos[1])
    if straight_diff < 0.05:
        straight_diff = 0
    # height deadzone is +- 0.2 meters
    height_diff = np.abs(qpos[2] - 1.0)
    if height_diff < 0.2:
        height_diff = 0
    straight_diff += height_diff

    # force/vel clock errors

    # These values represent if we want to allow foot foot forces / vels (1 -> good if forces/vels exist), or (-1 -> bad if forces/vels exist)
    left_frc_clock = self.left_clock[0](self.phase)
    right_frc_clock = self.right_clock[0](self.phase)
    left_vel_clock = self.left_clock[1](self.phase)
    right_vel_clock = self.right_clock[1](self.phase)
    
    left_frc_penalty = np.tanh(left_frc_clock * normed_left_frc)
    left_vel_penalty = np.tanh(left_vel_clock * normed_left_vel)
    right_frc_penalty = np.tanh(right_frc_clock * normed_right_frc)
    right_vel_penalty = np.tanh(right_vel_clock * normed_right_vel)

    foot_frc_penalty = left_frc_penalty + right_frc_penalty
    foot_vel_penalty = left_vel_penalty + right_vel_penalty

    reward = 0.1 * np.exp(-com_orient_error) +    \
                0.1 * np.exp(-foot_orient_error) +    \
                0.2 * np.exp(-com_vel_error) +    \
                0.1 * np.exp(-straight_diff) +      \
                0.25 * foot_frc_penalty +     \
                0.25 * foot_vel_penalty

    if self.debug:
        print("l_frc phase : {:.2f}\t l_frc applied : {:.2f}\t l_penalty: {:.2f}\t t_penalty: {:.2f}".format(left_frc_clock, normed_left_frc, left_frc_penalty, foot_frc_penalty))
        print("l_vel phase : {:.2f}\t l_vel applied : {:.2f}\t l_penalty: {:.2f}\t t_penalty: {:.2f}".format(left_vel_clock, normed_left_vel, left_vel_penalty, foot_vel_penalty))
        # print("r_frc phase : {:.2f}\t r_frc applied : {:.2f}\t r_penalty: {:.2f}\t t_penalty: {:.2f}".format(right_frc_clock, normed_right_frc, right_frc_penalty, foot_frc_penalty))
        # print("r_vel phase : {:.2f}\t r_vel applied : {:.2f}\t r_penalty: {:.2f}\t t_penalty: {:.2f}".format(right_vel_clock, normed_right_vel, right_vel_penalty, foot_vel_penalty))
        print("reward: {12}\nfoot_orient:\t{0:.2f}, % = {1:.2f}\ncom_vel:\t{2:.2f}, % = {3:.2f}\nfoot_frc_penalty:\t{4:.2f}, % = {5:.2f}\nfoot_vel_penalty:\t{6:.2f}, % = {7:.2f}\nstraight_diff:\t{8:.2f}, % = {9:.2f}\ncom_orient:\t{10:.2f}, % = {11:.2f}".format(
        0.1 * np.exp(-foot_orient_error),         0.1 * np.exp(-foot_orient_error) / reward * 100,
        0.2 * np.exp(-com_vel_error),             0.2 * np.exp(-com_vel_error) / reward * 100,
        0.25 * foot_frc_penalty,                  0.25 * foot_frc_penalty / reward * 100,
        0.25 * foot_vel_penalty,                  0.25 * foot_vel_penalty / reward * 100,
        0.1  * np.exp(-straight_diff),            0.1 * np.exp(-straight_diff) / reward * 100,
        0.1  * np.exp(-com_orient_error),         0.1 * np.exp(-com_orient_error) / reward * 100,
        reward
        )
        )
        print("actual speed: {}\tcommanded speed: {}\n\n".format(np.linalg.norm(qvel[0:3]), self.speed))
    return reward

def no_speed_clock_reward(self, action):

    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())

    # These used for normalizing the foot forces and velocities
    desired_max_foot_frc = 400
    desired_max_foot_vel = 3.0
    orient_targ = np.array([1, 0, 0, 0])

    # state info
    # put a cap on the frc and vel so as to prevent the policy from learning to maximize them during phase.
    normed_left_frc = min(self.l_foot_frc, desired_max_foot_frc) / desired_max_foot_frc
    normed_right_frc = min(self.r_foot_frc, desired_max_foot_frc) / desired_max_foot_frc
    normed_left_vel = min(np.linalg.norm(self.l_foot_vel), desired_max_foot_vel) / desired_max_foot_vel
    normed_right_vel = min(np.linalg.norm(self.r_foot_vel), desired_max_foot_vel) / desired_max_foot_vel

    com_orient_error  = 0
    foot_orient_error  = 0
    straight_diff     = 0
    foot_vel_error    = 0
    foot_frc_error    = 0

    # com orient error
    com_orient_error += 10 * (1 - np.inner(orient_targ, qpos[3:7]) ** 2)

    # foot orient error
    foot_orient_error += 10 * (self.l_foot_orient_cost + self.r_foot_orient_cost)
    
    # straight difference penalty
    straight_diff = np.abs(qpos[1])
    if straight_diff < 0.05:
        straight_diff = 0
    # height deadzone is +- 0.2 meters
    height_diff = np.abs(qpos[2] - 1.0)
    if height_diff < 0.2:
        height_diff = 0
    straight_diff += height_diff

    # force/vel clock errors

    # These values represent if we want to allow foot foot forces / vels (1 -> good if forces/vels exist), or (-1 -> bad if forces/vels exist)
    left_frc_clock = self.left_clock[0](self.phase)
    right_frc_clock = self.right_clock[0](self.phase)
    left_vel_clock = self.left_clock[1](self.phase)
    right_vel_clock = self.right_clock[1](self.phase)
    
    left_frc_penalty = np.tanh(left_frc_clock * normed_left_frc)
    left_vel_penalty = np.tanh(left_vel_clock * normed_left_vel)
    right_frc_penalty = np.tanh(right_frc_clock * normed_right_frc)
    right_vel_penalty = np.tanh(right_vel_clock * normed_right_vel)

    foot_frc_penalty = left_frc_penalty + right_frc_penalty
    foot_vel_penalty = left_vel_penalty + right_vel_penalty

    reward = 0.1 * np.exp(-com_orient_error) +    \
                0.1 * np.exp(-foot_orient_error) + \
                0.1 * np.exp(-straight_diff) +      \
                0.25 * foot_frc_penalty +     \
                0.25 * foot_vel_penalty

    if self.debug:
        print("l_frc phase : {:.2f}\t l_frc applied : {:.2f}\t l_penalty: {:.2f}\t t_penalty: {:.2f}".format(left_frc_clock, normed_left_frc, left_frc_penalty, foot_frc_penalty))
        print("l_vel phase : {:.2f}\t l_vel applied : {:.2f}\t l_penalty: {:.2f}\t t_penalty: {:.2f}".format(left_vel_clock, normed_left_vel, left_vel_penalty, foot_vel_penalty))
        # print("r_frc phase : {:.2f}\t r_frc applied : {:.2f}\t r_penalty: {:.2f}\t t_penalty: {:.2f}".format(right_frc_clock, normed_right_frc, right_frc_penalty, foot_frc_penalty))
        # print("r_vel phase : {:.2f}\t r_vel applied : {:.2f}\t r_penalty: {:.2f}\t t_penalty: {:.2f}".format(right_vel_clock, normed_right_vel, right_vel_penalty, foot_vel_penalty))
        print("reward: {10}\nfoot_orient:\t{0:.2f}, % = {1:.2f}\nfoot_frc_penalty:\t{2:.2f}, % = {3:.2f}\nfoot_vel_penalty:\t{4:.2f}, % = {5:.2f}\nstraight_diff:\t{6:.2f}, % = {7:.2f}\ncom_orient:\t{8:.2f}, % = {9:.2f}".format(
        0.1 * np.exp(-foot_orient_error),         0.1 * np.exp(-foot_orient_error) / reward * 100,
        0.25 * foot_frc_penalty,                  0.25 * foot_frc_penalty / reward * 100,
        0.25 * foot_vel_penalty,                  0.25 * foot_vel_penalty / reward * 100,
        0.1  * np.exp(-straight_diff),            0.1 * np.exp(-straight_diff) / reward * 100,
        0.1  * np.exp(-com_orient_error),         0.1 * np.exp(-com_orient_error) / reward * 100,
        reward
        )
        )
        print("actual speed: {}\tcommanded speed: {}\n\n".format(np.linalg.norm(qvel[0:3]), self.speed))
    return reward

def aslip_clock_reward(self, action):

    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())

    # These used for normalizing the foot forces and velocities
    desired_max_foot_frc = 400
    desired_max_foot_vel = 3.0
    orient_targ = np.array([1, 0, 0, 0])

    # state info
    com_vel = qvel[0:3]
    lfoot_pos = self.l_foot_pos  # only care about xy relative to pelvis
    rfoot_pos = self.r_foot_pos  # only care about xy relative to pelvis
    # put a cap on the frc and vel so as to prevent the policy from learning to maximize them during phase.
    normed_left_frc = min(self.l_foot_frc, desired_max_foot_frc) / desired_max_foot_frc
    normed_right_frc = min(self.r_foot_frc, desired_max_foot_frc) / desired_max_foot_frc
    normed_left_vel = min(np.linalg.norm(self.l_foot_vel), desired_max_foot_vel) / desired_max_foot_vel
    normed_right_vel = min(np.linalg.norm(self.r_foot_vel), desired_max_foot_vel) / desired_max_foot_vel

    com_orient_error  = 0
    foot_orient_error  = 0
    foot_pos_error     = 0
    com_vel_error     = 0
    straight_diff     = 0
    foot_vel_error    = 0
    foot_frc_error    = 0

    phase_to_match = self.phase

    # ref state info
    ref_rfoot, ref_rvel, ref_lfoot, ref_lvel, ref_cpos, ref_cvel = get_ref_aslip_unaltered_state(self, phase_to_match)

    # com orient error
    com_orient_error += 1 - np.inner(orient_targ, qpos[3:7]) ** 2

    # foot orient error
    foot_orient_error += self.l_foot_orient_cost + self.r_foot_orient_cost

    # xy foot pos error
    for j in [0, 1]:
        foot_pos_error += np.linalg.norm(lfoot_pos[j] - ref_lfoot[j]) +  np.linalg.norm(rfoot_pos[j] - ref_rfoot[j])

    # com vel error
    for j in [0, 1, 2]:
        com_vel_error += np.linalg.norm(com_vel[j] - ref_cvel[j])
    
    # straight difference penalty
    straight_diff = np.abs(qpos[1])
    if straight_diff < 0.05:
        straight_diff = 0
    # height deadzone is +- 0.2 meters
    height_diff = np.abs(qpos[2] - 1.0)
    if height_diff < 0.2:
        height_diff = 0
    straight_diff += height_diff

    # force/vel clock errors

    # These values represent if we want to allow foot foot forces / vels (0 -> don't penalize), or penalize them (1 -> don't allow)
    left_frc_clock = self.left_clock(self.phase)
    right_frc_clock = self.right_clock(self.phase)
    left_vel_clock = -self.left_clock(self.phase)
    right_vel_clock = -self.right_clock(self.phase)
    
    left_frc_penalty = np.tanh(left_frc_clock * normed_left_frc)
    left_vel_penalty = np.tanh(left_vel_clock * normed_left_vel)
    right_frc_penalty = np.tanh(right_frc_clock * normed_right_frc)
    right_vel_penalty = np.tanh(right_vel_clock * normed_right_vel)

    foot_frc_penalty = left_frc_penalty + right_frc_penalty
    foot_vel_penalty = left_vel_penalty + right_vel_penalty

    reward = 0.05 * np.exp(-com_orient_error) +    \
                0.05 * np.exp(-foot_orient_error) +    \
                0.2 * np.exp(-foot_pos_error) +    \
                0.2 * np.exp(-com_vel_error) +    \
                0.1 * np.exp(-straight_diff) +      \
                0.2 * foot_frc_penalty +     \
                0.2 * foot_vel_penalty

    if self.debug:
        print("l_frc phase : {:.2f}\t l_frc applied : {:.2f}\t l_penalty: {:.2f}\t t_penalty: {:.2f}".format(left_frc_clock, normed_left_frc, left_frc_penalty, foot_frc_penalty))
        print("l_vel phase : {:.2f}\t l_vel applied : {:.2f}\t l_penalty: {:.2f}\t t_penalty: {:.2f}".format(left_vel_clock, normed_left_vel, left_vel_penalty, foot_vel_penalty))
        # print("r_frc phase : {:.2f}\t r_frc applied : {:.2f}\t r_penalty: {:.2f}\t t_penalty: {:.2f}".format(right_frc_clock, normed_right_frc, right_frc_penalty, foot_frc_penalty))
        # print("r_vel phase : {:.2f}\t r_vel applied : {:.2f}\t r_penalty: {:.2f}\t t_penalty: {:.2f}".format(right_vel_clock, normed_right_vel, right_vel_penalty, foot_vel_penalty))
        print("reward: {14}\nfoot:\t{0:.2f}, % = {1:.2f}\ncom_vel:\t{2:.2f}, % = {3:.2f}\nfoot_frc_penalty:\t{4:.2f}, % = {5:.2f}\nfoot_vel_penalty:\t{6:.2f}, % = {7:.2f}\nstraight_diff:\t{8:.2f}, % = {9:.2f}\nfoot_orient:\t{10:.2f}, % = {11:.2f}\ncom_orient:\t{12:.2f}, % = {13:.2f}".format(
        0.2 * np.exp(-foot_pos_error),         0.2 * np.exp(-foot_pos_error) / reward * 100,
        0.2 * np.exp(-com_vel_error),          0.2 * np.exp(-com_vel_error) / reward * 100,
        0.2 * foot_frc_penalty,                0.2 * foot_frc_penalty / reward * 100,
        0.2 * foot_vel_penalty,                0.2 * foot_vel_penalty / reward * 100,
        0.1  * np.exp(-straight_diff),         0.1 * np.exp(-straight_diff) / reward * 100,
        0.05 * np.exp(-foot_orient_error),     0.05 * np.exp(-foot_orient_error) / reward * 100,
        0.05 * np.exp(-com_orient_error),      0.05 * np.exp(-com_orient_error) / reward * 100,
        reward
        )
        )
        print("actual speed: {}\tdesired_speed: {}\tcommanded speed: {}\n\n".format(np.linalg.norm(qvel[0:3]), np.linalg.norm(ref_cvel), self.speed))
    return reward

def max_vel_clock_reward(self, action):

    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())

    # These used for normalizing the foot forces and velocities
    desired_max_foot_frc = 400
    desired_max_foot_vel = 3.0
    orient_targ = np.array([1, 0, 0, 0])

    # state info
    com_vel = qvel[0] # only care about x velocity
    # put a cap on the frc and vel so as to prevent the policy from learning to maximize them during phase.
    normed_left_frc = min(self.l_foot_frc, desired_max_foot_frc) / desired_max_foot_frc
    normed_right_frc = min(self.r_foot_frc, desired_max_foot_frc) / desired_max_foot_frc
    normed_left_vel = min(np.linalg.norm(self.l_foot_vel), desired_max_foot_vel) / desired_max_foot_vel
    normed_right_vel = min(np.linalg.norm(self.r_foot_vel), desired_max_foot_vel) / desired_max_foot_vel

    com_orient_error  = 0
    foot_orient_error  = 0
    com_vel_bonus     = 0
    straight_diff     = 0
    foot_vel_error    = 0
    foot_frc_error    = 0

    # com orient error
    com_orient_error += 15 * (1 - np.inner(orient_targ, qpos[3:7]) ** 2)

    # foot orient error
    foot_orient_error += 10 * (self.l_foot_orient_cost + self.r_foot_orient_cost)
    
    # com vel bonus
    com_vel_bonus += com_vel / 3.0

    # straight difference penalty
    straight_diff = np.abs(qpos[1])
    if straight_diff < 0.05:
        straight_diff = 0
    # height deadzone is +- 0.2 meters
    height_diff = np.abs(qpos[2] - 1.0)
    if height_diff < 0.2:
        height_diff = 0
    straight_diff += height_diff

    # force/vel clock errors

    # These values represent if we want to allow foot foot forces / vels (0 -> don't penalize), or penalize them (1 -> don't allow)
    left_frc_clock = self.left_clock(self.phase)
    right_frc_clock = self.right_clock(self.phase)
    left_vel_clock = -self.left_clock(self.phase)
    right_vel_clock = -self.right_clock(self.phase)
    
    left_frc_penalty = np.tanh(left_frc_clock * normed_left_frc)
    left_vel_penalty = np.tanh(left_vel_clock * normed_left_vel)
    right_frc_penalty = np.tanh(right_frc_clock * normed_right_frc)
    right_vel_penalty = np.tanh(right_vel_clock * normed_right_vel)

    foot_frc_penalty = left_frc_penalty + right_frc_penalty
    foot_vel_penalty = left_vel_penalty + right_vel_penalty

    reward = 0.1 * np.exp(-com_orient_error) +    \
                0.1 * np.exp(-foot_orient_error) +    \
                0.1 * np.exp(-straight_diff) +      \
                0.2 * foot_frc_penalty +     \
                0.2 * foot_vel_penalty + \
                0.3 * com_vel_bonus
    
    if self.debug:
        print("l_frc phase : {:.2f}\t l_frc applied : {:.2f}\t l_penalty: {:.2f}\t t_penalty: {:.2f}".format(left_frc_clock, normed_left_frc, left_frc_penalty, foot_frc_penalty))
        print("l_vel phase : {:.2f}\t l_vel applied : {:.2f}\t l_penalty: {:.2f}\t t_penalty: {:.2f}".format(left_vel_clock, normed_left_vel, left_vel_penalty, foot_vel_penalty))
        # print("r_frc phase : {:.2f}\t r_frc applied : {:.2f}\t r_penalty: {:.2f}\t t_penalty: {:.2f}".format(right_frc_clock, normed_right_frc, right_frc_penalty, foot_frc_penalty))
        # print("r_vel phase : {:.2f}\t r_vel applied : {:.2f}\t r_penalty: {:.2f}\t t_penalty: {:.2f}".format(right_vel_clock, normed_right_vel, right_vel_penalty, foot_vel_penalty))
        print("reward: {12}\nfoot_orient:\t{0:.2f}, % = {1:.2f}\ncom_vel_bonus:\t{2:.2f}, % = {3:.2f}\nfoot_frc_penalty:\t{4:.2f}, % = {5:.2f}\nfoot_vel_penalty:\t{6:.2f}, % = {7:.2f}\nstraight_diff:\t{8:.2f}, % = {9:.2f}\ncom_orient:\t{10:.2f}, % = {11:.2f}".format(
        0.1 * np.exp(-foot_orient_error),         0.1 * np.exp(-foot_orient_error) / reward * 100,
        0.3 * com_vel_bonus,                   0.3 * com_vel_bonus / reward * 100,
        0.2 * foot_frc_penalty,                0.2 * foot_frc_penalty / reward * 100,
        0.2 * foot_vel_penalty,                0.2 * foot_vel_penalty / reward * 100,
        0.1  * np.exp(-straight_diff),         0.1 * np.exp(-straight_diff) / reward * 100,
        0.1  * np.exp(-com_orient_error),      0.1 * np.exp(-com_orient_error) / reward * 100,
        reward
        )
        )
        print("actual speed: {}\tcommanded speed: {}\n\n".format(np.linalg.norm(qvel[0:3]), self.speed))
    return reward