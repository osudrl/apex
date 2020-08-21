import numpy as np
from ..quaternion_function import *

def speedmatch_reward(self):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())
    orient_targ = np.array([1, 0, 0, 0])
    speed_targ = np.array([self.speed, 0, 0])
    if self.time >= self.orient_time:
        orient_targ = euler2quat(z=self.orient_add, y=0, x=0)
        iquaternion = inverse_quaternion(orient_targ)
        speed_targ = rotate_by_quaternion(speed_targ, iquaternion)
        new_orient = quaternion_product(iquaternion, self.cassie_state.pelvis.orientation[:])
        if new_orient[0] < 0:
            new_orient = -new_orient
    forward_diff = np.abs(qvel[0] - speed_targ[0])
    orient_diff = 1 - np.inner(orient_targ, qpos[3:7]) ** 2
    # orient_diff = np.linalg.norm(qpos[3:7] - np.array([1, 0, 0, 0]))
    y_vel = np.abs(qvel[1] - speed_targ[1])
    if forward_diff < 0.05:
        forward_diff = 0
    if y_vel < 0.05:
        y_vel = 0
    straight_diff = 8*np.abs(qpos[1] - self.y_offset)
    if np.abs(qpos[1] - self.y_offset) < 0.05:
        straight_diff = 0
    if orient_diff < 5e-3:
        orient_diff = 0
    else:
        orient_diff *= 30

    reward = .5*np.exp(-forward_diff) + .2*np.exp(-orient_diff) \
                + .15*np.exp(-straight_diff) + .15*np.exp(-y_vel) \

    return reward

def speedmatch_footorient_hiprollvelact_reward(self):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())
    orient_targ = np.array([1, 0, 0, 0])
    speed_targ = np.array([self.speed, 0, 0])
    forward_diff = np.abs(qvel[0] - speed_targ[0])
    orient_diff = 1 - np.inner(orient_targ, qpos[3:7]) ** 2
    y_vel = np.abs(qvel[1] - speed_targ[1])
    if forward_diff < 0.05:
        forward_diff = 0
    if y_vel < 0.05:
        y_vel = 0
    straight_diff = np.abs(qpos[1])
    if straight_diff < 0.05:
        straight_diff = 0
    if orient_diff < 5e-3:
        orient_diff = 0
    else:
        orient_diff *= 30

    reward = .3*np.exp(-forward_diff) + .2*np.exp(-orient_diff) \
                + .1*np.exp(-straight_diff) + .1*np.exp(-y_vel) \
                + .075*np.exp(-self.l_foot_orient) + .075*np.exp(-self.r_foot_orient) \
                + .1*np.exp(-self.hiproll_cost) + 0.05*np.exp(-self.hiproll_act)

    return reward

def old_speed_reward(self):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())
    diff = np.abs(qvel[0] - self.speed)
    orient_diff = np.linalg.norm(qpos[3:7] - np.array([1, 0, 0, 0]))
    y_vel = np.abs(qvel[1])
    if diff < 0.05:
        diff = 0
    if y_vel < 0.03:
        y_vel = 0
    straight_diff = np.abs(qpos[1])
    if straight_diff < 0.05:
        straight_diff = 0
    reward = .5*np.exp(-diff) + .15*np.exp(-orient_diff) + .1*np.exp(-y_vel) + .25 * np.exp(-straight_diff)
    # print('reward: ', reward)

    return reward

def old_speed_footorient_reward(self):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())
    diff = np.abs(qvel[0] - self.speed)
    orient_diff = np.linalg.norm(qpos[3:7] - np.array([1, 0, 0, 0]))
    y_vel = np.abs(qvel[1])
    if diff < 0.05:
        diff = 0
    if y_vel < 0.03:
        y_vel = 0
    straight_diff = np.abs(qpos[1])
    if straight_diff < 0.05:
        straight_diff = 0

    # neutral_foot_orient = np.array([-0.24790886454547323, -0.24679713195445646, -0.6609396704367185, 0.663921021343526])
    # l_foot_orient = (1 - np.inner(neutral_foot_orient, self.sim.xquat("left-foot")) ** 2)
    # r_foot_orient = (1 - np.inner(neutral_foot_orient, self.sim.xquat("right-foot")) ** 2)

    reward = .4*np.exp(-diff) + .1*np.exp(-orient_diff) + .1*np.exp(-y_vel) + .2 * np.exp(-straight_diff) \
            + .1*np.exp(-self.l_foot_orient) + .1*np.exp(-self.r_foot_orient) 
    # print('reward: ', reward)

    return reward

def speedmatch_footheightvelflag_reward(self):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())
    orient_targ = np.array([1, 0, 0, 0])
    speed_targ = np.array([self.speed, 0, 0])
    forward_diff = np.abs(qvel[0] - speed_targ[0])
    orient_diff = 1 - np.inner(orient_targ, qpos[3:7]) ** 2
    # orient_diff = np.linalg.norm(qpos[3:7] - np.array([1, 0, 0, 0]))
    y_vel = np.abs(qvel[1] - speed_targ[1])
    if forward_diff < 0.05:
        forward_diff = 0
    if y_vel < 0.05:
        y_vel = 0
    straight_diff = np.abs(qpos[1])
    if straight_diff < 0.05:
        straight_diff = 0
    if orient_diff < 5e-3:
        orient_diff = 0
    else:
        orient_diff *= 30

    reward = .3*np.exp(-forward_diff) + .2*np.exp(-orient_diff) \
                + .1*np.exp(-straight_diff) + .1*np.exp(-y_vel) \
                + .15*np.exp(-self.l_foot_cost) + .15*np.exp(-self.r_foot_cost) 

    return reward

def speedmatch_footheightvelflag_even_reward(self):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())
    orient_targ = np.array([1, 0, 0, 0])
    speed_targ = np.array([self.speed, 0, 0])
    forward_diff = np.abs(qvel[0] - speed_targ[0])
    orient_diff = 1 - np.inner(orient_targ, qpos[3:7]) ** 2
    # orient_diff = np.linalg.norm(qpos[3:7] - np.array([1, 0, 0, 0]))
    y_vel = np.abs(qvel[1] - speed_targ[1])
    if forward_diff < 0.05:
        forward_diff = 0
    if y_vel < 0.05:
        y_vel = 0
    straight_diff = np.abs(qpos[1])
    if straight_diff < 0.05:
        straight_diff = 0
    if orient_diff < 5e-3:
        orient_diff = 0
    else:
        orient_diff *= 30

    reward = .3*np.exp(-forward_diff) + .2*np.exp(-orient_diff) \
                + .1*np.exp(-straight_diff) + .1*np.exp(-y_vel) \
                + .15*np.exp(-self.l_foot_cost_even) + .15*np.exp(-self.r_foot_cost_even) 

    return reward

def speedmatch_footheightsmooth_footorient_reward(self):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())
    orient_targ = np.array([1, 0, 0, 0])
    speed_targ = np.array([self.speed, 0, 0])
    forward_diff = np.abs(qvel[0] - speed_targ[0])
    orient_diff = 1 - np.inner(orient_targ, qpos[3:7]) ** 2
    # orient_diff = np.linalg.norm(qpos[3:7] - np.array([1, 0, 0, 0]))
    y_vel = np.abs(qvel[1] - speed_targ[1])
    if forward_diff < 0.05:
        forward_diff = 0
    if y_vel < 0.05:
        y_vel = 0
    straight_diff = np.abs(qpos[1])
    if straight_diff < 0.05:
        straight_diff = 0
    if orient_diff < 5e-3:
        orient_diff = 0
    else:
        orient_diff *= 30

    reward = .2*np.exp(-forward_diff) + .1*np.exp(-orient_diff) \
                + .1*np.exp(-straight_diff) + .1*np.exp(-y_vel) \
                + .15*np.exp(-self.l_foot_cost_smooth) + .15*np.exp(-self.r_foot_cost_smooth) \
                + .1*np.exp(-self.l_foot_orient) + .1*np.exp(-self.r_foot_orient)

    return reward

def speedmatch_footheightsmooth_footorient_hiproll_torquecost_reward(self):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())
    orient_targ = np.array([1, 0, 0, 0])
    speed_targ = np.array([self.speed, 0, 0])
    forward_diff = np.abs(qvel[0] - speed_targ[0])
    orient_diff = 1 - np.inner(orient_targ, qpos[3:7]) ** 2
    # orient_diff = np.linalg.norm(qpos[3:7] - np.array([1, 0, 0, 0]))
    y_vel = np.abs(qvel[1] - speed_targ[1])
    if forward_diff < 0.05:
        forward_diff = 0
    if y_vel < 0.05:
        y_vel = 0
    straight_diff = np.abs(qpos[1])
    if straight_diff < 0.05:
        straight_diff = 0
    if orient_diff < 5e-3:
        orient_diff = 0
    else:
        orient_diff *= 30

    reward = .2*np.exp(-forward_diff) + .1*np.exp(-orient_diff) \
                + .05*np.exp(-straight_diff) + .05*np.exp(-y_vel) \
                + .15*np.exp(-self.l_foot_cost_smooth) + .15*np.exp(-self.r_foot_cost_smooth) \
                + .075*np.exp(-self.l_foot_orient) + .075*np.exp(-self.r_foot_orient) \
                + .1*np.exp(-self.hiproll_cost) + .05*np.exp(-self.torque_cost)

    return reward

def speedmatch_footheightsmooth_footorient_hiproll_reward(self):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())
    orient_targ = np.array([1, 0, 0, 0])
    speed_targ = np.array([self.speed, 0, 0])
    forward_diff = np.abs(qvel[0] - speed_targ[0])
    orient_diff = 1 - np.inner(orient_targ, qpos[3:7]) ** 2
    # orient_diff = np.linalg.norm(qpos[3:7] - np.array([1, 0, 0, 0]))
    y_vel = np.abs(qvel[1] - speed_targ[1])
    if forward_diff < 0.05:
        forward_diff = 0
    if y_vel < 0.05:
        y_vel = 0
    straight_diff = np.abs(qpos[1])
    if straight_diff < 0.05:
        straight_diff = 0
    if orient_diff < 5e-3:
        orient_diff = 0
    else:
        orient_diff *= 30

    reward = .2*np.exp(-forward_diff) + .1*np.exp(-orient_diff) \
                + .05*np.exp(-straight_diff) + .05*np.exp(-y_vel) \
                + .15*np.exp(-self.l_foot_cost_smooth) + .15*np.exp(-self.r_foot_cost_smooth) \
                + .1*np.exp(-self.l_foot_orient) + .1*np.exp(-self.r_foot_orient) \
                + .1*np.exp(-self.hiproll_cost)

    return reward

def speedmatch_footheightsmooth_footorient_hiprollvelact_reward(self):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())
    orient_targ = np.array([1, 0, 0, 0])
    speed_targ = np.array([self.speed, 0, 0])
    forward_diff = np.abs(qvel[0] - speed_targ[0])
    orient_diff = 1 - np.inner(orient_targ, qpos[3:7]) ** 2
    # orient_diff = np.linalg.norm(qpos[3:7] - np.array([1, 0, 0, 0]))
    y_vel = np.abs(qvel[1] - speed_targ[1])
    if forward_diff < 0.05:
        forward_diff = 0
    if y_vel < 0.05:
        y_vel = 0
    straight_diff = np.abs(qpos[1])
    if straight_diff < 0.05:
        straight_diff = 0
    if orient_diff < 5e-3:
        orient_diff = 0
    else:
        orient_diff *= 30

    reward = .2*np.exp(-forward_diff) + .1*np.exp(-orient_diff) \
                + .05*np.exp(-straight_diff) + .05*np.exp(-y_vel) \
                + .15*np.exp(-self.l_foot_cost_smooth) + .15*np.exp(-self.r_foot_cost_smooth) \
                + .075*np.exp(-self.l_foot_orient) + .075*np.exp(-self.r_foot_orient) \
                + .1*np.exp(-self.hiproll_cost) + 0.05*np.exp(-self.hiproll_act)

    return reward

def speedmatch_footheightsmooth_footorient_hiprollyawvelact_reward(self):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())
    orient_targ = np.array([1, 0, 0, 0])
    speed_targ = np.array([self.speed, 0, 0])
    forward_diff = np.abs(qvel[0] - speed_targ[0])
    orient_diff = 1 - np.inner(orient_targ, qpos[3:7]) ** 2
    # orient_diff = np.linalg.norm(qpos[3:7] - np.array([1, 0, 0, 0]))
    y_vel = np.abs(qvel[1] - speed_targ[1])
    if forward_diff < 0.05:
        forward_diff = 0
    if y_vel < 0.05:
        y_vel = 0
    straight_diff = np.abs(qpos[1])
    if straight_diff < 0.05:
        straight_diff = 0
    if orient_diff < 5e-3:
        orient_diff = 0
    else:
        orient_diff *= 30

    reward = .2*np.exp(-forward_diff) + .1*np.exp(-orient_diff) \
                + .05*np.exp(-straight_diff) + .05*np.exp(-y_vel) \
                + .15*np.exp(-self.l_foot_cost_smooth) + .15*np.exp(-self.r_foot_cost_smooth) \
                + .05*np.exp(-self.l_foot_orient) + .05*np.exp(-self.r_foot_orient) \
                + .05*np.exp(-self.hiproll_cost) + 0.05*np.exp(-self.hiproll_act) \
                + .05*np.exp(-self.hipyaw_vel) + 0.05*np.exp(-self.hipyaw_act)

    return reward

def speedmatch_footheightsmooth_footorient_hiprollyawphasetorque_reward(self):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())
    orient_targ = np.array([1, 0, 0, 0])
    speed_targ = np.array([self.speed, 0, 0])
    forward_diff = np.abs(qvel[0] - speed_targ[0])
    orient_diff = 1 - np.inner(orient_targ, qpos[3:7]) ** 2
    # orient_diff = np.linalg.norm(qpos[3:7] - np.array([1, 0, 0, 0]))
    y_vel = np.abs(qvel[1] - speed_targ[1])
    if forward_diff < 0.05:
        forward_diff = 0
    if y_vel < 0.05:
        y_vel = 0
    straight_diff = np.abs(qpos[1])
    if straight_diff < 0.05:
        straight_diff = 0
    if orient_diff < 5e-3:
        orient_diff = 0
    else:
        orient_diff *= 30

    reward = .2*np.exp(-forward_diff) + .1*np.exp(-orient_diff) \
                + .05*np.exp(-straight_diff) + .05*np.exp(-y_vel) \
                + .15*np.exp(-self.l_foot_cost_smooth) + .15*np.exp(-self.r_foot_cost_smooth) \
                + .05*np.exp(-self.l_foot_orient) + .05*np.exp(-self.r_foot_orient) \
                + .1*np.exp(-self.left_rollyaw_torque_cost) + 0.1*np.exp(-self.right_rollyaw_torque_cost) \

    return reward

def speedmatch_footvarclock_footorient_hiprollyawvelact_reward(self):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())
    orient_targ = np.array([1, 0, 0, 0])
    speed_targ = np.array([self.speed, 0, 0])
    forward_diff = np.abs(qvel[0] - speed_targ[0])
    orient_diff = 1 - np.inner(orient_targ, qpos[3:7]) ** 2
    # orient_diff = np.linalg.norm(qpos[3:7] - np.array([1, 0, 0, 0]))
    y_vel = np.abs(qvel[1] - speed_targ[1])
    if forward_diff < 0.05:
        forward_diff = 0
    if y_vel < 0.05:
        y_vel = 0
    straight_diff = np.abs(qpos[1])
    if straight_diff < 0.05:
        straight_diff = 0
    if orient_diff < 5e-3:
        orient_diff = 0
    else:
        orient_diff *= 30

    reward = .2*np.exp(-forward_diff) + .1*np.exp(-orient_diff) \
                + .05*np.exp(-straight_diff) + .05*np.exp(-y_vel) \
                + .15*np.exp(-self.l_foot_cost_var) + .15*np.exp(-self.r_foot_cost_var) \
                + .05*np.exp(-self.l_foot_orient) + .05*np.exp(-self.r_foot_orient) \
                + .05*np.exp(-self.hiproll_cost) + 0.05*np.exp(-self.hiproll_act) \
                + .05*np.exp(-self.hipyaw_vel) + 0.05*np.exp(-self.hipyaw_act)

    return reward

def speedmatch_footheightsmooth_footorient_stablepel_reward(self):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())
    orient_targ = np.array([1, 0, 0, 0])
    speed_targ = np.array([self.speed, 0, 0])
    forward_diff = np.abs(qvel[0] - speed_targ[0])
    orient_diff = 1 - np.inner(orient_targ, qpos[3:7]) ** 2
    # orient_diff = np.linalg.norm(qpos[3:7] - np.array([1, 0, 0, 0]))
    y_vel = np.abs(qvel[1] - speed_targ[1])
    if forward_diff < 0.05:
        forward_diff = 0
    if y_vel < 0.05:
        y_vel = 0
    straight_diff = np.abs(qpos[1])
    if straight_diff < 0.05:
        straight_diff = 0
    if orient_diff < 5e-3:
        orient_diff = 0
    else:
        orient_diff *= 30

    reward = .2*np.exp(-forward_diff) + .1*np.exp(-orient_diff) \
                + .05*np.exp(-straight_diff) + .05*np.exp(-y_vel) \
                + .15*np.exp(-self.l_foot_cost_smooth) + .15*np.exp(-self.r_foot_cost_smooth) \
                + .1*np.exp(-self.l_foot_orient) + .1*np.exp(-self.r_foot_orient) \
                + .1*np.exp(-self.pel_stable)

    return reward

def speedmatch_footheightsmooth_footorient_hiprollvelact_orientchange_reward(self):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())
    quaternion = euler2quat(z=self.orient_add, y=0, x=0)
    iquaternion = inverse_quaternion(quaternion)

    orient_targ = np.array([1, 0, 0, 0])
    actual_orient = quaternion_product(iquaternion, qpos[3:7])
    speed_targ = rotate_by_quaternion(np.array([self.speed, 0, 0]), iquaternion)
    x_vel = np.abs(qvel[0] - speed_targ[0])
    orient_diff = 1 - np.inner(orient_targ, actual_orient) ** 2
    # orient_diff = np.linalg.norm(qpos[3:7] - np.array([1, 0, 0, 0]))
    y_vel = np.abs(qvel[1] - speed_targ[1])
    if x_vel < 0.05:
        x_vel = 0
    if y_vel < 0.05:
        y_vel = 0
    if orient_diff < 5e-3:
        orient_diff = 0
    else:
        orient_diff *= 30

    reward = .15*np.exp(-x_vel) + .15*np.exp(-y_vel)  + .1*np.exp(-orient_diff) \
                + .15*np.exp(-self.l_foot_cost_smooth) + .15*np.exp(-self.r_foot_cost_smooth) \
                + .075*np.exp(-self.l_foot_orient) + .075*np.exp(-self.r_foot_orient) \
                + .1*np.exp(-self.hiproll_cost) + 0.05*np.exp(-self.hiproll_act)

    return reward


def speedmatch_footclock_footorient_reward(self):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())
    orient_targ = np.array([1, 0, 0, 0])
    speed_targ = np.array([self.speed, 0, 0])
    forward_diff = np.abs(qvel[0] - speed_targ[0])
    orient_diff = 1 - np.inner(orient_targ, qpos[3:7]) ** 2
    # orient_diff = np.linalg.norm(qpos[3:7] - np.array([1, 0, 0, 0]))
    y_vel = np.abs(qvel[1] - speed_targ[1])
    if forward_diff < 0.05:
        forward_diff = 0
    if y_vel < 0.05:
        y_vel = 0
    straight_diff = np.abs(qpos[1])
    if straight_diff < 0.05:
        straight_diff = 0
    if orient_diff < 5e-3:
        orient_diff = 0
    else:
        orient_diff *= 30

    reward = .2*np.exp(-forward_diff) + .1*np.exp(-orient_diff) \
                + .1*np.exp(-straight_diff) + .1*np.exp(-y_vel) \
                + .15*np.exp(-self.l_foot_cost_clock) + .15*np.exp(-self.r_foot_cost_clock) \
                + .1*np.exp(-self.l_foot_orient) + .1*np.exp(-self.r_foot_orient)

    return reward

def speedmatch_footheightvelflag_even_footorient_reward(self):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())
    orient_targ = np.array([1, 0, 0, 0])
    speed_targ = np.array([self.speed, 0, 0])
    forward_diff = np.abs(qvel[0] - speed_targ[0])
    orient_diff = 1 - np.inner(orient_targ, qpos[3:7]) ** 2
    # orient_diff = np.linalg.norm(qpos[3:7] - np.array([1, 0, 0, 0]))
    y_vel = np.abs(qvel[1] - speed_targ[1])
    if forward_diff < 0.05:
        forward_diff = 0
    if y_vel < 0.05:
        y_vel = 0
    straight_diff = np.abs(qpos[1])
    if straight_diff < 0.05:
        straight_diff = 0
    if orient_diff < 5e-3:
        orient_diff = 0
    else:
        orient_diff *= 30

    reward = .2*np.exp(-forward_diff) + .1*np.exp(-orient_diff) \
                + .1*np.exp(-straight_diff) + .1*np.exp(-y_vel) \
                + .15*np.exp(-self.l_foot_cost_even) + .15*np.exp(-self.r_foot_cost_even) \
                + .1*np.exp(-self.l_foot_orient) + .1*np.exp(-self.r_foot_orient) 

    return reward

def speedmatch_footheightvelflag_even_footorient_footdist_reward(self):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())
    orient_targ = np.array([1, 0, 0, 0])
    speed_targ = np.array([self.speed, 0, 0])
    forward_diff = np.abs(qvel[0] - speed_targ[0])
    orient_diff = 1 - np.inner(orient_targ, qpos[3:7]) ** 2
    # orient_diff = np.linalg.norm(qpos[3:7] - np.array([1, 0, 0, 0]))
    y_vel = np.abs(qvel[1] - speed_targ[1])
    if forward_diff < 0.05:
        forward_diff = 0
    if y_vel < 0.05:
        y_vel = 0
    straight_diff = np.abs(qpos[1])
    if straight_diff < 0.05:
        straight_diff = 0
    if orient_diff < 5e-3:
        orient_diff = 0
    else:
        orient_diff *= 30

    ######## Foot position penalty ########
    foot_pos = np.zeros(6)
    self.sim.foot_pos(foot_pos)
    foot_dist = np.linalg.norm(foot_pos[0:2]-foot_pos[3:5])
    foot_penalty = 0
    if foot_dist < 0.2:
       foot_penalty = -0.2

    reward = .2*np.exp(-forward_diff) + .1*np.exp(-orient_diff) \
                + .1*np.exp(-straight_diff) + .1*np.exp(-y_vel) \
                + .15*np.exp(-self.l_foot_cost_even) + .15*np.exp(-self.r_foot_cost_even) \
                + .1*np.exp(-self.l_foot_orient) + .1*np.exp(-self.r_foot_orient) \
                + foot_penalty

    return reward

def speedmatch_footheightvelflag_even_footorient_footdist_torquecost_reward(self):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())
    orient_targ = np.array([1, 0, 0, 0])
    speed_targ = np.array([self.speed, 0, 0])
    forward_diff = np.abs(qvel[0] - speed_targ[0])
    orient_diff = 1 - np.inner(orient_targ, qpos[3:7]) ** 2
    # orient_diff = np.linalg.norm(qpos[3:7] - np.array([1, 0, 0, 0]))
    y_vel = np.abs(qvel[1] - speed_targ[1])
    if forward_diff < 0.05:
        forward_diff = 0
    if y_vel < 0.05:
        y_vel = 0
    straight_diff = np.abs(qpos[1])
    if straight_diff < 0.05:
        straight_diff = 0
    if orient_diff < 5e-3:
        orient_diff = 0
    else:
        orient_diff *= 30

    ######## Foot position penalty ########
    foot_pos = np.zeros(6)
    self.sim.foot_pos(foot_pos)
    foot_dist = np.linalg.norm(foot_pos[0:2]-foot_pos[3:5])
    foot_penalty = 0
    if foot_dist < 0.15:
       foot_penalty = -0.2

    reward = .2*np.exp(-forward_diff) + .1*np.exp(-orient_diff) \
                + .075*np.exp(-straight_diff) + .075*np.exp(-y_vel) \
                + .15*np.exp(-self.l_foot_cost_even) + .15*np.exp(-self.r_foot_cost_even) \
                + .075*np.exp(-self.l_foot_orient) + .075*np.exp(-self.r_foot_orient) \
                + .1*np.exp(-self.torque_cost) + foot_penalty

    return reward

def speedmatch_footheightvelflag_even_footorient_footdist_torquecost_smooth_reward(self):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())
    orient_targ = np.array([1, 0, 0, 0])
    speed_targ = np.array([self.speed, 0, 0])
    forward_diff = np.abs(qvel[0] - speed_targ[0])
    orient_diff = 1 - np.inner(orient_targ, qpos[3:7]) ** 2
    # orient_diff = np.linalg.norm(qpos[3:7] - np.array([1, 0, 0, 0]))
    y_vel = np.abs(qvel[1] - speed_targ[1])
    if forward_diff < 0.05:
        forward_diff = 0
    if y_vel < 0.05:
        y_vel = 0
    straight_diff = np.abs(qpos[1])
    if straight_diff < 0.05:
        straight_diff = 0
    if orient_diff < 5e-3:
        orient_diff = 0
    else:
        orient_diff *= 30

    ######## Foot position penalty ########
    foot_pos = np.zeros(6)
    self.sim.foot_pos(foot_pos)
    foot_dist = np.linalg.norm(foot_pos[0:2]-foot_pos[3:5])
    foot_penalty = 0
    if foot_dist < 0.15:
       foot_penalty = -0.2

    reward = .2*np.exp(-forward_diff) + .05*np.exp(-orient_diff) \
                + .05*np.exp(-straight_diff) + .05*np.exp(-y_vel) \
                + .15*np.exp(-self.l_foot_cost_even) + .15*np.exp(-self.r_foot_cost_even) \
                + .075*np.exp(-self.l_foot_orient) + .075*np.exp(-self.r_foot_orient) \
                + .1*np.exp(-self.torque_cost) + .1*np.exp(-self.smooth_cost) + foot_penalty

    return reward

def speedmatch_footheightvelflag_even_footorient_smooth_reward(self):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())
    orient_targ = np.array([1, 0, 0, 0])
    speed_targ = np.array([self.speed, 0, 0])
    forward_diff = np.abs(qvel[0] - speed_targ[0])
    orient_diff = 1 - np.inner(orient_targ, qpos[3:7]) ** 2
    # orient_diff = np.linalg.norm(qpos[3:7] - np.array([1, 0, 0, 0]))
    y_vel = np.abs(qvel[1] - speed_targ[1])
    if forward_diff < 0.05:
        forward_diff = 0
    if y_vel < 0.05:
        y_vel = 0
    straight_diff = np.abs(qpos[1])
    if straight_diff < 0.05:
        straight_diff = 0
    if orient_diff < 5e-3:
        orient_diff = 0
    else:
        orient_diff *= 30

    reward = .2*np.exp(-forward_diff) + .1*np.exp(-orient_diff) \
                + .05*np.exp(-straight_diff) + .05*np.exp(-y_vel) \
                + .15*np.exp(-self.l_foot_cost_even) + .15*np.exp(-self.r_foot_cost_even) \
                + .1*np.exp(-self.l_foot_orient) + .1*np.exp(-self.r_foot_orient) \
                + .1*np.exp(-self.smooth_cost) \

    return reward

def speedmatch_footheightvelflag_even_capzvel_reward(self):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())
    orient_targ = np.array([1, 0, 0, 0])
    speed_targ = np.array([self.speed, 0, 0])
    forward_diff = np.abs(qvel[0] - speed_targ[0])
    orient_diff = 1 - np.inner(orient_targ, qpos[3:7]) ** 2
    # orient_diff = np.linalg.norm(qpos[3:7] - np.array([1, 0, 0, 0]))
    y_vel = np.abs(qvel[1] - speed_targ[1])
    if forward_diff < 0.05:
        forward_diff = 0
    if y_vel < 0.05:
        y_vel = 0
    straight_diff = np.abs(qpos[1])
    if straight_diff < 0.05:
        straight_diff = 0
    if orient_diff < 5e-3:
        orient_diff = 0
    else:
        orient_diff *= 30
    rfoot_vel_penalty = 0
    lfoot_vel_penalty = 0
    if self.r_high and np.abs(self.rfoot_vel[2]) > .6:
        rfoot_vel_penalty = -.4
    if self.l_high and np.abs(self.lfoot_vel[2]) > .6:
        lfoot_vel_penalty = -.4

    reward = .3*np.exp(-forward_diff) + .2*np.exp(-orient_diff) \
                + .1*np.exp(-straight_diff) + .1*np.exp(-y_vel) \
                + .15*np.exp(-self.l_foot_cost_even) + .15*np.exp(-self.r_foot_cost_even) \
                + lfoot_vel_penalty + rfoot_vel_penalty

    return reward


def speedmatch_footorient_reward(self):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())
    orient_targ = np.array([1, 0, 0, 0])
    speed_targ = np.array([self.speed, 0, 0])
    if self.time >= self.orient_time:
        orient_targ = euler2quat(z=self.orient_add, y=0, x=0)
        iquaternion = inverse_quaternion(orient_targ)
        speed_targ = rotate_by_quaternion(speed_targ, iquaternion)
        new_orient = quaternion_product(iquaternion, self.cassie_state.pelvis.orientation[:])
        if new_orient[0] < 0:
            new_orient = -new_orient
    forward_diff = np.abs(qvel[0] - speed_targ[0])
    orient_diff = 1 - np.inner(orient_targ, qpos[3:7]) ** 2
    # orient_diff = np.linalg.norm(qpos[3:7] - np.array([1, 0, 0, 0]))
    y_vel = np.abs(qvel[1] - speed_targ[1])
    if forward_diff < 0.05:
        forward_diff = 0
    if y_vel < 0.05:
        y_vel = 0
    straight_diff = 8*np.abs(qpos[1] - self.y_offset)
    if np.abs(qpos[1] - self.y_offset) < 0.05:
        straight_diff = 0
    if orient_diff < 5e-3:
        orient_diff = 0
    else:
        orient_diff *= 30

    reward = .3*np.exp(-forward_diff) + .2*np.exp(-orient_diff) \
                + .15*np.exp(-straight_diff) + .15*np.exp(-y_vel) \
                + .1*np.exp(-self.l_foot_orient) + .1*np.exp(-self.r_foot_orient) 

    return reward

def speedmatch_footorient_joint_smooth_reward(self):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())
    
    orient_targ = np.array([1, 0, 0, 0])
    speed_targ = np.array([self.speed, 0, 0])
    if self.time >= self.orient_time:
        orient_targ = euler2quat(z=self.orient_add, y=0, x=0)
        iquaternion = inverse_quaternion(orient_targ)
        speed_targ = rotate_by_quaternion(speed_targ, iquaternion)
        new_orient = quaternion_product(iquaternion, self.cassie_state.pelvis.orientation[:])
        if new_orient[0] < 0:
            new_orient = -new_orient
    forward_diff = np.abs(qvel[0] - speed_targ[0])
    orient_diff = 1 - np.inner(orient_targ, qpos[3:7]) ** 2
    # orient_diff = np.linalg.norm(qpos[3:7] - np.array([1, 0, 0, 0]))
    y_vel = np.abs(qvel[1] - speed_targ[1])
    if forward_diff < 0.05:
        forward_diff = 0
    if y_vel < 0.05:
        y_vel = 0
    straight_diff = 8*np.abs(qpos[1] - self.y_offset)
    if np.abs(qpos[1] - self.y_offset) < 0.05:
        straight_diff = 0
    if orient_diff < 5e-3:
        orient_diff = 0
    else:
        orient_diff *= 30

    reward = .25*np.exp(-forward_diff) + .1*np.exp(-orient_diff) \
                + .1*np.exp(-straight_diff) + .1*np.exp(-y_vel) \
                + .1*np.exp(-self.l_foot_orient) + .1*np.exp(-self.r_foot_orient) \
                + .1*np.exp(-self.smooth_cost) \
                + .15*np.exp(-self.joint_error) 

    return reward

def speedmatch_footorient_footheightvel_smooth_reward(self):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())
    
    orient_targ = np.array([1, 0, 0, 0])
    speed_targ = np.array([self.speed, 0, 0])
    if self.time >= self.orient_time:
        orient_targ = euler2quat(z=self.orient_add, y=0, x=0)
        iquaternion = inverse_quaternion(orient_targ)
        speed_targ = rotate_by_quaternion(speed_targ, iquaternion)
        new_orient = quaternion_product(iquaternion, self.cassie_state.pelvis.orientation[:])
        if new_orient[0] < 0:
            new_orient = -new_orient
    forward_diff = np.abs(qvel[0] - speed_targ[0])
    orient_diff = 1 - np.inner(orient_targ, qpos[3:7]) ** 2
    # orient_diff = np.linalg.norm(qpos[3:7] - np.array([1, 0, 0, 0]))
    y_vel = np.abs(qvel[1] - speed_targ[1])
    if forward_diff < 0.05:
        forward_diff = 0
    if y_vel < 0.05:
        y_vel = 0
    straight_diff = 8*np.abs(qpos[1] - self.y_offset)
    if np.abs(qpos[1] - self.y_offset) < 0.05:
        straight_diff = 0
    if orient_diff < 5e-3:
        orient_diff = 0
    else:
        orient_diff *= 30

    reward = .2*np.exp(-forward_diff) + .1*np.exp(-orient_diff) \
                + .1*np.exp(-straight_diff) + .1*np.exp(-y_vel) \
                + .1*np.exp(-self.lf_heightvel) + .1*np.exp(-self.rf_heightvel) \
                + .1*np.exp(-self.l_foot_orient) + .1*np.exp(-self.r_foot_orient) \
                + .1*np.exp(-self.smooth_cost) 

    return reward

def speedmatch_heuristic_reward(self):
    ######## Pelvis z accel penalty #########
    pelaccel = np.abs(self.cassie_state.pelvis.translationalAcceleration[2])
    pelaccel_penalty = 0
    if pelaccel > 5:
        pelaccel_penalty = (pelaccel - 5) / 10
    pelbonus = 0
    if 8 < pelaccel < 10:
        pelbonus = 0.2
    ######## Foot position penalty ########
    foot_pos = np.zeros(6)
    self.sim.foot_pos(foot_pos)
    foot_dist = np.linalg.norm(foot_pos[0:2]-foot_pos[3:5])
    foot_penalty = 0
    if foot_dist < 0.14:
       foot_penalty = 0.2
    ######## Foot force penalty ########
    foot_forces = self.sim.get_foot_forces()
    lforce = max((foot_forces[0] - 350)/1000, 0)
    rforce = max((foot_forces[1] - 350)/1000, 0)
    forcebonus = 0
    # print("foot force: ", lforce, rforce)
    # lbonus = max((800 - foot_forces[0])/1000, 0)
    if foot_forces[0] <= 1000 and foot_forces[1] <= 1000:
        forcebonus = foot_forces[0] / 5000 + foot_forces[1] / 5000
    ######## Foot velocity penalty ########
    lfoot_vel_bonus = 0     
    rfoot_vel_bonus = 0
    # if self.prev_foot is not None and foot_pos[2] < 0.3 and foot_pos[5] < 0.3:
    #     lfoot_vel = np.abs(foot_pos[2] - self.prev_foot[2]) / 0.03 * 0.03
    #     rfoot_vel = np.abs(foot_pos[5] - self.prev_foot[5]) / 0.03 * 0.03
    # if self.l_high:
    #     lfoot_vel_bonus = self.lfoot_vel * 0.3
    # if self.r_high:
    #     rfoot_vel_bonus = self.rfoot_vel * 0.3
    ######## Foot orientation ########
    lfoot_orient = 1 - np.inner(np.array([1, 0, 0, 0]), self.cassie_state.leftFoot.orientation[:]) ** 2
    rfoot_orient = 1 - np.inner(np.array([1, 0, 0, 0]), self.cassie_state.rightFoot.orientation[:]) ** 2
    ####### Hip yaw ########
    rhipyaw = np.abs(qpos[22])
    lhipyaw = np.abs(qpos[8])
    if lhipyaw < 0.05:
        lhipyaw = 0
    if rhipyaw < 0.05:
        rhipyaw = 0
    ####### Hip roll penalty #########
    lhiproll = np.abs(qpos[7])
    rhiproll = np.abs(qpos[21])
    if lhiproll < 0.05:
        lhiproll = 0
    if rhiproll < 0.05:
        rhiproll = 0
    ####### Prev action penalty ########
    if self.prev_action is not None:
        prev_penalty = np.linalg.norm(self.curr_action - self.prev_action) / 10 #* (30/self.simrate)
    else:
        prev_penalty = 0

    reward = .2*np.exp(-self.com_vel_error) + .1*np.exp(-self.com_error) + .1*np.exp(-self.orientation_error) \
            + .1*np.exp(-20*self.l_foot_diff) + .1*np.exp(-5*self.l_footvel_diff) \
            + .1*np.exp(-20*self.r_foot_diff) + .1*np.exp(-5*self.r_footvel_diff) \
            + .1*np.exp(-lfoot_orient) + .1*np.exp(-rfoot_orient)
    # reward = .4*np.exp(-forward_diff) + .3*np.exp(-orient_diff) \
                # + .15*np.exp(-straight_diff) + .15*np.exp(-y_vel) \
                # + .1*np.exp(-self.l_foot_orient) + .1*np.exp(-self.r_foot_orient) \
                # + .1*np.exp(-self.smooth_cost) \
                # + .15*np.exp(-self.joint_error) 
                # + .1*np.exp(-self.torque_cost) + .1*np.exp(-self.smooth_cost) #\
                #
                #  + .075*np.exp(-10*lhipyaw) + .075*np.exp(-10*rhipyaw) + .075*np.exp(-10*lhiproll) + .075*np.exp(-10*rhiproll)
    #         + .1*np.exp(-20*self.l_foot_diff) + .1*np.exp(-20*self.r_foot_diff) \
    #         + .1*np.exp(-5*self.l_footvel_diff) + .1*np.exp(-5*self.r_footvel_diff)
    # - lfoot_vel_bonus - rfoot_vel_bonus - foot_penalty
    # - lforce - rforce
    #+ pelbonus- pelaccel_penalty - foot_penalty