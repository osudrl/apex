import numpy as np
from ..quaternion_function import *

def tray_box_reward(self):

    reward = .1*np.exp(-self.forward_cost) + .1*np.exp(-self.orient_cost) \
                + .05*np.exp(-self.straight_cost) + .05*np.exp(-self.yvel_cost) \
                + .1*np.exp(-self.l_foot_cost_forcevel) + .1*np.exp(-self.r_foot_cost_forcevel) \
                + .05*np.exp(-self.l_foot_cost_pos) + .05*np.exp(-self.r_foot_cost_pos) \
                + .05*np.exp(-self.l_foot_orient) + .05*np.exp(-self.r_foot_orient) \
                + .05*np.exp(-self.hiproll_cost) + .05*np.exp(-self.hipyaw_vel) \
                + .05*np.exp(-self.pel_transacc) + .05*np.exp(-self.pel_rotacc) \
                + .025*np.exp(-self.act_cost) + 0.025*np.exp(-self.torque_penalty) \
                + .05*np.exp(-self.tray_box_cost)

    qpos = self.sim.qpos_full()
    if qpos[-5] <= qpos[2]: # Get no reward if box below pelvis height (box fell off)
        reward = 0

    return reward

def tray_box_reward_easy(self):
    reward = .1*np.exp(-self.forward_cost) + .1*np.exp(-self.orient_cost) \
                + .05*np.exp(-self.straight_cost) + .05*np.exp(-self.yvel_cost) \
                + .15*np.exp(-self.l_foot_cost_forcevel) + .15*np.exp(-self.r_foot_cost_forcevel) \
                + .1*np.exp(-self.l_foot_cost_pos) + .1*np.exp(-self.r_foot_cost_pos) \
                + .05*np.exp(-self.l_foot_orient) + .05*np.exp(-self.r_foot_orient) \
                + .1*np.exp(-self.tray_box_cost)

    return reward

def stand_up_pole_reward(self):

    reward = 0.2*np.exp(-self.forward_cost) + 0.2*np.exp(-self.com_height) \
                + 0.2*np.exp(-self.pole_pos_cost) + 0.2*np.exp(-self.pole_vel_cost) \
                + .05*np.exp(-self.pel_transacc) + .05*np.exp(-self.pel_rotacc) \
                + .05*np.exp(-self.act_cost) + 0.05*np.exp(-self.torque_penalty)

    return reward

def stand_up_pole_free_reward(self):

    reward = 0.2*np.exp(-self.com_height) + 0.2*np.exp(-self.orient_cost) \
                + 0.4*np.exp(-self.pole_pos_cost) + 0.2*np.exp(-self.pole_vel_cost) \

    return reward

def walk_pole_nopelorient_reward(self):
    reward = .1*np.exp(-self.forward_cost) \
                + .05*np.exp(-self.straight_cost) + .05*np.exp(-self.yvel_cost) \
                + .075*np.exp(-self.l_foot_cost_forcevel) + .075*np.exp(-self.r_foot_cost_forcevel) \
                + .075*np.exp(-self.l_foot_cost_pos) + .075*np.exp(-self.r_foot_cost_pos) \
                + .05*np.exp(-self.l_foot_orient) + .05*np.exp(-self.r_foot_orient) \
                + .05*np.exp(-self.hiproll_cost) + .05*np.exp(-self.hipyaw_vel) \
                + .05*np.exp(-self.act_cost) + 0.05*np.exp(-self.torque_penalty) \
                + .1*np.exp(-self.pole_pos_cost) + .1*np.exp(-self.pole_vel_cost)

    return reward

def walk_pole_reward(self):
    reward = .1*np.exp(-self.forward_cost) + 0.05*np.exp(-self.orient_pitch) \
                + .05*np.exp(-self.yvel_cost) \
                + .075*np.exp(-self.l_foot_cost_forcevel) + .075*np.exp(-self.r_foot_cost_forcevel) \
                + .075*np.exp(-self.l_foot_cost_pos) + .075*np.exp(-self.r_foot_cost_pos) \
                + .05*np.exp(-self.l_foot_orient) + .05*np.exp(-self.r_foot_orient) \
                + .05*np.exp(-self.hiproll_cost) + .05*np.exp(-self.hipyaw_vel) \
                + .05*np.exp(-self.act_cost) + 0.05*np.exp(-self.torque_penalty) \
                + .1*np.exp(-self.pole_pos_cost) + .1*np.exp(-self.pole_vel_cost)

    return reward