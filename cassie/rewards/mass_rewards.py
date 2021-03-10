import numpy as np
from ..quaternion_function import *

def tray_box_reward(self):

    reward = .2*np.exp(-self.forward_cost) + .1*np.exp(-self.orient_cost) \
                + .05*np.exp(-self.straight_cost) + .05*np.exp(-self.yvel_cost) \
                + .05*np.exp(-self.l_foot_cost_forcevel) + .05*np.exp(-self.r_foot_cost_forcevel) \
                + .05*np.exp(-self.l_foot_cost_pos) + .05*np.exp(-self.r_foot_cost_pos) \
                + .05*np.exp(-self.l_foot_orient) + .05*np.exp(-self.r_foot_orient) \
                + .05*np.exp(-self.hiproll_cost) + .05*np.exp(-self.hipyaw_vel) \
                + .025*np.exp(-self.pel_transacc) + .025*np.exp(-self.pel_rotacc) \
                + .025*np.exp(-self.act_cost) + 0.025*np.exp(-self.torque_penalty) \
                + .1*np.exp(-self.tray_box_cost)

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