import numpy as np
from ..quaternion_function import *



def speedmatchavg_footvarclock_footorient_stablepel_hiprollyawvel_smoothact_torquecost_reward(self):

    reward = .2*np.exp(-self.forward_cost) + .1*np.exp(-self.orient_cost) \
                + .05*np.exp(-self.straight_cost) + .05*np.exp(-self.yvel_cost) \
                + .15*np.exp(-self.l_foot_cost_var) + .15*np.exp(-self.r_foot_cost_var) \
                + .05*np.exp(-self.l_foot_orient) + .05*np.exp(-self.r_foot_orient) \
                + .05*np.exp(-self.hiproll_cost) + .05*np.exp(-self.hipyaw_vel) \
                + .025*np.exp(-self.pel_transacc) + .025*np.exp(-self.pel_rotacc) \
                + .025*np.exp(-self.act_cost) + 0.025*np.exp(-self.torque_penalty)

    return reward

def speedmatchavg_footforce_footspeedpos_footorient_stablepel_hiprollyawvel_smoothact_torquecost_reward(self):

    reward = .2*np.exp(-self.forward_cost) + .1*np.exp(-self.orient_cost) \
                + .05*np.exp(-self.straight_cost) + .05*np.exp(-self.yvel_cost) \
                + .10*np.exp(-self.l_foot_cost_speedpos) + .10*np.exp(-self.r_foot_cost_speedpos) \
                + .05*np.exp(-self.l_foot_cost_force) + .05*np.exp(-self.r_foot_cost_force) \
                + .05*np.exp(-self.l_foot_orient) + .05*np.exp(-self.r_foot_orient) \
                + .05*np.exp(-self.hiproll_cost) + .05*np.exp(-self.hipyaw_vel) \
                + .025*np.exp(-self.pel_transacc) + .025*np.exp(-self.pel_rotacc) \
                + .025*np.exp(-self.act_cost) + 0.025*np.exp(-self.torque_penalty)

    return reward

def speedmatchavg_forcevel_footpos_footorient_stablepel_hiprollyawvel_smoothact_torquecost_reward(self):

    reward = .1*np.exp(-self.forward_cost) + .1*np.exp(-self.orient_cost) \
                + .05*np.exp(-self.straight_cost) + .05*np.exp(-self.yvel_cost) \
                + .10*np.exp(-self.l_foot_cost_forcevel) + .10*np.exp(-self.r_foot_cost_forcevel) \
                + .1*np.exp(-self.l_foot_cost_pos) + .1*np.exp(-self.r_foot_cost_pos) \
                + .05*np.exp(-self.l_foot_orient) + .05*np.exp(-self.r_foot_orient) \
                + .05*np.exp(-self.hiproll_cost) + .05*np.exp(-self.hipyaw_vel) \
                + .025*np.exp(-self.pel_transacc) + .025*np.exp(-self.pel_rotacc) \
                + .025*np.exp(-self.act_cost) + 0.025*np.exp(-self.torque_penalty)

    return reward

def run_nopeltransacc_reward(self):

    reward = .1*np.exp(-self.forward_cost) + .1*np.exp(-self.orient_cost) \
                + .05*np.exp(-self.straight_cost) + .05*np.exp(-self.yvel_cost) \
                + .10*np.exp(-self.l_foot_cost_forcevel) + .10*np.exp(-self.r_foot_cost_forcevel) \
                + .1*np.exp(-self.l_foot_cost_pos) + .1*np.exp(-self.r_foot_cost_pos) \
                + .05*np.exp(-self.l_foot_orient) + .05*np.exp(-self.r_foot_orient) \
                + .05*np.exp(-self.hiproll_cost) + .05*np.exp(-self.hipyaw_vel) \
                + .025*np.exp(-self.pel_rotacc) \
                + .025*np.exp(-self.act_cost) + 0.05*np.exp(-self.torque_penalty)

    return reward

def speedmatch_foothop_reward(self):

    reward = .2*np.exp(-self.forward_cost) + .1*np.exp(-self.orient_cost) \
                + .05*np.exp(-self.straight_cost) + .05*np.exp(-self.yvel_cost) \
                + .15*np.exp(-self.l_foot_cost_hop) + .15*np.exp(-self.r_foot_cost_hop) \
                + .05*np.exp(-self.l_foot_orient) + .05*np.exp(-self.r_foot_orient) \
                + .05*np.exp(-self.hiproll_cost) + .05*np.exp(-self.hipyaw_vel) \
                + .025*np.exp(-self.pel_transacc) + .025*np.exp(-self.pel_rotacc) \
                + .025*np.exp(-self.act_cost) + 0.025*np.exp(-self.torque_penalty)

    return reward

def speedmatchavg_orientchange_forcevel_footpos_footorient_stablepel_hiprollyawvel_smoothact_torquecost_reward(self):

    reward = .10*np.exp(-self.forward_cost) + .1*np.exp(-self.orient_cost) \
                + .10*np.exp(-self.yvel_cost) \
                + .10*np.exp(-self.l_foot_cost_forcevel) + .10*np.exp(-self.r_foot_cost_forcevel) \
                + .1*np.exp(-self.l_foot_cost_pos) + .1*np.exp(-self.r_foot_cost_pos) \
                + .05*np.exp(-self.l_foot_orient) + .05*np.exp(-self.r_foot_orient) \
                + .05*np.exp(-self.hiproll_cost) + .05*np.exp(-self.hipyaw_vel) \
                + .025*np.exp(-self.pel_transacc) + .025*np.exp(-self.pel_rotacc) \
                + .025*np.exp(-self.act_cost) + 0.025*np.exp(-self.torque_penalty)

    return reward

def speedmatchavg_orientchange_forcevel_footpos_footorient_stablepel_hiprollyawvel_smoothact_torquecost_footydist_reward(self):

    reward = .10*np.exp(-self.forward_cost) + .1*np.exp(-self.orient_cost) \
                + .10*np.exp(-self.yvel_cost) \
                + .10*np.exp(-self.l_foot_cost_forcevel) + .10*np.exp(-self.r_foot_cost_forcevel) \
                + .1*np.exp(-self.l_foot_cost_pos) + .1*np.exp(-self.r_foot_cost_pos) \
                + .05*np.exp(-self.l_foot_orient) + .05*np.exp(-self.r_foot_orient) \
                + .025*np.exp(-self.hiproll_cost) + .025*np.exp(-self.hipyaw_vel) \
                + .025*np.exp(-self.pel_transacc) + .025*np.exp(-self.pel_rotacc) \
                + .025*np.exp(-self.act_cost) + 0.025*np.exp(-self.torque_penalty) \
                + .05*np.exp(-self.footydist_cost)

    return reward

def speedmatchavg_forcevel_footpos_footorient_stablepel_hiprollyawvel_smoothact_torquecost_traybox_reward(self):

    reward = .2*np.exp(-self.forward_cost) + .1*np.exp(-self.orient_cost) \
                + .05*np.exp(-self.straight_cost) + .05*np.exp(-self.yvel_cost) \
                + .10*np.exp(-self.l_foot_cost_forcevel) + .10*np.exp(-self.r_foot_cost_forcevel) \
                + .05*np.exp(-self.l_foot_cost_pos) + .05*np.exp(-self.r_foot_cost_pos) \
                + .05*np.exp(-self.l_foot_orient) + .05*np.exp(-self.r_foot_orient) \
                + .05*np.exp(-self.hiproll_cost) + .05*np.exp(-self.hipyaw_vel) \
                + .025*np.exp(-self.pel_transacc) + .025*np.exp(-self.pel_rotacc) \
                + .025*np.exp(-self.act_cost) + 0.025*np.exp(-self.torque_penalty)

    qpos = self.sim.qpos_full()
    if qpos[-1] <= qpos[2]: # Get no reward if box below pelvis height (box fell off)
        reward = 0

    return reward


def speedmatchavg_footlinclock_footorient_stablepel_smoothact_torquecost_reward(self):

    reward = .2*np.exp(-self.forward_cost) + .1*np.exp(-self.orient_cost) \
                + .1*np.exp(-self.straight_cost) + .05*np.exp(-self.yvel_cost) \
                + .15*np.exp(-self.l_foot_cost_linclock) + .15*np.exp(-self.r_foot_cost_linclock) \
                + .05*np.exp(-self.l_foot_orient) + .05*np.exp(-self.r_foot_orient) \
                + .025*np.exp(-self.pel_transacc) + .025*np.exp(-self.pel_rotacc) \
                + .05*np.exp(-self.act_cost) + 0.05*np.exp(-self.torque_penalty)

    return reward

def speedmatchavg_footlinclock_footorient_stablepel_orientrollyaw_smoothact_torquecost_reward(self):

    reward = .2*np.exp(-self.forward_cost) + .1*np.exp(-self.orient_rollyaw_cost) \
                + .1*np.exp(-self.straight_cost) + .05*np.exp(-self.yvel_cost) \
                + .15*np.exp(-self.l_foot_cost_linclock) + .15*np.exp(-self.r_foot_cost_linclock) \
                + .05*np.exp(-self.l_foot_orient) + .05*np.exp(-self.r_foot_orient) \
                + .025*np.exp(-self.pel_transacc) + .025*np.exp(-self.pel_rotacc) \
                + .05*np.exp(-self.act_cost) + 0.05*np.exp(-self.torque_penalty)

    return reward

def speedmatchavg_footlinclock_footorient_stablepel_hiprollyawvel_smoothact_torquecost_reward(self):

    reward = .2*np.exp(-self.forward_cost) + .1*np.exp(-self.orient_cost) \
                + .05*np.exp(-self.straight_cost) + .05*np.exp(-self.yvel_cost) \
                + .15*np.exp(-self.l_foot_cost_linclock) + .15*np.exp(-self.r_foot_cost_linclock) \
                + .05*np.exp(-self.l_foot_orient) + .05*np.exp(-self.r_foot_orient) \
                + .05*np.exp(-self.hiproll_cost) + .05*np.exp(-self.hipyaw_vel) \
                + .025*np.exp(-self.pel_transacc) + .025*np.exp(-self.pel_rotacc) \
                + .025*np.exp(-self.act_cost) + 0.025*np.exp(-self.torque_penalty)

    return reward

def sidespeedmatchavg_footlinclock_footorient_stablepel_smoothact_torquecost_reward(self):

    reward = .25*np.exp(-self.forward_cost) + .1*np.exp(-self.orient_cost) \
                + .1*np.exp(-self.yvel_cost) \
                + .15*np.exp(-self.l_foot_cost_linclock) + .15*np.exp(-self.r_foot_cost_linclock) \
                + .05*np.exp(-self.l_foot_orient) + .05*np.exp(-self.r_foot_orient) \
                + .025*np.exp(-self.pel_transacc) + .025*np.exp(-self.pel_rotacc) \
                + .05*np.exp(-self.act_cost) + 0.05*np.exp(-self.torque_penalty)

    return reward

def sprint_reward(self):
    reward = .25*self.max_speed_cost + .1*np.exp(-self.orient_cost) \
                + .05*np.exp(-self.straight_cost) + .05*np.exp(-self.yvel_cost) \
                + .15*np.exp(-self.l_foot_cost_forcevel) + .15*np.exp(-self.r_foot_cost_forcevel) \
                + .05*np.exp(-self.l_foot_orient) + .05*np.exp(-self.r_foot_orient) \
                + .025*np.exp(-self.pel_transacc) + .025*np.exp(-self.pel_rotacc) \
                + .05*np.exp(-self.act_cost) + 0.05*np.exp(-self.torque_penalty)
    return reward

def sprint_pure_speed_reward(self):
    reward = .25*self.max_speed_cost / 2 + .1*np.exp(-self.orient_cost) \
                + .1*np.exp(-self.straight_cost) + .05*np.exp(-self.yvel_cost) \
                + .1*np.exp(-self.l_foot_orient) + .1*np.exp(-self.r_foot_orient) \
                + .1*np.exp(-self.pel_rotacc) \
                + .05*np.exp(-self.act_cost) + 0.15*np.exp(-self.torque_penalty)
    return reward

def sprint_pure_speed_hiprollyaw_reward(self):
    reward = .2*self.max_speed_cost / 2 + .1*np.exp(-self.orient_cost) \
                + .1*np.exp(-self.straight_cost) + .05*np.exp(-self.yvel_cost) \
                + .1*np.exp(-self.l_foot_orient) + .1*np.exp(-self.r_foot_orient) \
                + .05*np.exp(-self.hiproll_cost) + .1*np.exp(-self.hipyaw_vel) \
                + .05*np.exp(-self.pel_rotacc) \
                + .05*np.exp(-self.act_cost) + 0.1*np.exp(-self.torque_penalty)
    return reward

def empty_reward(self):
    return 0