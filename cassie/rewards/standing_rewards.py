import numpy as np

def stand_reward(self):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())

    com_vel = np.linalg.norm(qvel[0:3])
    com_height = (0.9 - qpos[2]) ** 2

    reward = 0.5*np.exp(-com_vel) + 0.5*np.exp(-com_height)

    return reward

def step_even_reward(self):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())

    com_vel = np.linalg.norm(qvel[0:3])
    com_height = (0.9 - qpos[2]) ** 2

    reward = 0.2*np.exp(-com_vel) + 0.2*np.exp(-com_height) \
            + 0.3*np.exp(-self.l_foot_cost_even) + 0.3*np.exp(-self.r_foot_cost_even)

    return reward

def step_even_pelheight_reward(self):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())

    com_height = (0.9 - qpos[2]) ** 2
    if qpos[2] > 0.8:
        com_height = 0

    reward = 0.2*np.exp(-com_height) \
            + 0.4*np.exp(-self.l_foot_cost_even) + 0.4*np.exp(-self.r_foot_cost_even)

    return reward

def step_smooth_pelheight_reward(self):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())

    com_height = (0.9 - qpos[2]) ** 2
    if qpos[2] > 0.8:
        com_height = 0

    reward = 0.2*np.exp(-com_height) \
            + 0.4*np.exp(-self.l_foot_cost_smooth) + 0.4*np.exp(-self.r_foot_cost_smooth)

    return reward

def stand_up_pole_reward(self):

    reward = 0.2*np.exp(-self.forward_cost) + 0.2*np.exp(-self.com_height) \
                + 0.2*np.exp(-self.pole_pos_cost) + 0.2*np.exp(-self.pole_vel_cost) \
                + .05*np.exp(-self.pel_transacc) + .05*np.exp(-self.pel_rotacc) \
                + .05*np.exp(-self.act_cost) + 0.05*np.exp(-self.torque_penalty)

    return reward

def stand_up_pole_free_reward(self):

    reward = 0.2*np.exp(-self.com_height) \
                + 0.5*np.exp(-self.pole_pos_cost) + 0.3*np.exp(-self.pole_vel_cost) \

    return reward

def stand_smooth_reward(self):

    reward = 0.2*np.exp(-self.forward_cost) + 0.6*np.exp(-self.com_height) \
                + .05*np.exp(-self.pel_transacc) + .05*np.exp(-self.pel_rotacc) \
                + .05*np.exp(-self.act_cost) + 0.05*np.exp(-self.torque_penalty)

    return reward

def stand_smooth_footorient_reward(self):

    reward = 0.2*np.exp(-self.forward_cost) +  0.2*np.exp(-self.yvel_cost) \
                + 0.2*np.exp(-self.com_height) + .1*np.exp(-self.orient_cost) \
                + .05*np.exp(-self.l_foot_orient) + .05*np.exp(-self.r_foot_orient) \
                + .05*np.exp(-self.pel_transacc) + .05*np.exp(-self.pel_rotacc) \
                + .05*np.exp(-self.act_cost) + 0.05*np.exp(-self.torque_penalty)

    return reward

def stand_smooth_footorient_motorvel_reward(self):

    reward = 0.15*np.exp(-self.forward_cost) +  0.15*np.exp(-self.yvel_cost) \
                + 0.2*np.exp(-self.com_height) + .1*np.exp(-self.orient_cost) \
                + .05*np.exp(-self.l_foot_orient) + .05*np.exp(-self.r_foot_orient) \
                + .05*np.exp(-self.pel_transacc) + .05*np.exp(-self.pel_rotacc) \
                + .05*np.exp(-self.act_cost) + 0.05*np.exp(-self.torque_penalty) \
                + 0.1*np.exp(-self.motor_vel_cost)

    return reward

def stand_sprint_reward(self):
    if self.sprint == 0:
        reward = 0.2*np.exp(-self.forward_cost) +  0.2*np.exp(-self.yvel_cost) \
                + 0.2*np.exp(-self.com_height) + .1*np.exp(-self.orient_cost) \
                + .05*np.exp(-self.l_foot_orient) + .05*np.exp(-self.r_foot_orient) \
                + .05*np.exp(-self.pel_transacc) + .05*np.exp(-self.pel_rotacc) \
                + .05*np.exp(-self.act_cost) + 0.05*np.exp(-self.torque_penalty)
    elif self.sprint == 1:
        reward = .2*np.exp(-self.forward_cost) + .1*np.exp(-self.orient_cost) \
                + .05*np.exp(-self.straight_cost) + .05*np.exp(-self.yvel_cost) \
                + .10*np.exp(-self.l_foot_cost_forcevel) + .10*np.exp(-self.r_foot_cost_forcevel) \
                + .05*np.exp(-self.l_foot_cost_pos) + .05*np.exp(-self.r_foot_cost_pos) \
                + .05*np.exp(-self.l_foot_orient) + .05*np.exp(-self.r_foot_orient) \
                + .05*np.exp(-self.hiproll_cost) + .05*np.exp(-self.hipyaw_vel) \
                + .025*np.exp(-self.pel_transacc) + .025*np.exp(-self.pel_rotacc) \
                + .025*np.exp(-self.act_cost) + 0.025*np.exp(-self.torque_penalty)
    
    return reward