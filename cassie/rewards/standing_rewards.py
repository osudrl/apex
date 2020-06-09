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