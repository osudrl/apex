import numpy as np

def side_speedmatch_foottraj_reward(self):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())
    
    forward_diff = np.abs(qvel[0] -self.speed)
    orient_diff = np.linalg.norm(qpos[3:7] - np.array([1, 0, 0, 0]))
    side_diff = np.abs(qvel[1] - self.side_speed)
    if forward_diff < 0.05:
        forward_diff = 0
    if side_diff < 0.05:
        side_diff = 0

    reward = .15*np.exp(-forward_diff) + .15*np.exp(-side_diff) + .1*np.exp(-orient_diff) \
                    + .1*np.exp(-20*self.l_foot_diff) + .1*np.exp(-20*self.r_foot_diff) \
                    + .1*np.exp(-5*self.l_footvel_diff) + .1*np.exp(-5*self.r_footvel_diff) \
                    + .1*np.exp(-self.lfoot_orient_cost) + .1*np.exp(-self.rfoot_orient_cost)

    return reward