import numpy as np

def side_speedmatch_torquesmooth_reward(self):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())
    
    forward_diff = np.abs(qvel[0] -self.speed)
    orient_diff = np.linalg.norm(qpos[3:7] - np.array([1, 0, 0, 0]))
    side_diff = np.abs(qvel[1] - self.side_speed)
    if forward_diff < 0.05:
        forward_diff = 0
    if side_diff < 0.05:
        side_diff = 0

    reward = .25*np.exp(-forward_diff) + .25*np.exp(-side_diff) + .2*np.exp(-orient_diff) \
                    + .1*np.exp(-self.torque_cost) + .2*np.exp(-self.smooth_cost)
    return reward