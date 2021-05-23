import numpy as np
from ..quaternion_function import *

def speed_torque_reward(self):

    reward = .55*np.exp(-self.forward_cost) + .1*np.exp(-self.orient_cost) \
                + .1*np.exp(-self.straight_cost) + .1*np.exp(-self.yvel_cost) \
                + 0.15*np.exp(-self.torque_penalty)

    return reward

def speed_motoraccel_reward(self):

    reward = .4*np.exp(-self.forward_cost) + .1*np.exp(-self.orient_cost) \
                + .1*np.exp(-self.straight_cost) + .1*np.exp(-self.yvel_cost) \
                + 0.3*np.exp(-self.torque_penalty)

    return reward