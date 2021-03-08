import numpy as np
from ..quaternion_function import *

def getup_height_reward(self):

    reward = np.exp(-self.com_height)

    return reward

def roll_over_reward(self):

    reward = np.exp(-self.face_up_cost)

    return reward
