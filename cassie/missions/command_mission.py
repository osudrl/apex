import pickle
import numpy as np
import os

class CommandTrajectory:
    def __init__(self, mission_name):
        mission_path = os.path.join(mission_name, "command_trajectory.pkl")
        with open(mission_path, "rb") as f:
            trajectory = pickle.load(f)

        self.global_pos = np.copy(trajectory["compos"])
        self.speed_cmd = np.copy(trajectory["speed"])
        
        # NOTE: still need to rotate translational velocity and accleration
        self.orient = np.copy(trajectory["orient"])
        self.prev_orient = 0
        
        self.trajlen = len(self.speed_cmd)

        # print("positions:\n{}\n\nvelocities:\n{}\n\norient:\n{}\n".format(self.global_pos[:5], self.speed_cmd[:5], self.orient[:5]))
        # print(self.speed_cmd.shape)
        # print(self.orient.shape)
        # print(np.max(self.speed_cmd))
        # input()