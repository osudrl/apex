import numpy as np
import pickle, os

"""
Aslip-IK trajectories, for several speeds
"""
def getAllTrajectories(speeds):
    trajectories = []

    for i, speed in enumerate(speeds):
        dirname = os.path.dirname(__file__)
        traj_path = os.path.join(dirname, "aslipTrajsTaskSpace", "walkCycle_{}.pkl".format(speed))
        trajectories.append(CassieAslipTrajectory(traj_path))

    # print("Got all trajectories")
    return trajectories

class CassieAslipTrajectory:
    def __init__(self, filepath):
        with open(filepath, "rb") as f:
            trajectory = pickle.load(f)

        self.qpos = np.copy(trajectory["qpos"])
        self.length = self.qpos.shape[0]
        self.qvel = np.copy(trajectory["qvel"])
        self.rpos = np.copy(trajectory["rpos"])
        self.rvel = np.copy(trajectory["rvel"])
        self.lpos = np.copy(trajectory["lpos"])
        self.lvel = np.copy(trajectory["lvel"])
        self.cpos = np.copy(trajectory["cpos"])
        self.cvel = np.copy(trajectory["cvel"])