import numpy as np
import pickle, os

"""
Aslip-IK trajectories, for several speeds

- cpos (com position) is increasing in x and slightly oscillating in y and z.
- correct use of this is calculating the difference between current robot pelvis and com offset by last robot position,
and using that as the input to the network.
- this should prevent the com x input from being used like a clock.
- get_ref_aslip_ext_state below
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

# delta position : difference between desired taskspace position and current position.
def get_ref_aslip_ext_state(self, current_state, last_compos, phase=None, offset=None):

    current_compos = np.array(current_state.pelvis.position[:])
    current_compos[2] -= current_state.terrain.height
    current_lpos = np.array(current_state.leftFoot.position[:])
    current_rpos = np.array(current_state.rightFoot.position[:])

    # compos is global, while lpos and rpos are relative to compos. So only need to adjust compos
    current_compos = current_compos - last_compos

    if phase is None:
        phase = self.phase

    if phase > self.phaselen:
        phase = 0

    phase = int(phase)

    rpos = np.copy(self.trajectory.rpos[phase])
    rvel = np.copy(self.trajectory.rvel[phase])
    lpos = np.copy(self.trajectory.lpos[phase])
    lvel = np.copy(self.trajectory.lvel[phase])
    cpos = np.copy(self.trajectory.cpos[phase])
    cvel = np.copy(self.trajectory.cvel[phase])

    # Manual z offset to get taller walking
    if offset is not None:
        cpos[2] += offset
        # need to update these because they 
        lpos[2] -= offset
        rpos[2] -= offset
    
    cpos = cpos - current_compos
    lpos = lpos - current_lpos
    rpos = rpos - current_rpos

    return rpos, rvel, lpos, lvel, cpos, cvel

def get_ref_aslip_unaltered_state(self, phase=None, offset=None):

    if phase is None:
        phase = self.phase

    if phase > self.phaselen:
        phase = 0

    phase = int(phase)

    rpos = np.copy(self.trajectory.rpos[phase])
    rvel = np.copy(self.trajectory.rvel[phase])
    lpos = np.copy(self.trajectory.lpos[phase])
    lvel = np.copy(self.trajectory.lvel[phase])
    cpos = np.copy(self.trajectory.cpos[phase])
    cvel = np.copy(self.trajectory.cvel[phase])

    # Manual z offset to get taller walking
    if offset is not None:
        cpos[2] += offset
        # need to update these because they 
        lpos[2] -= offset
        rpos[2] -= offset

    return rpos, rvel, lpos, lvel, cpos, cvel