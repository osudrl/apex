import numpy as np
import pickle, os

import torch
import torch.nn as nn
import torch.functional as F

class IKNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_sizes):
        super(IKNet, self).__init__()

        self.layers = nn.ModuleList()
        
        self.layers += [nn.Linear(input_size, hidden_layer_sizes[0])]
        
        for i in range(len(hidden_layer_sizes)-1):
            self.layers += [nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i+1])]
        
        self.nonlinearity = torch.relu
        
        self.out = nn.Linear(hidden_layer_sizes[-1], output_size)

        # print("# of params: ", sum(p.numel() for p in self.parameters()))
    
    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = self.nonlinearity(layer(x))
        x = self.out(x)
        return x

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

    dirname = os.path.dirname(__file__)

    model = IKNet(9, 35, (15, 15))
    model.load_state_dict(torch.load(os.path.join(dirname, "ikNet_state_dict.pt")))

    for i, speed in enumerate(speeds):
        dirname = os.path.dirname(__file__)
        traj_path = os.path.join(dirname, "aslipTrajsTaskSpace", "walkCycle_{}.pkl".format(speed))
        trajectory = CassieAslipTrajectory(traj_path)
        time = np.linspace(0, trajectory.time[-1], num=50*trajectory.length)
        x = trajectory.pos_f_interp(time).T
        ik_pos = model(torch.Tensor(x)).detach().numpy()
        trajectory.ik_pos = ik_pos
        trajectories.append(trajectory)

    # print("Got all trajectories")
    return trajectories

# # return list of 1-d interp curves parameterized by each trajectory's length
# def aslip_interp(trajectories, simrate):
#     from scipy.interpolate import interp1d

#     trajectory_curves = []
#     for trajectory in trajectories:
#         time_points = np.linspace(0, simrate, num=trajectory.length)
#         time_points = trajectory.time
#         task_trajectory = np.hstack([trajectory.rpos, trajectory.lpos, trajectory.cpos]).T
#         pos_f_interp = interp1d(time_points, task_trajectory, 'linear')
#         pos_f_interp = trajectory.pos_f_interp
#         trajectory_curves.append(pos_f_interp)
#     return trajectory_curves



class CassieAslipTrajectory:
    def __init__(self, filepath):
        with open(filepath, "rb") as f:
            trajectory = pickle.load(f)

        self.qpos = np.copy(trajectory["qpos"])
        self.qvel = np.copy(trajectory["qvel"])
        self.rpos = np.copy(trajectory["rpos"])
        self.rvel = np.copy(trajectory["rvel"])
        self.lpos = np.copy(trajectory["lpos"])
        self.lvel = np.copy(trajectory["lvel"])
        self.cpos = np.copy(trajectory["cpos"])
        self.cvel = np.copy(trajectory["cvel"])
        self.length = self.qpos.shape[0]
        self.time = np.copy(trajectory["time"])
        self.pos_f_interp = trajectory["pos_f_interp"]
        self.ik_pos = None

    def __len__(self):
        return self.length

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

# same as above, but feet are in global coordinates
def get_ref_aslip_global_state(self, phase=None, offset=None):

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

    # x offset for com based on current step count
    cpos[0] += (self.trajectory.cpos[-1, 0] - self.trajectory.cpos[0, 0]) * self.counter

    # Manual z offset to get taller walking
    if offset is not None:
        cpos[2] += offset
        # need to update these because they 
        lpos[2] -= offset
        rpos[2] -= offset

    # Put feet into global coordinates. this also adjusts the x
    lpos += cpos
    rpos += cpos

    return rpos, rvel, lpos, lvel, cpos, cvel

# only alteration to state is adding cpos. no drift correction
def get_ref_aslip_global_state_no_drift_correct(self, phase=None, offset=None):

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

    # Put feet into global coordinates. this also adjusts the x
    lpos += cpos
    rpos += cpos

    return rpos, rvel, lpos, lvel, cpos, cvel