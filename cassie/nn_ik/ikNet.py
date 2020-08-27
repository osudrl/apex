import os

import numpy as np

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

class IK_solver:
    """Sets up Neural Network IK solver and a simple interface for calling it.

    Creates an IKNet Pytorch model and loads weights into it. Provides a function for getting qpos given global left foot, right foot, and pelvis xyz coords.

    Attributes:
        model: IKNet PyTorch model

    """

    def __init__(self, path="new_ikNet_state_dict.pt"):
        """Create and load weights into IKNet module.
        
        Args:
            path (str): path to ikNet state_dict (default is "ikNet_state_dict.pt")
        """
        dirname = os.path.dirname(__file__)
        self.model = IKNet(9, 35, (15, 15))
        self.model.load_state_dict(torch.load(os.path.join(dirname, path), map_location=torch.device('cpu')))
        self.model.eval()

    def forward(self, rpos, lpos, cpos):
        """Fetch mujoco qpos given global right foot, left foot, and pelvis xyz positions

        Note:
            rpos and lpos are in global coordinates

        Args:
            rpos (array_like): right foot xyz
            lpos (array_like): left foot xyz
            cpos (array_like): pelvis xyz
        """

        combined = np.concatenate((rpos, lpos, cpos))

        with torch.no_grad():
            qpos = self.model(torch.Tensor(combined)).numpy()

        return qpos

    __call__ = forward

def get_sample_pos():
    """Get reasonable taskspace starting positions for passing through IK Solver.

    Chooses a pelvis position, bounding box, and associated foot positions to create a sample. Validates sample through some naive checks.

    """
    good_selection = False
    while not good_selection:
        # choose pelvis x and y first
        pelvis_x, pelvis_y = np.random.uniform(low=-0.5, high=0.5), np.random.uniform(low=-0.5, high=0.5)
        # next create size of bounding box for foot placements
        bound_x, bound_y = np.random.uniform(low=-0.7/2, high=0.7/2), np.random.uniform(low=0.1, high=0.4)
        # sample left and right feet pos from opposite corners of bounding box
        sample_rpos = np.random.normal(loc=[pelvis_x + bound_x/2, pelvis_y + bound_y/2, 0.01667096],  scale=[4.46822818e-01, 1.66121367e-04, 3.09021099e-02])
        sample_lpos = np.random.normal(loc=[pelvis_x - bound_x/2, pelvis_y - bound_y/2,  0.01667096], scale=[4.46822818e-01, 1.66121367e-04, 3.09021099e-02])
        sample_cpos = np.random.normal(loc=[pelvis_x, pelvis_y, 0.95], scale=[4.46822818e-01, 1.66121367e-04, 0.05])
        # no z values below 0:
        if sample_rpos[2] < 0 or sample_lpos[2] < 0:
            continue
        # no z values above 0.25:
        if sample_rpos[2] > 0.25 or sample_lpos[2] > 0.25:
            continue
        # pelvis x not too far from feet x
        if abs(sample_cpos[0]-sample_rpos[0]) > 0.2 or abs(sample_cpos[0]-sample_lpos[0]) > 0.2:
            continue
        # pelvis z not too low or high
        if sample_cpos[2] < 0.65 or sample_cpos[2] > 1.05:
            continue
        good_selection = True
    return sample_rpos, sample_lpos, sample_cpos

def vectorized_get_sample_pos(size):
    samples = np.array([get_sample_pos() for _ in range(size)])
    return samples[:, 0, :], samples[:, 1, :], samples[:, 2, :]
