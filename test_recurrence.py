import torch
import numpy as np
from rl.policies.recurrent import RecurrentNet

x = torch.ones(1, 4)
actor = RecurrentNet(4, 3)

for i in range(5):
  print(actor.forward(x))
  input()
