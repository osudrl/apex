import torch
import numpy as np
from rl.policies.recurrent import RecurrentActor

x = torch.ones(1, 4)
actor = RecurrentActor(4, 3)

for i in range(5):
  actor.forward(x)
  input()
