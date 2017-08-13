import torch
from torch.autograd import Variable
import time


def renderpolicy(env, policy, trj_len):
    obs = env.reset()
    for t in range(trj_len):
        obs_var = Variable(torch.Tensor(obs).unsqueeze(0))
        means, log_stds, stds = policy(obs_var)

        action = means.data.numpy()  # don't explore when evaluating

        obs = env.step(action)[0]
        env.render()
        time.sleep(1/30)
