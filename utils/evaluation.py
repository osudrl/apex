import torch
from torch.autograd import Variable
import time


def renderpolicy(env, policy, trj_len, explore=False, dt=0.05, speedup=1):
    # the ravel()[None, :] is because for some envs reset returns shape (N)
    # but step returns shape (N,1), leading to dimension problems
    obs = env.reset().ravel()[None, :]
    for t in range(trj_len):
        obs_var = Variable(torch.Tensor(obs))
        means, log_stds, stds = policy(obs_var)

        if explore:
            action = policy.get_action(means, stds)
        else:
            action = means.data.numpy()  # don't explore when evaluating

        obs = env.step(action)[0].ravel()[None, :]
        env.render()
        time.sleep(dt / speedup)

def renderloop(env, policy, trj_len, explore=False, speedup=1):
    while True:
        renderpolicy(env, policy, trj_len, explore)
