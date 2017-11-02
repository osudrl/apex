import torch
from torch.autograd import Variable
import time


def renderpolicy(env, policy, trj_len, explore=True, speedup=1, dt=0.05):
    R = 0
    # the ravel()[None, :] is because for some envs reset returns shape (N)
    # but step returns shape (N,1), leading to dimension problems
    obs = env.reset().ravel()[None, :]
    for t in range(trj_len):
        obs_var = Variable(torch.Tensor(obs), volatile=True)

        _, action = policy.act(obs_var, explore)

        obs, reward, done, _ = env.step(action.data.numpy().ravel())
        obs = obs.ravel()[None, :]

        R +=  reward

        if done:
            break

        env.render()
        time.sleep(dt / speedup)


def renderloop(env, policy, trj_len, explore=False, speedup=1):
    while True:
        renderpolicy(env, policy, trj_len, explore, speedup)
