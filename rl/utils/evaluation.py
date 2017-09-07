import torch
from torch.autograd import Variable
import time


def renderpolicy(env, policy, trj_len, explore=True, speedup=1, dt=0.05):
    R = 0
    # the ravel()[None, :] is because for some envs reset returns shape (N)
    # but step returns shape (N,1), leading to dimension problems
    obs = env.reset().ravel()[None, :]
    for t in range(trj_len):
        obs_var = Variable(torch.Tensor(obs))

        action = policy.get_action(obs_var, explore).data.numpy()

        obs, reward, done, _ = env.step(action.ravel())
        obs = obs.ravel()[None, :]

        R +=  reward

        if done:
            break

        env.render()
        time.sleep(dt / speedup)
    #print("\n==================R = %s===============\n" % str(R))
    R = 0

def renderloop(env, policy, trj_len, explore=False, speedup=1):
    while True:
        renderpolicy(env, policy, trj_len, explore, speedup)
