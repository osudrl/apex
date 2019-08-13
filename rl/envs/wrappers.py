import numpy as np
import torch

# Gives a vectorized interface to a single environment
class WrapEnv:
    def __init__(self, env_fn):
        self.env = env_fn()

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def step(self, action):
        state, reward, done, info = self.env.step(action[0])
        return np.array([state]), np.array([reward]), np.array([done]), np.array([info])

    def render(self):
        self.env.render()

    def reset(self):
        return np.array([self.env.reset()])

# TODO: this is probably a better case for inheritance than for a wrapper
# Gives an interface to exploit mirror symmetry
class SymmetricEnv:    
    def __init__(self, env_fn, mirrored_obs=None, mirrored_act=None, obs_fn=None, act_fn=None):

        assert (bool(mirrored_act) ^ bool(act_fn)) and (bool(mirrored_obs) ^ bool(obs_fn)), \
            "You must provide either mirror indices or a mirror function, but not both, for \
             observation and action."

        if mirrored_act:
            self.act_mirror_matrix = torch.Tensor(_get_symmetry_matrix(mirrored_act))

        elif act_fn:
            assert callable(act_fn), "Action mirror function must be callable"
            self.mirror_action = act_fn

        if mirrored_obs:
            self.obs_mirror_matrix = torch.Tensor(_get_symmetry_matrix(mirrored_obs))

        elif obs_fn:
            assert callable(obs_fn), "Observation mirror function must be callable"
            self.mirror_observation = obs_fn

        self.env = env_fn()

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def mirror_action(self, action):
        return action @ self.act_mirror_matrix

    def mirror_observation(self, obs):
        return obs @ self.obs_mirror_matrix

    # To be used when there is a clock in the observation. In this case, the mirrored_obs vector inputted
    # when the SymmeticEnv is created should not move the clock input order. The indices of the obs vector
    # where the clocks are located need to be inputted.
    def mirror_clock_observation(self, obs, clock_inds):
        mirror_obs = obs @ self.obs_mirror_matrix
        clock = mirror_obs[:, self.clock_inds]
        # print("clock: ", clock)
        for i in range(np.shape(clock)[1]):
            mirror_obs[:, clock_inds[i]] = np.sin(np.arcsin(clock[:, i]) + np.pi)
        return mirror_obs


def _get_symmetry_matrix(mirrored):
    numel = len(mirrored)
    mat = np.zeros((numel, numel))

    for (i, j) in zip(np.arange(numel), np.abs(mirrored)):
        mat[i, j] = np.sign(mirrored[i])

    return mat