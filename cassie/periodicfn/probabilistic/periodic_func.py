import numpy as np
from collections import namedtuple
from scipy.stats import norm, vonmises

Phase = namedtuple('Phase', ['start', 'end', 'std', 'coeff'])

def vonmises_func(x, behavior, shift=0):

    out = 0
    x   = (x + shift) * 2 * np.pi
    for phase in behavior:
        x_0 = phase.start * 2 * np.pi
        x_1 = phase.end   * 2 * np.pi
        std = phase.std
        c   = phase.coeff

        kappa = 1 / (std ** 2)

        p1 = vonmises.cdf(x, kappa=kappa, loc=x_0, scale=1.0)
        p2 = vonmises.cdf(x, kappa=kappa, loc=x_1, scale=1.0)
        out += c * (p1 - p2)
    return out

def shift_behavior(behavior, shift=0.5, xlim=1.0):
    shifted = []
    # print(f"{behavior}")
    # use fmod to shift each phase in behavior
    for phase in behavior:
        # new_start = np.fmod(phase.start+shift, xlim)
        # new_end = np.fmod(phase.end+shift, xlim)
        new_start = phase.start+shift
        new_end = phase.end+shift
        shifted.append(Phase(start=new_start, end=new_end, std=phase.std, coeff=phase.coeff))
    # print(f"{shifted}")
    return shifted

def _repeat_behavior(behavior, xlim=1.0):
    repeated_behavior = []
    for phase in behavior:
        repeated_behavior.append(Phase(start=phase.start-xlim, end=phase.end-xlim, std=phase.std, coeff=phase.coeff))
        repeated_behavior.append(Phase(start=phase.start+xlim, end=phase.end+xlim, std=phase.std, coeff=phase.coeff))
        repeated_behavior.append(phase)
    return repeated_behavior

def probabilistic_periodic_func(x, behavior, xlim=1.0):
    repeated_behavior = _repeat_behavior(behavior, xlim=xlim)
    # ASSUMPTION: times in behavior are monotonically increasing from 0.0 to 1.0
    out = 0.0
    for phase in repeated_behavior:
        if phase.start < phase.end:
            past_start = norm.cdf(x, loc=phase.start, scale=phase.std**2)
            before_end = 1 - norm.cdf(x, loc=phase.end, scale=phase.std**2)
            out += phase.coeff * (past_start * before_end)
        else:
            past_start = norm.cdf(x, loc=phase.start, scale=phase.std**2)
            before_end = 1 - norm.cdf(x, loc=phase.end, scale=phase.std**2)
            past_begin = norm.cdf(x, loc=0.0, scale=phase.std**2)
            before_limit = 1 - norm.cdf(x, loc=xlim, scale=phase.std**2)
            out += phase.coeff * (past_start * before_limit + past_begin * before_end)
    return out

# THIS IS VERY SLOW BUT IT SUPPORTS WRAPPING
def probabilistic_periodic_func_single(x, coefficients, ratios, ratio_shift=0, std=0.01):
    # piecewise function based on ratios.
    # for loop used because we don't know how many phases are in coefficeints or ratios
    # in practice this shouldn't slow down too much cause there aren't that many phases in period
    x = np.fmod(x + ratio_shift, sum(ratios))
    phase_idx = 0
    last_phase_x = 0
    next_phase_time = ratios[phase_idx]
    for coeff, ratio in zip(coefficients, ratios):
        if x <= next_phase_time:
            return coeff * (norm.cdf(x, loc=last_phase_x, scale=std**2) - norm.cdf(x, loc=next_phase_time, scale=std**2))
        else:
            last_phase_x = next_phase_time
            phase_idx += 1
            next_phase_time += ratios[phase_idx]
probabilistic_periodic_func_single = np.vectorize(probabilistic_periodic_func_single, excluded=(1,2,3,4), doc='Vectorized `probabilistic_periodic_func_single`')
