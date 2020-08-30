import numpy as np

def smooth_square_wave(x, coefficient, ratio, alpha=0.05):
    clock = coefficient * np.sin(np.pi * 1/ratio * x)
    out = (1 / np.arctan(1 / np.exp(-5 * alpha))) * np.arctan(clock / np.exp(-5 * alpha))
    return out

def periodic_func(x, coefficients, ratios, ratio_shift=0, alpha=1):
    # piecewise function based on ratios.
    # for loop used because we don't know how many phases are in coefficeints or ratios
    # in practice this shouldn't slow down too much cause there aren't that many phases in period
    x = np.fmod(x + ratio_shift, sum(ratios))
    phase_idx = 0
    last_phase_x = 0
    next_phase_time = ratios[phase_idx]
    for coeff, ratio in zip(coefficients, ratios):
        if x <= next_phase_time:
            # do something
            return smooth_square_wave(x - last_phase_x, coeff, ratio, alpha)
        else:
            last_phase_x = next_phase_time
            phase_idx += 1
            next_phase_time += ratios[phase_idx]
periodic_func = np.vectorize(periodic_func, excluded=(1,2,3,4), doc='Vectorized `periodic_func`')
