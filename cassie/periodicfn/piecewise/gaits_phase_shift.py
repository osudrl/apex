import numpy as np

from collections import OrderedDict

SAMPLE_FREQ = 2000

def smooth_square_wave(x, coefficient, ratio, alpha=0.05):
    clock = coefficient * np.sin(np.pi * 1/ratio * x)
    out = (1 / np.arctan(1 / alpha)) * np.arctan(clock / alpha)
    return out

def periodic_func(x, coefficients, ratios, mirror_shift=0, alpha=0.01):
    if mirror_shift != 0:
        coefficients = coefficients[mirror_shift::] + coefficients[:mirror_shift]
        ratios = ratios[mirror_shift::] + ratios[:mirror_shift]
    # piecewise function based on ratios.
    # for loop used because we don't know how many phases are in coefficeints or ratios
    # in practice this shouldn't slow down too much
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

class PeriodicBehavior:
    def __init__(self, phase_coefficients, phase_ratios, shift_for_mirror, period_time, alpha=0+1e-10, name='undefined gait'):
        self.phase_coefficients = phase_coefficients
        self.phase_ratios = phase_ratios
        self.shift_for_mirror = shift_for_mirror
        self.phaselen = len(self.phase_coefficients)
        self.period_time = period_time
        self.alpha = alpha
        self.name = name
        assert(sum(self.phase_ratios) <= 1.0)
        assert(len(self.phase_coefficients) == len(self.phase_ratios))
        assert(self.alpha != 0)

        self.xlim = sum(self.phase_ratios)

        # domain
        self.x = np.linspace(0, self.xlim, num=SAMPLE_FREQ)

    def plot(self):

        self.fig, self.axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(10,5))
        self.axs[0].set_title(f'{self.name}')

        left_foot = periodic_func(self.x, self.phase_coefficients, self.phase_ratios, mirror_shift=0, alpha=self.alpha)
        right_foot = periodic_func(self.x, self.phase_coefficients, self.phase_ratios, mirror_shift=self.shift_for_mirror, alpha=self.alpha)

        self.axs[0].plot(self.x * self.period_time, left_foot)
        self.axs[1].plot(self.x * self.period_time, right_foot)

        left_swing_ranges = self._find_overlapping_regions((left_foot > 0) & (right_foot < 0))
        right_swing_ranges = self._find_overlapping_regions((left_foot < 0) & (right_foot > 0))
        aerial_ranges = self._find_overlapping_regions((left_foot > 0) & (right_foot > 0))
        stance_ranges = self._find_overlapping_regions((left_foot < 0) & (right_foot < 0))
        print(stance_ranges)
        for rng in left_swing_ranges:
            self.axs[2].axvspan(self.x[rng[0]], self.x[rng[1]], alpha=0.5, color='tab:orange', label='left swing')
        for rng in right_swing_ranges:
            self.axs[2].axvspan(self.x[rng[0]], self.x[rng[1]], alpha=0.5, color='tab:red', label='right swing')
        for rng in aerial_ranges:
            self.axs[2].axvspan(self.x[rng[0]], self.x[rng[1]], alpha=0.5, color='tab:blue', label='aerial')
        for rng in stance_ranges:
            self.axs[2].axvspan(self.x[rng[0]], self.x[rng[1]], alpha=0.5, color='tab:green', label='grounded')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.savefig(f'example_gaits/{self.name}.png')

    def _find_overlapping_regions(self, bool_arr):
        i = 0
        start, end = None, None
        in_range = False
        ranges = []
        while i < len(bool_arr):
            # start of new range
            if bool_arr[i] and not in_range:
                start = i
                in_range = True
            # middle of range
            elif bool_arr[i]:
                end = i
            # end of range
            elif in_range:
                ranges.append((start, end))
                start, end = None, None
                in_range = False
            else:
                # print(f'{i} -> {bool_arr[i]}')
                pass
            i += 1
        # finish off unfinished range
        if in_range and end is not None:
            ranges.append((start, end))

        return ranges

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider  # , Button, RadioButtons
    # TODO: Get mirroring working so gait descriptions can be greatly simplified
    S = 1   # swing
    G = -1  # stabce
    hopping = PeriodicBehavior(
        phase_coefficients=[G, S],
        phase_ratios=[0.5, 0.5],
        shift_for_mirror=0,
        period_time=1.0,
        alpha=0.01,
        name='hopping'
    )
    walking = PeriodicBehavior(
        phase_coefficients=[S, G, G, G],
        phase_ratios=[0.4, 0.1, 0.4, 0.1],
        shift_for_mirror=2,
        period_time=1.0,
        alpha=0.01,
        name='walking'
    )
    running = PeriodicBehavior(
        phase_coefficients=[G, S, S, S],
        phase_ratios=[0.4, 0.1, 0.4, 0.1],
        shift_for_mirror=2,
        period_time=1.0,
        alpha=0.01,
        name='running'
    )
    skipping = PeriodicBehavior(
        phase_coefficients=[G, G, S, S, S, G, G, S],
        phase_ratios=[0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.2],
        shift_for_mirror=4,
        period_time=1.0,
        alpha=0.01,
        name='skipping'
    )
    hopping.plot()
    walking.plot()
    running.plot()
    skipping.plot()
    plt.show()
