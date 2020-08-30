import numpy as np
from collections import OrderedDict

try:
    from .periodic_func import periodic_func
except ImportError:
    from periodic_func import periodic_func

import matplotlib.pyplot as plt

SAMPLE_FREQ = 200
PYPLOT_IMPORTED = False

class PeriodicBehavior:
    def __init__(self, phase_coefficients, phase_ratios, ratio_shift, period_time, alpha=0+1e-10, name='undefined gait'):
        # if not PYPLOT_IMPORTED:
        #     import matplotlib.pyplot as plt
        #     PYPLOT_IMPORTED = True

        self.phase_coefficients = phase_coefficients
        self.phase_ratios = phase_ratios
        self.ratio_shift = ratio_shift
        self.phaselen = len(self.phase_coefficients)
        self.period_time = period_time
        self.alpha = alpha
        self.name = name
        assert(self.ratio_shift < 1.0)
        assert(len(self.phase_coefficients) == len(self.phase_ratios))
        assert(self.alpha != 0)

        self.xlim = sum(self.phase_ratios)

        # domain
        self.x = np.linspace(0, self.xlim, num=SAMPLE_FREQ)

        # for saving plot png
        import os
        if not os.path.exists('example_gaits'):
            os.makedirs('example_gaits')

    def plot(self):

        self.fig, self.axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(10,5))
        self.axs[0].set_title(f'{self.name}')

        left_foot = periodic_func(self.x, self.phase_coefficients, self.phase_ratios, ratio_shift=0, alpha=self.alpha)
        right_foot = periodic_func(self.x, self.phase_coefficients, self.phase_ratios, ratio_shift=self.ratio_shift, alpha=self.alpha)

        self.axs[0].plot(self.x * self.period_time, left_foot)
        self.axs[1].plot(self.x * self.period_time, right_foot)

        left_swing_ranges = self._find_overlapping_regions((left_foot > 0) & (right_foot < 0))
        right_swing_ranges = self._find_overlapping_regions((left_foot < 0) & (right_foot > 0))
        aerial_ranges = self._find_overlapping_regions((left_foot > 0) & (right_foot > 0))
        stance_ranges = self._find_overlapping_regions((left_foot < 0) & (right_foot < 0))

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

class LivePlot:
    def __init__(self, wrap=False):

        self.wrap_viz = wrap

    def __call__(self, pipe):
        print('starting plotter...')

        self.pipe = pipe
        self.fig, self.axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(10,5))
        self.axs[0].set_title('Live PeriodicFcn Plot')
        self.axs[0].set_ylabel("left")
        self.axs[1].set_ylabel("right")
        
        # domain
        self.ratios = [0.5, 0.5]
        self.x = np.linspace(0, sum(self.ratios), num=SAMPLE_FREQ)

        # first draw
        left_foot = periodic_func(self.x, [-1, 1], [0.5, 0.5], ratio_shift=0.0, alpha=0.01)
        right_foot = periodic_func(self.x, [-1, 1], [0.5, 0.5], ratio_shift=0.5, alpha=0.01)
        self._draw(left_foot, right_foot)

        timer = self.fig.canvas.new_timer(interval=10)
        timer.add_callback(self.call_back)
        timer.start()

        print('...done')
        plt.show()

    def terminate(self):
        plt.close('all')

    def call_back(self):
        while self.pipe.poll():
            command = self.pipe.recv()
            if command is None:
                self.terminate()
                return False
            else:
                coeffs, self.ratios, ratio_shift, alpha = command
                left_foot = periodic_func(self.x, coeffs, self.ratios, ratio_shift=0.0, alpha=alpha)
                right_foot = periodic_func(self.x, coeffs, self.ratios, ratio_shift=ratio_shift, alpha=alpha)
                self._redraw(left_foot, right_foot)
        self.fig.canvas.draw()
        return True

    def _draw(self, left_foot, right_foot):
        self.x = np.linspace(0, sum(self.ratios), num=SAMPLE_FREQ)
        self.l0, = self.axs[0].plot(self.x, left_foot)
        self.l1, = self.axs[1].plot(self.x, right_foot)

    def _redraw(self, left_foot, right_foot):
        # get right foot force / vel data
        self.x = np.linspace(0, sum(self.ratios), num=SAMPLE_FREQ)
        self.l0.set_xdata(self.x)
        self.l1.set_xdata(self.x)
        self.l0.set_ydata(left_foot)
        self.l1.set_ydata(right_foot)

if __name__ == '__main__':

    # TODO: Get mirroring working so gait descriptions can be greatly simplified
    S = 1   # swing
    G = -1  # stabce
    hopping = PeriodicBehavior(
        phase_coefficients=[G, S],
        phase_ratios=[0.5, 0.5],
        ratio_shift=0.0,
        period_time=1.0,
        alpha=0.01,
        name='hopping'
    )
    walking = PeriodicBehavior(
        phase_coefficients=[S, G],
        phase_ratios=[0.4, 0.6],
        ratio_shift=0.5,
        period_time=1.0,
        alpha=0.01,
        name='walking'
    )
    running = PeriodicBehavior(
        phase_coefficients=[G, S],
        phase_ratios=[0.4, 0.6],
        ratio_shift=0.5,
        period_time=1.0,
        alpha=0.01,
        name='running'
    )
    skipping = PeriodicBehavior(
        phase_coefficients=[G, S, G, S],
        phase_ratios=[0.2, 0.4, 0.2, 0.2],
        ratio_shift=0.5,
        period_time=1.0,
        alpha=0.01,
        name='skipping'
    )
    hopping.plot()
    walking.plot()
    running.plot()
    skipping.plot()
    plt.show()
