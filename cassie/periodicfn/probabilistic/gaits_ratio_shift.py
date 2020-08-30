import numpy as np
from collections import namedtuple, OrderedDict
import matplotlib.pyplot as plt

try:
    from .periodic_func import Phase, probabilistic_periodic_func, shift_behavior
except ImportError:
    from periodic_func import Phase, probabilistic_periodic_func, shift_behavior

SAMPLE_FREQ = 2000

class PeriodicBehavior:
    def __init__(self, phase_coefficients, phase_ratios, phase_stds, ratio_shift, name='undefined gait'):

        assert(ratio_shift < 1.0)
        assert(len(phase_coefficients) == len(phase_ratios))
        assert(len(phase_coefficients) == len(phase_stds))

        self.phases, time = [], 0.0
        for coeff, ratio, std in zip(phase_coefficients, phase_ratios, phase_stds):
            self.phases.append(Phase(start=time, end=time+ratio, std=std, coeff=coeff))
            time += ratio

        self.ratio_shift = ratio_shift
        self.name = name

        self.xlim = sum([(phase.end - phase.start) for phase in self.phases])

        # domain
        self.x = np.linspace(0, self.xlim, num=SAMPLE_FREQ)

        # for saving plot png
        import os
        if not os.path.exists('example_gaits'):
            os.makedirs('example_gaits')

    def plot(self):

        self.fig, self.axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(10,5))
        self.axs[0].set_title(f'{self.name}')

        left_foot = probabilistic_periodic_func(self.x, self.phases)
        right_foot = probabilistic_periodic_func(self.x, shift_behavior(self.phases, shift=self.ratio_shift))

        self.axs[0].plot(self.x, left_foot)
        self.axs[1].plot(self.x, right_foot)

        left_swing_ranges = self._find_overlapping_regions((left_foot >= 0) & (right_foot < 0))
        right_swing_ranges = self._find_overlapping_regions((left_foot < 0) & (right_foot >= 0))
        aerial_ranges = self._find_overlapping_regions((left_foot >= 0) & (right_foot >= 0))
        stance_ranges = self._find_overlapping_regions((left_foot < 0) & (right_foot < 0))

        print(left_swing_ranges)
        print(right_swing_ranges)
        print(aerial_ranges)
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
                # check if single spot or if actual range is here
                if i+1 < len(bool_arr) and not bool_arr[i+1]:
                    ranges.append((start, start+1))
                    start, end = None, None
                    in_range = False
                else:
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

# class LivePlot:
#     def __init__(self, wrap=False):

#         self.wrap_viz = wrap

#     def __call__(self, pipe):
#         print('starting plotter...')

#         self.pipe = pipe
#         self.fig, self.axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(10,5))
#         self.axs[0].set_title('Live PeriodicFcn Plot')
#         self.axs[0].set_ylabel("left")
#         self.axs[1].set_ylabel("right")
        
#         # domain
#         self.ratios = [0.5, 0.5]
#         self.x = np.linspace(0, sum(self.ratios), num=SAMPLE_FREQ)

#         # first draw
#         left_foot = periodic_func(self.x, [-1, 1], [0.5, 0.5], ratio_shift=0.0, alpha=0.01)
#         right_foot = periodic_func(self.x, [-1, 1], [0.5, 0.5], ratio_shift=0.5, alpha=0.01)
#         self._draw(left_foot, right_foot)

#         timer = self.fig.canvas.new_timer(interval=10)
#         timer.add_callback(self.call_back)
#         timer.start()

#         print('...done')
#         plt.show()

#     def terminate(self):
#         plt.close('all')

#     def call_back(self):
#         while self.pipe.poll():
#             command = self.pipe.recv()
#             if command is None:
#                 self.terminate()
#                 return False
#             else:
#                 coeffs, self.ratios, ratio_shift, alpha = command
#                 left_foot = periodic_func(self.x, coeffs, self.ratios, ratio_shift=0.0, alpha=alpha)
#                 right_foot = periodic_func(self.x, coeffs, self.ratios, ratio_shift=ratio_shift, alpha=alpha)
#                 self._redraw(left_foot, right_foot)
#         self.fig.canvas.draw()
#         return True

#     def _draw(self, left_foot, right_foot):
#         self.x = np.linspace(0, sum(self.ratios), num=SAMPLE_FREQ)
#         self.l0, = self.axs[0].plot(self.x, left_foot)
#         self.l1, = self.axs[1].plot(self.x, right_foot)

#     def _redraw(self, left_foot, right_foot):
#         # get right foot force / vel data
#         self.x = np.linspace(0, sum(self.ratios), num=SAMPLE_FREQ)
#         self.l0.set_xdata(self.x)
#         self.l1.set_xdata(self.x)
#         self.l0.set_ydata(left_foot)
#         self.l1.set_ydata(right_foot)


if __name__ == '__main__':

    STD = 0.2  # variance

    # TODO: Get mirroring working so gait descriptions can be greatly simplified
    S = 1   # swing
    G = -1  # stabce
    hopping = PeriodicBehavior(
        phase_coefficients=[G, S],
        phase_ratios=[0.5, 0.5],
        phase_stds=[STD, STD],
        ratio_shift=0.0,
        name='hopping'
    )
    walking = PeriodicBehavior(
        phase_coefficients=[S, G],
        phase_ratios=[0.2, 0.8],
        phase_stds=[STD, STD],
        ratio_shift=0.5,
        name='walking'
    )
    running = PeriodicBehavior(
        phase_coefficients=[G, S],
        phase_ratios=[0.2, 0.8],
        phase_stds=[STD, STD],
        ratio_shift=0.5,
        name='running'
    )
    skipping = PeriodicBehavior(
        phase_coefficients=[G, S, G, S],
        phase_ratios=[0.2, 0.4, 0.2, 0.2],
        phase_stds=[STD, STD, STD, STD],
        ratio_shift=0.5,
        name='skipping'
    )
    hopping.plot()
    walking.plot()
    running.plot()
    skipping.plot()
    plt.show()
