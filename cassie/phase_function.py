import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt

def create_phase_reward(swing_duration, stance_duration, strict_relaxer, stance_mode, have_incentive, FREQ=40, for_viz=False):

    total_duration = 2 * swing_duration + 2 * stance_duration
    phaselength = total_duration * FREQ

    # NOTE: these times are being converted from time in seconds to phaselength
    right_swing = np.array([0.0, swing_duration]) * FREQ
    first_dblstance = np.array([swing_duration, swing_duration + stance_duration]) * FREQ
    left_swing = np.array([swing_duration + stance_duration, 2 * swing_duration + stance_duration]) * FREQ
    second_dblstance = np.array([2 * swing_duration + stance_duration, total_duration]) * FREQ

    r_frc_phase_points = np.zeros((2, 8))
    r_vel_phase_points = np.zeros((2, 8))
    l_frc_phase_points = np.zeros((2, 8))
    l_vel_phase_points = np.zeros((2, 8))

    right_swing_relax_offset = (right_swing[1] - right_swing[0]) * strict_relaxer
    l_frc_phase_points[0,0] = r_frc_phase_points[0,0] = right_swing[0] + right_swing_relax_offset
    l_frc_phase_points[0,1] = r_frc_phase_points[0,1] = right_swing[1] - right_swing_relax_offset
    l_vel_phase_points[0,0] = r_vel_phase_points[0,0] = right_swing[0] + right_swing_relax_offset
    l_vel_phase_points[0,1] = r_vel_phase_points[0,1] = right_swing[1] - right_swing_relax_offset
    # During right swing we want foot velocities and don't want foot forces
    if not have_incentive:
        l_vel_phase_points[1,:2] = r_frc_phase_points[1,:2] = np.negative(np.ones(2))  # penalize l vel and r force
        l_frc_phase_points[1,:2] = r_vel_phase_points[1,:2] = np.zeros(2)  # don't incentivize l force or r vel
    else:
        l_vel_phase_points[1,:2] = r_frc_phase_points[1,:2] = np.negative(np.ones(2))  # penalize l vel and r force
        l_frc_phase_points[1,:2] = r_vel_phase_points[1,:2] = np.ones(2)  # incentivize l force and r vel

    dbl_stance_relax_offset = (first_dblstance[1] - first_dblstance[0]) * strict_relaxer
    l_frc_phase_points[0,2] = r_frc_phase_points[0,2] = first_dblstance[0] + dbl_stance_relax_offset
    l_frc_phase_points[0,3] = r_frc_phase_points[0,3] = first_dblstance[1] - dbl_stance_relax_offset
    l_vel_phase_points[0,2] = r_vel_phase_points[0,2] = first_dblstance[0] + dbl_stance_relax_offset
    l_vel_phase_points[0,3] = r_vel_phase_points[0,3] = first_dblstance[1] - dbl_stance_relax_offset
    if stance_mode == "aerial":
        # During aerial we want foot velocities and don't want foot forces
        # During grounded walking we want foot forces and don't want velocities
        if not have_incentive:
            l_frc_phase_points[1,2:4] = r_frc_phase_points[1,2:4] = np.negative(np.ones(2))  # penalize l and r foot force
            l_vel_phase_points[1,2:4] = r_vel_phase_points[1,2:4] = np.zeros(2)  # don't incentivize l and r foot velocity
        else:
            l_frc_phase_points[1,2:4] = r_frc_phase_points[1,2:4] = np.negative(np.ones(2))  # penalize l and r foot force
            l_vel_phase_points[1,2:4] = r_vel_phase_points[1,2:4] = np.ones(2)  # incentivize l and r foot velocity
    elif stance_mode == "zero":
        l_frc_phase_points[1,2:4] = r_frc_phase_points[1,2:4] = np.zeros(2)
        l_vel_phase_points[1,2:4] = r_vel_phase_points[1,2:4] = np.zeros(2)
    else:
        # During grounded walking we want foot forces and don't want velocities
        if not have_incentive:
            l_frc_phase_points[1,2:4] = r_frc_phase_points[1,2:4] = np.zeros(2)  # don't incentivize l and r foot force
            l_frc_phase_points[1,2:4] = r_vel_phase_points[1,2:4] = np.negative(np.ones(2))  # penalize l and r foot velocity
        else:
            l_frc_phase_points[1,2:4] = r_frc_phase_points[1,2:4] = np.ones(2)  # incentivize l and r foot force
            l_vel_phase_points[1,2:4] = r_vel_phase_points[1,2:4] = np.negative(np.ones(2))  # penalize l and r foot velocity

    left_swing_relax_offset = (left_swing[1] - left_swing[0]) * strict_relaxer
    l_frc_phase_points[0,4] = r_frc_phase_points[0,4] = left_swing[0] + left_swing_relax_offset
    l_frc_phase_points[0,5] = r_frc_phase_points[0,5] = left_swing[1] - left_swing_relax_offset
    l_vel_phase_points[0,4] = r_vel_phase_points[0,4] = left_swing[0] + left_swing_relax_offset
    l_vel_phase_points[0,5] = r_vel_phase_points[0,5] = left_swing[1] - left_swing_relax_offset
    # During left swing we want foot forces and don't want foot velocities (from perspective of right foot)
    if not have_incentive:
        l_vel_phase_points[1,4:6] = r_frc_phase_points[1,4:6] = np.zeros(2)  # don't incentivize l vel and r force
        l_frc_phase_points[1,4:6] = r_vel_phase_points[1,4:6] = np.negative(np.ones(2))  # penalize l force and r vel
    else:
        l_vel_phase_points[1,4:6] = r_frc_phase_points[1,4:6] = np.ones(2)  # incentivize l vel and r force
        l_frc_phase_points[1,4:6] = r_vel_phase_points[1,4:6] = np.negative(np.ones(2))  # penalize l force and r vel

    dbl_stance_relax_offset = (second_dblstance[1] - second_dblstance[0]) * strict_relaxer
    l_frc_phase_points[0,6] = r_frc_phase_points[0,6] = second_dblstance[0] + dbl_stance_relax_offset
    l_frc_phase_points[0,7] = r_frc_phase_points[0,7] = second_dblstance[1] - dbl_stance_relax_offset
    l_vel_phase_points[0,6] = r_vel_phase_points[0,6] = second_dblstance[0] + dbl_stance_relax_offset
    l_vel_phase_points[0,7] = r_vel_phase_points[0,7] = second_dblstance[1] - dbl_stance_relax_offset
    if stance_mode == "aerial":
        # During aerial we want foot velocities and don't want foot forces
        # During grounded walking we want foot forces and don't want velocities
        if not have_incentive:
            l_frc_phase_points[1,6:] = r_frc_phase_points[1,6:] = np.negative(np.ones(2))  # penalize l and r foot force
            l_vel_phase_points[1,6:] = r_vel_phase_points[1,6:] = np.zeros(2)  # don't incentivize l and r foot velocity
        else:
            l_frc_phase_points[1,6:] = r_frc_phase_points[1,6:] = np.negative(np.ones(2))  # penalize l and r foot force
            l_vel_phase_points[1,6:] = r_vel_phase_points[1,6:] = np.ones(2)  # incentivize l and r foot velocity
    elif stance_mode == "zero":
        l_frc_phase_points[1,6:] = r_frc_phase_points[1,6:] = np.zeros(2)
        l_vel_phase_points[1,6:] = r_vel_phase_points[1,6:] = np.zeros(2)
    else:
        # During grounded walking we want foot forces and don't want velocities
        if not have_incentive:
            l_frc_phase_points[1,6:] = r_frc_phase_points[1,6:] = np.zeros(2)  # don't incentivize l and r foot force
            l_vel_phase_points[1,6:] = r_vel_phase_points[1,6:] = np.negative(np.ones(2))  # penalize l and r foot velocity
        else:
            l_frc_phase_points[1,6:] = r_frc_phase_points[1,6:] = np.ones(2)  # incentivize l and r foot force
            l_vel_phase_points[1,6:] = r_vel_phase_points[1,6:] = np.negative(np.ones(2))  # penalize l and r foot velocity

    ## extend the data to three cycles : one before and one after : this ensures continuity

    r_frc_prev_cycle = np.copy(r_frc_phase_points)
    r_vel_prev_cycle = np.copy(r_vel_phase_points)
    l_frc_prev_cycle = np.copy(l_frc_phase_points)
    l_vel_prev_cycle = np.copy(l_vel_phase_points)
    l_frc_prev_cycle[0] = r_frc_prev_cycle[0] = r_frc_phase_points[0] - r_frc_phase_points[0, -1] - dbl_stance_relax_offset
    l_vel_prev_cycle[0] = r_vel_prev_cycle[0] = r_vel_phase_points[0] - r_vel_phase_points[0, -1] - dbl_stance_relax_offset

    r_frc_second_cycle = np.copy(r_frc_phase_points)
    r_vel_second_cycle = np.copy(r_vel_phase_points)
    l_frc_second_cycle = np.copy(l_frc_phase_points)
    l_vel_second_cycle = np.copy(l_vel_phase_points)
    l_frc_second_cycle[0] = r_frc_second_cycle[0] = r_frc_phase_points[0] + r_frc_phase_points[0, -1] + dbl_stance_relax_offset
    l_vel_second_cycle[0] = r_vel_second_cycle[0] = r_vel_phase_points[0] + r_vel_phase_points[0, -1] + dbl_stance_relax_offset

    r_frc_phase_points_repeated = np.hstack((r_frc_prev_cycle, r_frc_phase_points, r_frc_second_cycle))
    r_vel_phase_points_repeated = np.hstack((r_vel_prev_cycle, r_vel_phase_points, r_vel_second_cycle))
    l_frc_phase_points_repeated = np.hstack((l_frc_prev_cycle, l_frc_phase_points, l_frc_second_cycle))
    l_vel_phase_points_repeated = np.hstack((l_vel_prev_cycle, l_vel_phase_points, l_vel_second_cycle))

    ## Create the smoothing function with cubic spline and cutoff at limits -1 and 1
    r_frc_phase_spline = PchipInterpolator(r_frc_phase_points_repeated[0], r_frc_phase_points_repeated[1])
    r_vel_phase_spline = PchipInterpolator(r_vel_phase_points_repeated[0], r_vel_phase_points_repeated[1])
    l_frc_phase_spline = PchipInterpolator(l_frc_phase_points_repeated[0], l_frc_phase_points_repeated[1])
    l_vel_phase_spline = PchipInterpolator(l_vel_phase_points_repeated[0], l_vel_phase_points_repeated[1])
    
    if for_viz:
        repeat_time = np.linspace(r_frc_phase_points_repeated[0,0], r_frc_phase_points_repeated[0,-1], num=int(2000*total_duration))
        r_frc_phase_spline_out = np.vstack([repeat_time, r_frc_phase_spline(repeat_time)])
        r_vel_phase_spline_out = np.vstack([repeat_time, r_vel_phase_spline(repeat_time)])
        l_frc_phase_spline_out = np.vstack([repeat_time, l_frc_phase_spline(repeat_time)])
        l_vel_phase_spline_out = np.vstack([repeat_time, l_vel_phase_spline(repeat_time)])
        right_foot_info = [r_frc_phase_spline, r_vel_phase_spline, r_frc_phase_spline_out, r_vel_phase_spline_out, r_frc_phase_points_repeated, r_vel_phase_points_repeated]
        left_foot_info = [l_frc_phase_spline, l_vel_phase_spline, l_frc_phase_spline_out, l_vel_phase_spline_out, l_frc_phase_points_repeated, l_vel_phase_points_repeated]
        return right_foot_info, left_foot_info, repeat_time
    
    return [r_frc_phase_spline, r_vel_phase_spline], [l_frc_phase_spline, l_vel_phase_spline], phaselength

def encode_stance_mode(mode):
    if mode == "grounded":
        return np.array([1.,0.,0.])
    elif mode == "aerial":
        return np.array([0.,1.,0.])
    elif mode == "zero":
        return np.array([0.,0.,1.])
    else:
        print("oops")
        exit()

def decode_stance_mode(code):
    if (code == np.array([1.,0.,0.])).all():
        return "grounded"
    elif (code == np.array([0.,1.,0.])).all():
        return "aerial"
    elif (code == np.array([0.,0.,1.])).all():
        return "zero"
    else:
        print("oops2")
        exit()


def time_to_phase(x, FREQ=40):
    return x * FREQ
def phase_to_time(x, FREQ=40):
    return x / FREQ

class LivePlot:
    def __init__(self, wrap=False):

        self.wrap_viz = wrap

    def __call__(self, pipe):
        print('starting plotter...')

        self.pipe = pipe
        self.fig, self.axs = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(10,5), constrained_layout=True)
        plt.subplots_adjust(left=0.25, bottom=0.3)
        self.axs[0].set_ylabel("l force function")
        self.axs[1].set_ylabel("l vel function")
        self.axs[2].set_ylabel("r force function")
        self.axs[3].set_ylabel("r vel function")
        self.second_x_axis = self.axs[0].secondary_xaxis('top', functions=(phase_to_time, time_to_phase))
        self._draw(*create_phase_reward(0.25, 0.15, 0.1, "zero", True, for_viz=True))
        timer = self.fig.canvas.new_timer(interval=1000)
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
                self._redraw(*create_phase_reward(*command, for_viz=True))
        self.fig.canvas.draw()
        return True

    def _set_axis(self, points):
        if self.wrap_viz:
            [self.axs[i].set_xlim(points[0, 0], points[0, -1]) for i in range(4)]
        else:
            [self.axs[i].set_xlim(points[0, 7], points[0, 15]) for i in range(4)]

    def _draw(self, right_foot_info, left_foot_info, repeat_time):
        r_frc_phase_spline, r_vel_phase_spline, r_frc_phase_spline_out, r_vel_phase_spline_out, r_frc_phase_points_repeated, r_vel_phase_points_repeated = right_foot_info
        l_frc_phase_spline, l_vel_phase_spline, l_frc_phase_spline_out, l_vel_phase_spline_out, l_frc_phase_points_repeated, l_vel_phase_points_repeated = left_foot_info
        self.s0 = self.axs[0].scatter(r_frc_phase_points_repeated[0], r_frc_phase_points_repeated[1], color='black')
        self.s1 = self.axs[1].scatter(r_vel_phase_points_repeated[0], r_vel_phase_points_repeated[1], color='black')
        self.s2 = self.axs[2].scatter(l_frc_phase_points_repeated[0], l_frc_phase_points_repeated[1], color='black')
        self.s3 = self.axs[3].scatter(l_vel_phase_points_repeated[0], l_vel_phase_points_repeated[1], color='black')
        self.l0, = self.axs[0].plot(r_frc_phase_spline_out[0], r_frc_phase_spline_out[1], color='black')
        self.l1, = self.axs[1].plot(r_vel_phase_spline_out[0], r_vel_phase_spline_out[1], color='black')
        self.l2, = self.axs[2].plot(l_frc_phase_spline_out[0], l_frc_phase_spline_out[1], color='black')
        self.l3, = self.axs[3].plot(l_vel_phase_spline_out[0], l_vel_phase_spline_out[1], color='black')
        self._set_axis(r_frc_phase_points_repeated)

    def _redraw(self, right_foot_info, left_foot_info, repeat_time):
        # get right foot force / vel data
        r_frc_phase_spline, r_vel_phase_spline, r_frc_phase_spline_out, r_vel_phase_spline_out, r_frc_phase_points_repeated, r_vel_phase_points_repeated = right_foot_info
        l_frc_phase_spline, l_vel_phase_spline, l_frc_phase_spline_out, l_vel_phase_spline_out, l_frc_phase_points_repeated, l_vel_phase_points_repeated = left_foot_info
        self.l0.set_xdata(r_frc_phase_spline_out[0])
        self.l0.set_ydata(r_frc_phase_spline_out[1])
        self.l1.set_xdata(r_vel_phase_spline_out[0])
        self.l1.set_ydata(r_vel_phase_spline_out[1])
        self.s0.set_offsets(r_frc_phase_points_repeated.T)
        self.s1.set_offsets(r_vel_phase_points_repeated.T)
        # left vel == right force | left force == right vel
        self.l2.set_xdata(l_frc_phase_spline_out[0])
        self.l2.set_ydata(l_frc_phase_spline_out[1])
        self.l3.set_xdata(l_vel_phase_spline_out[0])
        self.l3.set_ydata(l_vel_phase_spline_out[1])
        self.s2.set_offsets(l_frc_phase_points_repeated.T)
        self.s3.set_offsets(l_vel_phase_points_repeated.T)
        self._set_axis(r_frc_phase_points_repeated)
