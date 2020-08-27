import torch
import os
import sys, time

from tkinter import *
import multiprocessing as mp
from cassie.periodicfn.probabilistic.vonmises_clock import LivePlot
import matplotlib.pyplot as plt

from cassie.cassiemujoco import CassieSim

import tty
import termios
import select
import numpy as np

from util.env import env_factory

class EvalProcessClass():
    def __init__(self, args, run_args):

        if run_args.command_profile == "phase" and not args.no_viz:
            self.live_plot = True
            self.plot_pipe, plotter_pipe = mp.Pipe()
            self.plotter = LivePlot()
            self.plot_process = mp.Process(target=self.plotter, args=(plotter_pipe,), daemon=True)
            self.plot_process.start()
        else:
            self.live_plot = False

    #TODO: Add pausing, and window quiting along with other render functionality
    def eval_policy(self, policy, args, run_args):

        def print_input_update(e):
            print(f"\n\ratio: {e.ratio[0]:.2f},{e.ratio[1]:.2f}\t shift: {e.period_shift[0]:.2f},{e.period_shift[1]:.2f}\t coeff: {e.coeff[0]},{e.coeff[1]}\n")

        if self.live_plot:
            send = self.plot_pipe.send

        def isData():
            return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])
        
        if args.debug:
            args.stats = False

        if args.reward is None:
            args.reward = run_args.reward

        max_traj_len = args.traj_len
        visualize = not args.no_viz
        print("env name: ", run_args.env_name)
        if run_args.env_name is None:
            env_name = args.env_name
        else:
            env_name = run_args.env_name
        
        if hasattr(run_args, "has_side_speed"):
            has_side_speed = run_args.has_side_speed
        else:
            has_side_speed = True

        env = env_factory(
            args.env_name,
            command_profile=run_args.command_profile,
            input_profile=run_args.input_profile,
            simrate=run_args.simrate,
            dynamics_randomization=run_args.dyn_random,
            mirror=run_args.mirror,
            learn_gains=run_args.learn_gains,
            reward=run_args.reward,
            history=run_args.history,
            no_delta=run_args.no_delta,
            traj=run_args.traj,
            ik_baseline=run_args.ik_baseline,
            has_side_speed=has_side_speed,
            training=False,
            debug=args.debug
        )()

        if self.live_plot:
            send((env.gait, env.period_shift))

        if args.terrain is not None and ".npy" in args.terrain:
            print("HJERE")
            env.sim = CassieSim("cassie_hfield.xml")
            hfield_data = np.load(os.path.join("./cassie/cassiemujoco/terrains/", args.terrain))
            env.sim.set_hfield_data(hfield_data.flatten())

        if hasattr(policy, 'init_hidden_state'):
            policy.init_hidden_state()

        old_settings = termios.tcgetattr(sys.stdin)

        slowmo = False

        if visualize:
            env.render()
        render_state = True
        try:
            tty.setcbreak(sys.stdin.fileno())

            state = env.reset_for_test()
            done = False
            timesteps = 0
            eval_reward = 0
            speed = 0.0
            side_speed = 0.0

            env.update_speed(speed, side_speed)

            while render_state:
            
                if isData():
                    gait_updated = False
                    c = sys.stdin.read(1)
                    if c == 'w':
                        speed += 0.1
                    elif c == 's':
                        speed -= 0.1
                    elif c == 'd':
                        side_speed += 0.02
                    elif c == 'a':
                        side_speed -= 0.02
                    elif c == 'j':
                        env.phase_add += .1
                        # print("Increasing frequency to: {:.1f}".format(env.phase_add))
                    elif c == 'h':
                        env.phase_add -= .1
                        # print("Decreasing frequency to: {:.1f}".format(env.phase_add))
                    elif c == 'l':
                        env.orient_add -= .1
                        # print("Increasing orient_add to: ", orient_add)
                    elif c == 'k':
                        env.orient_add += .1
                        # print("Decreasing orient_add to: ", orient_add)
                    
                    # increase first phase ratio
                    elif c == 'x':
                        env.ratio[0] += .01
                        env.ratio[1] = 1 - env.ratio[0]
                        gait_updated = True
                    # decrease first phase ratio
                    elif c == 'z':
                        env.ratio[0] -= .01
                        env.ratio[1] = 1 - env.ratio[0]
                        gait_updated = True

                    # # change first phase to opposite
                    # elif c == 'v':
                    #     env.coeff[0] *= -1
                    #     gait_updated = True
                    # # change second phase to opposite
                    # elif c == 'c':
                    #     env.coeff[1] *= -1
                    #     gait_updated = True

                    # Hopping
                    elif c == '1':
                        env.period_shift = [0.0, 0.0]
                        gait_updated = True
                    # Walking/Running
                    elif c == '2':
                        env.period_shift = [0.0, 0.5]
                        gait_updated = True

                    elif c == '0':
                        if hasattr(policy, 'init_hidden_state'):
                            policy.init_hidden_state()
                    elif c == 'r':
                        state = env.reset()
                        speed = env.speed
                        if hasattr(policy, 'init_hidden_state'):
                            policy.init_hidden_state()
                        print("Resetting environment via env.reset()")
                    elif c == 'p':
                        push = 100
                        push_dir = 2
                        force_arr = np.zeros(6)
                        force_arr[push_dir] = push
                        env.sim.apply_force(force_arr)
                    elif c == 't':
                        slowmo = not slowmo
                        print("Slowmo : \n", slowmo)
                    
                    if gait_updated:
                        print_input_update(env)
                        env.update_gait()

                    env.update_speed(speed, side_speed)
                    # print(speed)

                    if self.live_plot:
                        send((env.gait, env.period_shift))
                
                if args.stats:
                    print(f"act spd: {env.sim.qvel()[0]:+.2f}   cmd speed: {env.speed:+.2f}   cmd_sd_spd: {env.side_speed:+.2f}   phase add: {env.phase_add:.2f}   orient add: {env.orient_add:+.2f}", end="\r")

                if hasattr(env, 'simrate'):
                    start = time.time()

                if (not env.vis.ispaused()):
                        
                    action = policy.forward(torch.Tensor(state), deterministic=True).detach().numpy()

                    state, reward, done, _ = env.step(action)
                    
                    eval_reward += reward
                    timesteps += 1

                if visualize:
                    render_state = env.render()
                if hasattr(env, 'simrate'):
                    end = time.time()
                    delaytime = max(0, env.simrate / 2000 - (end-start))
                    if slowmo:
                        while(time.time() - end < delaytime*10):
                            env.render()
                            time.sleep(delaytime)
                    else:
                        time.sleep(delaytime)

            print("Eval reward: ", eval_reward)

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)