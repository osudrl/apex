import atexit, os
import os.path as osp
from subprocess import Popen
import torch.multiprocessing as mp
from .render import renderloop
from .logging import Logger


def run_experiment(algo, policy, env_fn, args, renv, log=True, monitor=False, render=False):
    logger = Logger(args) if log else None

    policy.share_memory()

    train_p = mp.Process(target=algo.train,
                         args=(env_fn, policy, args.n_itr),
                         kwargs=dict(logger=logger))
    train_p.start()

    if render:
        render_p = mp.Process(target=renderloop,
                              args=(renv, policy))
        render_p.start()

    if monitor:
        assert log, \
        "Log must also be set to True if monitor is set to True"

        active_log = logger.output_file.name
        bokeh_dir = osp.join(os.getcwd(), 'rl', 'monitors', 'bokeh_monitor.py')

        monitor_proc = Popen(
            ['bokeh', 'serve', '--show', bokeh_dir, '--args', active_log],
            #cwd=osp.join(os.getcwd(), 'rl', 'monitors')
        )

        atexit.register(monitor_proc.kill)

    train_p.join()

    if render:
        render_p.join()
