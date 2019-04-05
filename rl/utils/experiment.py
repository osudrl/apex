import atexit, os
import os.path as osp
from subprocess import Popen
from functools import partial
import torch.multiprocessing as mp
from .render import renderloop
from .logging import Logger
from rl.envs import Normalize, Vectorize


def run_experiment(algo, policy, env_fn, args, log=True, monitor=False, render=False):
    logger = Logger(args, viz=monitor) if log else None

    # HOTFIX for Patrick's desktop: (MP is buggy on it for some reason)
    if render:
        policy.share_memory()

        train_p = mp.Process(target=algo.train,
                         args=(env_fn, policy, args.n_itr),
                         kwargs=dict(logger=logger))
        train_p.start()

        # TODO: add normalize as a commandline argument
        renv_fn = partial(env_fn)

        renv = Normalize(Vectorize([renv_fn]))
        render_p = mp.Process(target=renderloop,
                              args=(renv, policy))
        render_p.start()

        train_p.join()
        render_p.join()
    
    else:
        algo.train(env_fn, policy, args.n_itr, logger=logger)
