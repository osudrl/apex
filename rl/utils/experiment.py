import torch.multiprocessing as mp
from rl.utils import renderloop


def run_experiment(algo, args, render=False, logger=None):
    algo.policy.share_memory()

    train_p = mp.Process(target=algo.train,
                         args=(args.n_itr, args.n_trj, args.max_trj_len),
                         kwargs=dict(adaptive=True, logger=logger))
    train_p.start()

    if render:
        render_p = mp.Process(target=renderloop,
                              args=(algo.env, algo.policy, args.max_trj_len))
        render_p.start()

    train_p.join()

    if render:
        render_p.join()
