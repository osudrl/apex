import torch.multiprocessing as mp
from utils.evaluation import renderloop


def run_experiment(algo, args, render=False):
    algo.policy.share_memory()

    train_p = mp.Process(target=algo.train,
                         args=(args.n_itr, args.n_trj, args.max_trj_len, True))
    train_p.start()

    if render:
        render_p = mp.Process(target=renderloop,
                              args=(algo.env, algo.policy, args.max_trj_len))
        render_p.start()

    train_p.join()

    if render:
        render_p.join()
