"""
Script for transferring policies trained under apex v0.2.0 to apex v0.3.0
"""
import os
import argparse
import pickle

from collections import OrderedDict
import hashlib

def rename_experiment(logdir, new_name):
    os.rename(os.path.join(logdir, "experiment.pkl"), os.path.join(logdir, new_name+".pkl"))
    os.rename(os.path.join(logdir, "experiment.info"), os.path.join(logdir, new_name+".info"))
    print(f"Renamed old files to {new_name}.info and {new_name}.pkl")

# Modification of util.log.create_logger
def create_experiment(args, logdir):
    """Use hyperparms to set a directory to output diagnostic files."""

    arg_dict = args.__dict__
    assert "env_name" in arg_dict, \
    "You must provide a 'env_name' key in your command line arguments."

    # sort the keys so the same hyperparameters will always have the same hash
    arg_dict = OrderedDict(sorted(arg_dict.items(), key=lambda t: t[0]))

    # remove seed so it doesn't get hashed, store value for filename
    # same for logging directory
    run_name = arg_dict.pop('run_name')
    env_name = str(arg_dict['env_name'])

    os.makedirs(logdir, exist_ok=True)

    # Create a file with all the hyperparam settings in human-readable plaintext,
    # also pickle file for resuming training easily
    info_path = os.path.join(logdir, "experiment.info")
    pkl_path = os.path.join(logdir, "experiment.pkl")
    with open(pkl_path, 'wb') as file:
        pickle.dump(args, file)
    with open(info_path, 'w') as file:
        for key, val in arg_dict.items():
            file.write("%s: %s" % (key, val))
            file.write('\n')
    print("Wrote new files to experiment.info and experiment.pkl")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="./nodelta_neutral_StateEst_symmetry_speed0-3_freq1-2", help="path to folder containing policy and run details")
    script_args = parser.parse_args()

    run_args = pickle.load(open(os.path.join(script_args.path, "experiment.pkl"), "rb"))

    if hasattr(run_args, "command_profile"):
        print("Looks like this policy has already been converted to Apex v0.3.0.")
        ans = input("Sure you want to continue with conversion? (y/n)")
        if ans.lower() == "n":
            exit()
    
    # rename old files
    rename_experiment(script_args.path, "experiment_bkup")

    args = argparse.Namespace()
    args.simrate = run_args.simrate
    args.recurrent = run_args.recurrent
    if hasattr(run_args, "learn_gains"):
        args.learn_gains = run_args.learn_gains
    else:
        args.learn_gains = False
    args.history = run_args.history
    args.reward = run_args.reward
    args.dyn_random = run_args.dyn_random
    args.mirror = run_args.mirror
    args.run_name = run_args.run_name
    # Transfer to new traj vs. no-traj env separation
    if run_args.no_delta:
        args.env_name = "Cassie-v0"
        args.traj = None
        args.ik_baseline = None
        args.no_delta = True
    else:
        args.env_name = "CassieTraj-v0"
        args.traj = run_args.traj
        args.ik_baseline = run_args.ik_baseline
        args.no_delta = run_args.no_delta

    # Transfer to new command_profile system
    if run_args.clock_based:
        args.command_profile = "clock"
    elif run_args.phase_based:
        args.command_profile = "phase"
    else:
        args.command_profile = "clock"

    # transfer to new input profile system
    if "Min" in run_args.env_name:
        args.input_profile = "min_foot"
    else:
        args.input_profile = "full"

    # old policies did not have side speed
    args.has_side_speed = False

    create_experiment(args, script_args.path)
