import numpy as np
import sys, os, argparse
# from .eval_perturb import plot_perturb

def process_commands(data):
    stats = {}
    num_iters = data.shape[0]
    pass_rate = np.sum(data[:, 0]) / num_iters
    stats["Pass Rate"] = pass_rate
    success_inds = np.where(data[:, 0] == 1)[0]
    speed_fail_inds = np.where(data[:, 1] == 0)[0]
    orient_fail_inds = np.where(data[:, 1] == 1)[0]

    speed_change = data[speed_fail_inds, 4]
    orient_change = data[orient_fail_inds, 5]
    speed_neg_inds = np.where(speed_change < 0)
    speed_pos_inds = np.where(speed_change > 0)
    orient_neg_inds = np.where(orient_change < 0)
    orient_pos_inds = np.where(orient_change > 0)
    stats["Number of speed fails"] = len(speed_fail_inds)
    stats["Number of orient fails"] = len(orient_fail_inds)
    if len(speed_fail_inds) == 0:
        avg_pos_speed = "N/A"
        avg_neg_speed = "N/A"
    else:
        avg_pos_speed = np.mean(speed_change[speed_pos_inds])
        avg_neg_speed = np.mean(speed_change[speed_neg_inds])
    if len(orient_fail_inds) == 0:
        avg_pos_orient = "N/A"
        avg_neg_orient = "N/A"
    else:
        avg_pos_orient = np.mean(orient_change[orient_pos_inds])
        avg_neg_orient = np.mean(orient_change[orient_neg_inds])

    stats["Avg pos speed fails"] = avg_pos_speed
    stats["Avg neg speed fails"] = avg_neg_speed
    stats["Avg pos_orient fails"] = avg_pos_orient
    stats["Avg neg_orient fails"] = avg_neg_orient

    return stats

def process_perturbs(data):
    stats = {}
    num_angles, num_phases = data.shape
    angles = 360*np.linspace(0, 1, num_angles+1)
    
    stats["Avg Force"] = round(np.mean(data[:, 14]), 2)
    stats["Max Force"] = np.max(data)
    max_ind = np.unravel_index(np.argmax(data, axis=None), data.shape)
    stats["Max Location (angle, phase)"] = (str(round(angles[max_ind[0]], 2))+chr(176), max_ind[1])
    angle_avg = np.mean(data, axis=1)
    phase_avg = np.mean(data, axis=0)
    stats["Most Robust Angle"] = angles[np.argmax(angle_avg)]
    stats["Most Robust Phase"] = np.argmax(phase_avg)

    return stats

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="./trained_models/nodelta_neutral_StateEst_symmetry_speed0-3_freq1-2", help="path to folder containing policy and run details")
args = parser.parse_args()

# for filename in os.listdir(args.path):
#     if "eval_commands" in filename:
#         command_data = np.load(os.path.join(args.path, filename))
#     elif "eval_perturbs" in filename:
#         perturb_data = np.load(os.path.join(args.path, filename))

command_data = np.load(os.path.join(args.path, "eval_commands_cassie_tray_box.npy"))
perturb_data = np.load(os.path.join(args.path, "eval_perturbs_cassie_tray_box.npy"))

command_stats = process_commands(command_data)
for key in command_stats.keys():
    print("{}:\t\t{}".format(key, command_stats[key]))
print("\n")
perturb_stats = process_perturbs(perturb_data)
for key in perturb_stats.keys():
    print("{}:\t\t{}".format(key, perturb_stats[key]))
