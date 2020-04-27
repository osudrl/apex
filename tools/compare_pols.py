import numpy as np
import sys, os
import fpdf

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

if len(sys.argv) != 2:
    print("Error: Only takes in two policies as input")

def compare_pols(pol1, pol2):
    print("opening file ", os.path.join(pol1, "eval_commands.npy"))
    # pol1_command = np.load(os.path.join(pol1, "eval_commands.npy"))
    pol2_command = np.load(os.path.join(pol2, "eval_commands.npy"))

    # pol1_command_stats = process_commands(pol1_command)
    pol2_command_stats = process_commands(pol2_command)

    pdf = fpdf.FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "Policy Robustness Comparison", "C")



    pdf.output("./test.pdf")


