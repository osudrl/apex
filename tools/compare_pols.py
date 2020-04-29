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

    pdf = fpdf.FPDF(format='letter', unit='in')
    pdf.add_page()
    pdf.set_font('Times','',10.0) 
    # Effective page width, or just epw
    epw = pdf.w - 2*pdf.l_margin
    th = pdf.font_size

    pdf.cell(epw, 2*th, "Policy Robustness Comparison", 0, 1, "C")
    pdf.ln(2*th)

    pdf.cell(epw, 2*th, "Command Test", 0, 1, "L")
    pdf.ln(th)
    # Set column width to 1/4 of effective page width to distribute content 
    # evenly across table and page
    col_width = epw / 3
    # pol_names = os.path.basename(pol1) + "\t" + os.path.basename(pol2)
    pdf.cell(col_width, 2*th, "", border=1, align="C")
    pdf.cell(col_width, 2*th, os.path.basename(pol1), border=1, align="C")
    pdf.cell(col_width, 2*th, os.path.basename(pol2), border=1, align="C")
    pdf.ln(2*th)
    for key in pol2_command_stats.keys():
        pdf.cell(col_width, 2*th, key, border=1, align="C")
        pdf.cell(col_width, 2*th, str(pol2_command_stats[key]), border=1, align="C")
        pdf.cell(col_width, 2*th, str(pol2_command_stats[key]), border=1, align="C")
        pdf.ln(2*th)
    print(pol2_command_stats.keys())



    pdf.output("./test.pdf")


