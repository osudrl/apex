import numpy as np
import sys, os
import fpdf
from .eval_perturb import plot_perturb

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
    
    stats["Avg Force"] = round(np.mean(data), 2)
    stats["Max Force"] = np.max(data)
    max_ind = np.unravel_index(np.argmax(data, axis=None), data.shape)
    stats["Max Location (angle, phase)"] = (str(round(angles[max_ind[0]], 2))+chr(176), max_ind[1])
    angle_avg = np.mean(data, axis=1)
    phase_avg = np.mean(data, axis=0)
    stats["Most Robust Angle"] = angles[np.argmax(angle_avg)]
    stats["Most Robust Phase"] = np.argmax(phase_avg)

    return stats


# Note that for the spacing of the multi_cells to work out, this function assumes that 
# pol1's name is at least longer than pol2's name
def draw_headers(pdf, pol1, pol2, key_col_width, min_width):
    epw = pdf.w - 2*pdf.l_margin
    th = pdf.font_size
    pol1_width = max(pdf.get_string_width(pol1), min_width) + 0.1
    pol2_width = max(pdf.get_string_width(pol2), min_width) + 0.1
    pol2_split = False
    if pol1_width + pol2_width + key_col_width>= epw:
        pol1_width = (epw - key_col_width) / 2
        if pol2_width > pol1_width:
            pol2_split = True
        pol2_width = pol1_width

    start_x = pdf.get_x()
    start_y = pdf.get_y()
    pdf.set_x(start_x + key_col_width)

    # Draw pol1 and pol2 multicell first to figure out y height
    pdf.multi_cell(pol1_width, 2*th, pol1, border=1, align="C")
    pol1_height = pdf.get_y() - start_y
    
    pdf.set_xy(start_x+key_col_width+pol1_width, start_y)
    if pol2_split:
        pdf.multi_cell(pol2_width, 2*th, pol2, border=1, align="C")
    else:
        pdf.cell(pol2_width, pol1_height, pol2, border=1, align="C")
    pdf.set_xy(start_x, start_y)
    pdf.cell(key_col_width, pol1_height, "", border=1, align="C")
    pdf.set_xy(start_x, start_y + pol1_height)

    return pol1_width, pol2_width

def compare_pols(pol1, pol2):
    pol1 = pol1.strip("/")
    pol2 = pol2.strip("/")
    # For spacing concerns later, need pol1 to be the "longer" (name wise) of the two
    if len(os.path.basename(pol2)) > len(os.path.basename(pol1)):
        temp = pol1
        pol1 = pol2
        pol2 = temp
    pol1_name = os.path.basename(pol1)
    pol2_name = os.path.basename(pol2)
    print("pol1: ", pol1_name)
    print("pol2: ", pol2_name)

    # Initial PDF setup
    pdf = fpdf.FPDF(format='letter', unit='in')
    pdf.add_page()
    pdf.set_font('Times','',10.0) 
    # Effective page width, or just epw
    epw = pdf.w - 2*pdf.l_margin
    th = pdf.font_size
    # Set title
    pdf.cell(epw, 2*th, "Policy Robustness Comparison", 0, 1, "C")
    pdf.ln(2*th)

    # Print command test table
    pol1_command = np.load(os.path.join(pol1, "eval_commands.npy"))
    pol2_command = np.load(os.path.join(pol2, "eval_commands.npy"))
    pol1_command_stats = process_commands(pol1_command)
    pol2_command_stats = process_commands(pol2_command)

    pdf.cell(epw, 2*th, "Command Test", 0, 1, "L")
    pdf.ln(th)
    # Set column widths
    key_col_width = pdf.get_string_width(max(pol2_command_stats.keys(), key=len)) + .2 

    pol1_width, pol2_width = draw_headers(pdf, pol1_name, pol2_name, key_col_width, pdf.get_string_width(str(9.9999)))

    for key in pol2_command_stats.keys():
        pdf.cell(key_col_width, 2*th, key, border=1, align="C")
        pdf.cell(pol1_width, 2*th, str(round(pol1_command_stats[key], 4)), border=1, align="C")
        pdf.cell(pol2_width, 2*th, str(round(pol2_command_stats[key], 4)), border=1, align="C")
        pdf.ln(2*th)

    # Print perturb test table
    pdf.ln(2*th)
    pdf.cell(epw, 2*th, "Perturbation Test", 0, 1, "L")
    pdf.ln(th)
    pol1_perturb = np.load(os.path.join(pol1, "eval_perturbs.npy"))
    pol2_perturb = np.load(os.path.join(pol2, "eval_perturbs.npy"))
    pol1_perturb_stats = process_perturbs(pol1_perturb)
    pol2_perturb_stats = process_perturbs(pol2_perturb)

    # Set column widths
    key_col_width = pdf.get_string_width(max(pol2_perturb_stats.keys(), key=len)) + .2 
    pol1_width, pol2_width = draw_headers(pdf, pol1_name, pol2_name, key_col_width, pdf.get_string_width(str(999.99)))

    for key in pol2_perturb_stats.keys():
        pdf.cell(key_col_width, 2*th, key, border=1, align="C")
        pdf.cell(pol1_width, 2*th, str(pol1_perturb_stats[key]), border=1, align="C")
        pdf.cell(pol2_width, 2*th, str(pol2_perturb_stats[key]), border=1, align="C")
        pdf.ln(2*th)

    max_force = max(np.max(np.mean(pol1_perturb, axis=1)), np.max(np.mean(pol2_perturb, axis=1)))
    max_force = 50*np.ceil(max_force / 50)
    pol1_perturb_plot = os.path.join(pol1, "perturb_plot.png")
    pol2_perturb_plot = os.path.join(pol2, "perturb_plot.png")
    plot_perturb(os.path.join(pol1, "eval_perturbs.npy"), pol1_perturb_plot, max_force)
    plot_perturb(os.path.join(pol2, "eval_perturbs.npy"), pol2_perturb_plot, max_force)
    pdf.ln(2*th)

    pdf.cell(epw, 2*th, "Perturbation Plot", 0, 1, "L")
    pol2_split = False
    if pdf.get_string_width(pol2) > epw / 2:
        pol2_split = True
    start_x = pdf.get_x()
    start_y = pdf.get_y()
    pdf.multi_cell(epw/2, 2*th, pol1_name, border=0, align="C")
    pol1_height = pdf.get_y() - start_y
    pdf.set_xy(start_x+epw/2, start_y)
    if pol2_split:
        pdf.multi_cell(epw/2, 2*th, pol2_name, border=0, align="C")
    else:
        pdf.cell(epw/2, pol1_height, pol2_name, border=0, align="C")
    pdf.set_xy(start_x, start_y+pol1_height)
    pdf.image(pol1_perturb_plot, x=start_x, y=start_y+pol1_height, w = epw/2-.1)
    pdf.image(pol2_perturb_plot, x=start_x+epw/2, y = start_y+pol1_height, w = epw/2-.1)

    pdf.output("./policy_compare.pdf")


