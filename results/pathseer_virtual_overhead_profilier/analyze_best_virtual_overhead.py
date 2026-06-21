# This script finds the best virtual overhead for each workload and performs:
# 1) Quadratic fits of throughput vs virtual overhead per Max_val.
# 2) For each Max_val, computes the VO at which the fitted quadratic reaches its
#    maximum within [min_vo, max_vo].
# 3) A linear fit between selectivity (1/Max_val) and that per-Max_val "best VO".
# 4) Draws a dashed vertical line at the PREDICTED VO from the linear fit evaluated
#    at each Max_val's selectivity, using the SAME color as that Max_val's curve.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys
import os
import re

assert len(sys.argv) == 2, "Usage: python script.py <input_dir>"

def quadratic_func(x, a, b, c):
    return a * x**2 + b * x + c

result_dir = os.path.join(sys.argv[1], "plots")
log_dir = os.path.join(sys.argv[1], "logs")
os.makedirs(result_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

for ff in sorted(os.listdir(sys.argv[1])):
    if not ff.endswith(".txt") or ff.find("_prev")!=-1:
        continue

    # Per-file containers
    file_res = {}              # {max_val_str: [vo_list, tput_list]}
    dist_comp_ns = None
    filter_time_ns = None

    # Parse filename
    fname = ff
    pos = fname.find("_")
    method = fname[:pos]
    fname = fname[pos+1:]
    pos = fname.find("_")
    dataset = fname[:pos]
    nsleep = int(re.search(r"sleep_(\d+).txt", ff).group(1))
    M = int(re.search(r"M_(\d+)_", ff).group(1))

    # for the TLE ones, we skip them for the final fitting
    skip_mark = False

    # Parse file content
    with open(os.path.join(sys.argv[1], ff), "r") as ifs:
        current_key = None
        for line in ifs:
            if line.startswith("[Max_val]"):
                current_key = line.split()[1]  # e.g., "1000"
                file_res[current_key] = [[], []]  # VO list, TPUT list
            elif line.startswith("[EarlyStop]"):
                skip_mark = True
            elif line.startswith("[Virtual_overhead_each_summary]"):
                content = line.split()
                tm = 1e6 * 1e3 / int(content[1].split("=")[-1])  # preserve original conversion
                vo = int(content[2].split("=")[-1])
                if current_key is None:
                    continue
                file_res[current_key][0].append(vo)
                if skip_mark:
                    file_res[current_key][1].append(-tm)  # if this should be skip, we mark the time < 0 to denote that it should only be plot, not participating in fitting
                    skip_mark = False
                else:
                    file_res[current_key][1].append(tm)
            elif line.startswith("[Dist-comp-profile]"):
                dist_comp_ns = int(line.split()[1])
            elif line.startswith("[Filter-time-profile]"):
                filter_time_ns = int(line.split()[1])

    # Output paths
    plot_file = os.path.join(result_dir, f"{dataset}_{method}_M_{M}_nsleep_{nsleep}.png")
    log_file = os.path.join(log_dir, f"{dataset}_{method}_M_{M}_nsleep_{nsleep}.txt")

    with open(log_file, "w") as ofs:
        if dist_comp_ns is not None:
            ofs.write(f"[Dist-comp-profile] {dist_comp_ns}\n")
        if filter_time_ns is not None:
            ofs.write(f"[Filter-time-profile] {filter_time_ns}\n")

        plt.clf()
        all_r = []
        all_coefficients = []
        all_best_tput = []
        all_boundary = []

        min_boundary = -1
        max_boundary = -1

        # Accumulate raw points text for the log tail
        output_str = ""

        # For linear fit: x=selectivity, y=VO_at_interval_max
        selectivities = []           # x = 1 / Max_val
        best_vo_positions = []       # y = argmax VO within interval
        sel_vo_pairs = []            # (sel, best_vo, max_val_str)
        color_map = {}               # key -> curve color (for predicted verticals)

        # First pass: fit each Max_val, plot curves, collect data (no predicted verticals yet)
        for key, value in file_res.items():

            x_plot = []
            x_fit = []
            y_plot = []
            y_fit = []

            for iii in range(len(value[0])):
                if value[1][iii] < 0:
                    x_plot.append(value[0][iii])
                    y_plot.append(-value[1][iii])
                else:
                    x_plot.append(value[0][iii])
                    y_plot.append(value[1][iii])
                    x_fit.append(value[0][iii])
                    y_fit.append(value[1][iii])

            x_plot = np.array(x_plot, dtype=float)  # virtual overhead
            y_plot = np.array(y_plot, dtype=float)  # throughput

            x_fit = np.array(x_fit, dtype=float)  # virtual overhead
            y_fit = np.array(y_fit, dtype=float)  # throughput

            if x_plot.size == 0 or y_plot.size == 0:
                continue

            # Scatter raw points
            plt.scatter(x_plot, y_plot, label=f"Max_val={key}")

            # Log raw points
            output_str += f"virtual_overhead(max_val={key})\t"
            ofs.write(f"Max_val={key}\n")
            ofs.write("Virtual overhead\t")
            for i in x_plot:
                ofs.write(str(int(i)) + "\t")
                output_str += (str(int(i)) + "\t")
            ofs.write("\n")
            ofs.write("Throughput\t")
            output_str += "\nthroughput\t"
            for i in y_plot:
                ofs.write(f"{i}\t")
                output_str += (f"{i}\t")
            ofs.write("\n")
            output_str += "\n"

            x, y = x_fit, y_fit

            # Quadratic fit
            try:
                params_scipy, _ = curve_fit(quadratic_func, x, y)
                a, b, c = params_scipy
            except Exception as e:
                ofs.write(f"[Quadratic fit failed for Max_val={key}] {e}\n")
                continue

            ofs.write(f"Fit result: tput = {a} * VO^2 + {b} * VO + {c}\n")

            # Bounds for this Max_val
            min_virtual_overhead = float(np.min(x))
            max_virtual_overhead = float(np.max(x))

            if min_boundary == -1:
                min_boundary = min_virtual_overhead
            else:
                min_boundary = min(min_boundary, min_virtual_overhead)

            if max_boundary == -1:
                max_boundary = max_virtual_overhead
            else:
                max_boundary = max(max_boundary, max_virtual_overhead)

            all_boundary.append((min_virtual_overhead, max_virtual_overhead))
            all_coefficients.append([a, b, c])

            # VO where the quadratic achieves MAX within the interval
            candidates_vo = [min_virtual_overhead, max_virtual_overhead]
            axis = -b / (2 * a) if abs(a) >= 1e-12 else float('nan')
            if not np.isnan(axis) and (a < 0) and (min_virtual_overhead < axis < max_virtual_overhead):
                candidates_vo.append(axis)

            cand_vals = [quadratic_func(v, a, b, c) for v in candidates_vo]
            idx = int(np.argmax(cand_vals))
            best_vo_for_this_max = float(candidates_vo[idx])
            best_val = float(cand_vals[idx])

            ofs.write(f"[VO_at_interval_max] {best_vo_for_this_max}\n")
            ofs.write(f"Interval best tput: {best_val:.2f}\n")
            all_best_tput.append(best_val)

            # R^2 for quadratic fit
            y_fit = quadratic_func(x, a, b, c)
            ss_res_manual = np.sum((y - y_fit) ** 2)
            ss_tot_manual = np.sum((y - np.mean(y)) ** 2)
            r = 1 - (ss_res_manual / ss_tot_manual) if ss_tot_manual > 0 else float('nan')
            all_r.append(r)
            ofs.write(f"R={r:.4f}\n")

            # Plot quadratic curve and remember its color
            x_smooth = np.linspace(min_virtual_overhead, max_virtual_overhead, 200)
            line_obj, = plt.plot(
                x_smooth,
                quadratic_func(x_smooth, a, b, c),
                label=f"Max_val={key}",
                linewidth=1
            )
            curve_color = line_obj.get_color()
            color_map[key] = curve_color

            # Collect (selectivity, best_vo) for linear fit (all tests participate)
            try:
                max_val_float = float(key)
                sel = 1.0 / max_val_float
            except:
                sel = np.nan

            if not (np.isnan(sel) or np.isnan(best_vo_for_this_max)):
                selectivities.append(sel)
                best_vo_positions.append(best_vo_for_this_max)
                sel_vo_pairs.append((sel, best_vo_for_this_max, key))

        # Average correlation of quadratic fits
        if len(all_r) > 0:
            ofs.write(f"[Average correlation coefficient] {np.nanmean(all_r):.4f}\n")
        else:
            ofs.write(f"[Average correlation coefficient] NA\n")

        # Global VO scan
        if min_boundary == -1 or max_boundary == -1 or len(all_coefficients) == 0:
            ofs.write("[Scan skipped] insufficient fit results.\n")
        else:
            best_performance_degradation = -1.0
            best_vo = 0.0
            max_deg_best = 0.0
            best_vo_max = 0.0
            current_max_deg = None
            avg_deg = []
            final_perf_sum = 0

            all_step_nr = 100
            step_size = (max_boundary - min_boundary) / max(1, (all_step_nr - 1))
            ofs.write(f"[Step_size] {step_size}\n")

            for vo in np.linspace(min_boundary, max_boundary, all_step_nr):
                degradation = 0.0
                max_deg = 0.0
                perf_sum = 0
                for i in range(len(all_coefficients)):
                    boundary = all_boundary[i]
                    this_vo = max(min(vo, boundary[1]), boundary[0])  # clamp to bounds
                    a, b, c = all_coefficients[i]
                    perf = quadratic_func(this_vo, a, b, c)

                    if perf > all_best_tput[i] + 1e-9:
                        perf = all_best_tput[i]
                    
                    perf_sum += perf

                    deg = max(0.0, 1.0 - perf / max(all_best_tput[i], 1e-12))
                    degradation += deg
                    max_deg = max(max_deg, deg)
                    avg_deg.append(deg)

                degradation /= len(all_coefficients)

                if best_performance_degradation < 0 or degradation < best_performance_degradation:
                    best_performance_degradation = degradation
                    best_vo = vo
                    max_deg_best = max_deg
                    final_perf_sum = perf_sum

                if current_max_deg is None or max_deg < current_max_deg:
                    current_max_deg = max_deg
                    best_vo_max = vo

            ofs.write(f"[Best_virtual_overhead] {best_vo}\n")
            ofs.write(f"[Best_virtual_overhead_max] {best_vo_max}\n")
            ofs.write(
                f"[Performance_degradation] avg={best_performance_degradation * 100:.2f}%, "
                f"max={max_deg_best * 100:.2f}%, "
                f"avg_deg={np.mean(avg_deg) * 100:.2f}%\n"
            )
            ofs.write(f"[Performance_sum] {final_perf_sum}\n")

            # Mark chosen VO on plot
            plt.axvline(best_vo, linewidth=1.2)

        # Raw text result block
        ofs.write("\n\n[Text Result]\n" + output_str)

        # write out the best average VO
        ofs.write("[Best_VO_Average]" + str(np.mean(best_vo_positions)) + "\n")

        # Linear fit: Best-VO vs selectivity, then draw predicted verticals at each selectivity
        ofs.write("\n[Best-VO vs selectivity linear fit]\n")
        if len(selectivities) >= 2:
            x_sel = np.array(selectivities, dtype=float)
            y_best_vo = np.array(best_vo_positions, dtype=float)

            # Linear fit: y = p * x + q
            p, q = np.polyfit(x_sel, y_best_vo, 1)
            y_pred = p * x_sel + q

            ss_res = np.sum((y_best_vo - y_pred) ** 2)
            ss_tot = np.sum((y_best_vo - np.mean(y_best_vo)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')

            ofs.write(f"BestVO = {p} * selectivity + {q}\n")
            ofs.write(f"[LoadCoefficient] {p} {q}\n")
            ofs.write(f"R={r2:.4f}\n")
            ofs.write("selectivity\tpredicted_bestVO\tMax_val\n")

            # Draw predicted vertical lines at each selectivity using the curve color
            for sel, _, k in sorted(sel_vo_pairs, key=lambda t: t[0]):
                pred_vo = p * sel + q
                ofs.write(f"{sel}\t{pred_vo}\t{k}\n")
                if k in color_map:
                    plt.axvline(pred_vo, linestyle='--', linewidth=0.9, alpha=0.8, color=color_map[k])
        else:
            ofs.write("Insufficient points for linear fit (need >= 2)\n")

        # Finalize plot
        plt.legend()
        plt.xlabel("Virtual overhead (ns)")
        plt.ylabel("Throughput")
        plt.tight_layout()
        plt.savefig(plot_file)
