import re
import matplotlib.pyplot as plt
import os
import sys
from collections import defaultdict
import pandas as pd
import numpy as np

# this script plot the performance of different methods with incremental techniques

def parser_performance(file_path, target_recall):
    res_tput = 0
    res_recall = 0
    res_dist = 0
    res_filter = 0
    with open(file_path, "r") as ifs:
        for line in ifs:
            if line.startswith("[PointSearch]") or line.startswith("[Result]"):
                recall = float(re.search(r"avg_recall=(\d+\.\d+)\s%", line).group(1))
                tput = float(re.search(r"Tput=(\d+\.\d+)\sops", line).group(1))
                nfilter = int(re.search(r"avg_filter=(\d+),", line).group(1))
                ndist = int(re.search(r"avg_dist=(\d+),", line).group(1))

                if abs(target_recall - recall) < abs(target_recall - res_recall):
                    res_recall = recall
                    res_tput = tput
                    res_dist = ndist
                    res_filter = nfilter

    return (res_recall, res_tput, res_dist, res_filter)

def get_avg_tput_recall(base_dir, filepath_list, target_recall):
    each_res = [[], [], [], []] # [recall, tput, dist, filter]
    for target_name in filepath_list:
        # get the result
        target_file = os.path.join(base_dir, target_name)
        if os.path.isdir(target_file):
            continue
        (res_recall, res_tput, res_dist, res_filter) = parser_performance(target_file, target_recall)
        each_res[0].append(res_recall)
        each_res[1].append(res_tput)
        each_res[2].append(res_dist)
        each_res[3].append(res_filter)

    # merge multiple results
    avg_recall = np.mean(each_res[0])
    avg_tput = np.mean(each_res[1])
    avg_dist = np.mean(each_res[2])
    avg_filter = np.mean(each_res[3])
    return (avg_recall, avg_tput, avg_dist, avg_filter)

if __name__=="__main__":
    if len(sys.argv) != 2:
        print("Usage: python get_individual_performance.py <path_dir>")
        exit(1)

    inspected_dataset = ["SIFT1M", "HM", "LAION", "Paper", "Arxiv", "Amazon"]

    # merge and handle multiple results
    workload_to_files = defaultdict(list)
    for ff in sorted(os.listdir(sys.argv[1])):
        if not ff.endswith(".txt"):
            continue

        pos = ff.find(":")
        if pos!=-1:
            fname = ff[pos+1:]
        else:
            fname = ff
        workload_to_files[fname].append(ff)

    # method -> (recall, tput)
    result_all = defaultdict(list)

    # plot the result
    result_dir = os.path.join(sys.argv[1], "plots")
    log_dir = os.path.join(sys.argv[1], "logs")
    plt.figure(figsize=(10, 6))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    for target_recall in [85, 90]:
        all_data = defaultdict(list)
        for ff in sorted(list(workload_to_files.keys())):
            basename = ff.replace(".txt", "")
            idx = basename.find("_")
            method = basename[:idx]
            basename = basename[idx+1:]
            idx = basename.find("_")
            dataset = basename[:idx]
            basename = basename[idx+1:]

            if dataset not in inspected_dataset:
                continue

            if method.find("pathseer")!=-1:
                idx = basename.find("_")
                basename = basename[idx+1:]
                idx = basename.find("_")
                Mexp = int(basename[:idx])
                search_param = basename[idx+1:]
                # record the best result
                if Mexp==0:
                    if method=="pathseer":
                        if search_param.find("coef_1")!=-1:
                            method = "pathseer-full-opt"
                        else:
                            method = "pathseer-full"
                    # we don't turn on the auto parameter decision for the other workloads
                    else:
                        continue
            elif method!="filtered-hnsw" and method!="acorn" and method!="navix":
                continue

            (recall, tput, ndist, nfilter) = get_avg_tput_recall(sys.argv[1], workload_to_files[ff], target_recall)

            # we record the Mexp, but only useful for the pathseer related method
            key = f"{dataset}-{method}"
            all_data[key].append([tput, recall, ndist, nfilter])

        # we set the order of the methods
        methods = ["filtered-hnsw", "acorn", "navix", "pathseer-exp-only", "pathseer", "pathseer-full"]

        log_path = os.path.join(log_dir, f"incremental_recall_{target_recall}.txt")
        # a dedicated file for original
        log_path2 = os.path.join(log_dir, f"incremental_original_recall_{target_recall}.txt")
        log_path3 = os.path.join(log_dir, f"incremental_table_recall_{target_recall}.txt")
        # using pandas to plot the basic plot
        data = {}
        error_bars = []
        with open(log_path, "w") as ofs, open(log_path2, "w") as ofs2, open(log_path3, "w") as ofs3:

            ofs3.write("Workload\tHNSW-Vbase\tACORN\tNaviX\t+Dynamic Neighbor Traversal\t+Fusion Vector Index\t+Heuristic Parameter Tuning\n")

            for dataset in inspected_dataset:
                ofs.write(f"## dataset {dataset} ##\n")
                ofs.write("Method\tTput(mean)\tTput(min)\tTput(max)\tRecall\n")

                if dataset=="HM":
                    ofs2.write(f"H&M\t")
                    ofs3.write(f"H&M\t")
                elif dataset=="Arxiv":
                    ofs2.write(f"arXiv\t")
                    ofs3.write(f"arXiv\t")
                else:
                    ofs2.write(f"{dataset}\t")
                    ofs3.write(f"{dataset}\t")

                tputs = []
                # maybe we should record the min/max tput
                error_bars.append([])
                tmp_arr = []
                for method in methods:
                    key = f"{dataset}-{method}"
                    if key not in all_data:
                        print(f"method {key} not found")
                        assert False
                    # if there are more than one result, we average them all
                    avg_tput = []
                    avg_recall = []
                    tput_min = 1e100
                    tput_max = 0
                    for item in all_data[key]:
                        avg_tput.append(item[0])
                        avg_recall.append(item[1])

                        tput_min = min(tput_min, item[0])
                        tput_max = max(tput_max, item[0])

                    tput = np.mean(avg_tput)
                    recall = np.mean(avg_recall)

                    ofs.write(f"{method}\t{tput:.2f}\t{tput_min:.2f}\t{tput_max:.2f}\t{recall}\n")
                    ofs2.write(f"{tput}\t{tput-tput_min}\t{tput_max-tput}\t")
                    tputs.append(tput)
                    error_bars[-1].append([tput - tput_min, tput_max - tput])

                    if tput_min==tput and tput_max==tput:
                        ofs3.write(f"{tput:.1f}\t")
                    else:
                        ofs3.write(f"{tput:.1f} ({tput_min:.1f}/{tput_max:.1f})\t")

                ofs2.write("\n")
                ofs3.write("\n")

                data[dataset] = tputs

        plt.clf()
        labels = ["DCF", "MFF", "MFF2", "+hybrid neighbor traversing", "+fusion vector index", "+heuristic parameter selection"]

        df = pd.DataFrame(data, index=labels)

        yerr = np.array(error_bars).transpose(1, 2, 0)

        ax = df.T.plot(kind="bar", yerr=yerr)
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Throughput")

        ax.set_title("Effects of individual techniques")

        plt.xticks()
        plt.tight_layout()

        output_path = os.path.join(result_dir, f"incremental_test_recall_{target_recall}.png")
        plt.savefig(output_path)