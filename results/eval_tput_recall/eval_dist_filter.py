import re
import matplotlib.pyplot as plt
import os
import sys
from collections import defaultdict
import pandas as pd
import numpy as np

# this script plot the performance of different methods with incremental techniques

target_recall = 90
def parser_performance(file_path, target_recall):
    res_tput = 0
    res_recall = 0
    res_dist = 0
    res_filter = 0
    second_check = False
    with open(file_path, "r") as ifs:
        for line in ifs:
            if line.startswith("[PointSearch]") or line.startswith("[Result]"):
                try:
                    recall = float(re.search(r"avg_recall=(\d+\.\d+)\s%", line).group(1))
                    tput = float(re.search(r"Tput=(\d+\.\d+)\sops", line).group(1))
                    nfilter = int(re.search(r"avg_filter=(\d+),", line).group(1))
                    ndist = int(re.search(r"avg_dist=(\d+),", line).group(1))
                except:
                    print(file_path, line)
                    assert(False)

                if abs(target_recall - recall) < abs(target_recall - res_recall):
                    res_recall = recall
                    res_tput = tput
                    res_dist = ndist
                    res_filter = nfilter

    return (res_recall, res_tput, res_dist, res_filter)

def get_avg_tput_recall(base_dir, filepath_list):
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

    inspected_dataset = ["SIFT1M", "HM", "LAION"]

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

        if method.find("pathseer")!=-1 and basename.find("Mexp_0")==-1:
            continue

        (recall, tput, ndist, nfilter) = get_avg_tput_recall(sys.argv[1], workload_to_files[ff])

        # we record the Mexp, but only useful for the pathseer related method
        key = f"{dataset}-{method}"
        all_data[key].append([tput, recall, ndist, nfilter])

    # we set the order of the methods
    # methods = ["filtered-ivf", "filtered-hnsw", "acorn", "pathseer"]
    methods = ["filtered-hnsw", "acorn", "pathseer"]

    log_path = os.path.join(log_dir, f"filter_dist.txt")

    with open(log_path, "w") as ofs:
        for dataset in inspected_dataset:
            ofs.write(f"## dataset {dataset} ##\n")
           
            dist_arr = []
            filter_arr = []
            for method in methods:
                key = f"{dataset}-{method}"
                if key not in all_data:
                    print(f"method {key} not found")
                    assert False

                td = []
                tf = []
                for item in all_data[key]:
                    td.append(item[2])
                    tf.append(item[3])

                dist_arr.append(np.mean(td))
                filter_arr.append(np.mean(tf))


            ofs.write("Metric\t")
            for method in methods:
                ofs.write(method + "\t")
            ofs.write("\n")

            metrics = ["#Distance Computation", "#Metadata Filtering"]
            for i, target in enumerate([dist_arr, filter_arr]):
                ofs.write(metrics[i] + "\t")
                for j in range(len(methods)):
                    ofs.write(str(target[j]) + "\t")
                ofs.write("\n")

            ofs.write("\n")

    # plt.clf()
    # # labels = ["vamana", "+hybrid neighbor traversing", "+fusion vector index", "+heuristic parameter selection"]

    # labels = ["DCF", "MFF", "+hybrid neighbor traversing", "+fusion vector index", "+heuristic parameter selection"]

    # df = pd.DataFrame(data, index=labels)

    # yerr = np.array(error_bars).transpose(1, 2, 0)

    # ax = df.T.plot(kind="bar", yerr=yerr)
    # ax.set_xlabel("Dataset")
    # ax.set_ylabel("Throughput")

    # ax.set_title("Effects of individual techniques")

    # plt.xticks()
    # plt.tight_layout()

    # output_path = os.path.join(result_dir, f"incremental_test.png")
    # plt.savefig(output_path)