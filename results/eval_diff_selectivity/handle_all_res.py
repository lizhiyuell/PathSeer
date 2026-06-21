# This script outputs result comparisons at specific recall values.

import re
import matplotlib.pyplot as plt
import os
import sys
from collections import defaultdict
import numpy as np

# for a certain log file, parser the performance whose recall nearest to a certain value
target_recall = 90
def parser_performance(file_path, target_recall):
    res_tput = 0
    res_recall = 0
    with open(file_path, "r") as ifs:
        for line in ifs:
            if line.startswith("[Result]") or line.startswith("[PointSearch]"):
                recall = float(re.search(r"avg_recall=(\d+\.\d+)\s%", line).group(1))
                tput = float(re.search(r"Tput=(\d+\.\d+)\sops", line).group(1))

                if abs(target_recall - recall) < abs(target_recall - res_recall):
                    res_recall = recall
                    res_tput = tput

    return (res_recall, res_tput)

if __name__=="__main__":
    if len(sys.argv) != 2:
        print("Usage: python handle_all_res.py <path_dir>")
        exit(1)

    # merge and handle multiple results
    workload_to_files = defaultdict(list)
    for ff in sorted(os.listdir(sys.argv[1])):
        if not ff.endswith(".txt"):
            continue

        # # skip the first experiment
        # if ff.find("0:")!=-1:
        #     continue

        pos = ff.find(":")
        if pos!=-1:
            fname = ff[pos+1:]
        else:
            fname = ff
        workload_to_files[fname].append(ff)

    possible_dataset = ["SIFT1M"]
    possible_sleep_val = set()
    possible_methods = ["filtered-ivf", "filtered-hnsw", "acorn", "navix", "pathseer"] # we let the methods plot in certain order
    possible_max_val = [2, 4, 8, 16, 32, 64]

    all_data = defaultdict(list)
    # sort the file names for unified access order
    for ff in sorted(list(workload_to_files.keys())):
        basename = ff.replace(".txt", "")
        idx = basename.find("_")
        method = basename[:idx]
        basename = basename[idx+1:]
        idx = basename.find("_")
        dataset = basename[:idx]
        basename = basename[idx+1:]
        search_param = basename

        assert method in possible_methods

        # Extract max and sleep values from params.
        pos = search_param.find("_")
        search_param = search_param[pos+1:]
        pos = search_param.find("_")
        max_val = int(search_param[:pos])
        if max_val not in possible_max_val:
            continue
        search_param = search_param[pos+1:]

        pos = search_param.rfind("_")
        sleep_val = int(search_param[pos+1:])
        search_param = search_param[:pos]
        pos = search_param.rfind("_")
        search_param = search_param[:pos]

        assert dataset in possible_dataset
        # assert method in possible_methods
        possible_sleep_val.add(sleep_val)

        # analyze of each experiment and get the average result
        each_res = [[], []] # [recall, tput]
        for target_name in workload_to_files[ff]:
            # get the result
            target_file = os.path.join(sys.argv[1], target_name)
            if os.path.isdir(target_file):
                continue
            (res_recall, res_tput) = parser_performance(target_file, target_recall)
            each_res[0].append(res_recall)
            each_res[1].append(res_tput)

        # merge multiple results
        avg_recall = np.mean(each_res[0])
        avg_tput = np.mean(each_res[1])

        workload_name = f"{dataset}-{method}-{sleep_val}"

        # for each workload & Rp, list the performance of Sp-tput, record the avg recall
        all_data[workload_name].append((max_val, avg_tput, avg_recall))
    
    # plot the result
    result_dir = os.path.join(sys.argv[1], "plots")
    log_dir = os.path.join(sys.argv[1], "logs")
    plt.figure(figsize=(10, 6))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # plot the result for each sleep val respectively
    for dataset in possible_dataset:
        for sleep_val in sorted(list(possible_sleep_val)):
            # prepare the plot
            plt.clf()
            log_path = os.path.join(log_dir, f"dataset_{dataset}-sleep_val_{sleep_val}-target_recall{target_recall}.txt")

            with open(log_path, "w") as ofs:

                # record the result for easier plot
                all_plot = []
                for val in possible_max_val:
                    all_plot.append([f"1/{val}"])

                # for each system, plot the result
                for method in possible_methods:
                    # if method.find("pathseer")==-1:
                    #     continue
                    workload_name = f"{dataset}-{method}-{sleep_val}"

                    ofs.write(f"Method: {method}\n")

                    [val_arr, tput_arr, recall_arr] = list(zip(*sorted(all_data[workload_name], key=lambda x: x[0])))

                    labels = [f"1/{val}" for val in val_arr]
                    plt.plot(range(len(val_arr)), tput_arr, label=method)

                    plt.xticks(range(len(val_arr)), labels)
                    plt.xlabel("Sp")
                    plt.ylabel("Tput (OPS)")
                    plt.title(f"Comparing of Sp-tput of {dataset}, sleep={sleep_val}, recall={target_recall}")
                    plt.legend()
                    # plt.yscale("log")
                    # plt.ylim(0)

                    output_path = os.path.join(result_dir, f"{dataset}_{sleep_val}_R{target_recall}.png")
                    plt.savefig(output_path)

                    # output the log
                    ofs.write(f"Dataset: {dataset}, Sleep_val: {sleep_val}, Recall: {target_recall}\n")

                    ofs.write("Sp\tTput\trecall\n")

                    for i in range(len(val_arr)):
                        ofs.write(f"{labels[i]}\t{tput_arr[i]}\t{recall_arr[i]}\n")

                        all_plot[i].append(str(tput_arr[i]))
                    
                    ofs.write("\n")
                
                ofs.write("\nTotal result:\nSp\t")
                for method in possible_methods:
                    ofs.write(method + "\t")
                ofs.write("\n")

                for i in all_plot:
                    for j in i:
                        ofs.write(j + "\t")
                    ofs.write("\n")