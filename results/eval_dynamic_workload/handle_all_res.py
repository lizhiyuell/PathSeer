import re
import matplotlib.pyplot as plt
import os
import sys
from collections import defaultdict
import numpy as np

# this script plot the basic results of all data with one dir (i.e. benchmark), and output the numbers to txt files

def parse_file_content(file_path):
    tput_arr = []
    with open(file_path, "r") as file:
        for line in file:
            if line.startswith("[PointSearch]"):
                tput_arr.append(float(re.search(r"Tput=(\d+\.\d+)\sops", line).group(1)))

    # Separate the results for the two recall targets.
    assert len(tput_arr)%2==0
    tput_r85 = tput_arr[::2]
    tput_r90 = tput_arr[1::2]

    return tput_r85, tput_r90

# Merge results from multiple runs.
def merge_all_res(each_res):
    return np.mean(np.array(each_res), axis=0)

if __name__=="__main__":
    if len(sys.argv) != 2:
        print("Usage: python handle_all_res.py <path_dir>")
        exit(1)
    
    # save the data: simd_method -> dataset -> figure (method)
    all_data = {}
    # Preprocess files by grouping repeated tests of the same workload.
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


        # Analyze each result.
        each_res_r85 = []
        each_res_r90 = []
        for target_name in workload_to_files[ff]:
            # get the result
            target_file = os.path.join(sys.argv[1], target_name)
            if os.path.isdir(target_file):
                continue
            res = parse_file_content(target_file)
            each_res_r85.append(res[0])
            each_res_r90.append(res[1])

        # Merge multiple results.
        res = [merge_all_res(each_res_r85), merge_all_res(each_res_r90)]

        workload_name = f"{method}"

        all_data[workload_name] = res
    
    # the plot order
    methods = ["filtered-ivf", "filtered-hnsw", "acorn", "navix", "pathseer"]

    nr_item = 0

    # plot the result
    result_dir = os.path.join(sys.argv[1], "plots")
    log_dir = os.path.join(sys.argv[1], "logs")
    plt.figure(figsize=(10, 6))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    for idx, item in enumerate(["85", "90"]):
        plt.clf()
        for key in methods:
            value = all_data[key][idx] / 1000
            plt.plot(np.arange(len(value)), value, label=key)
            nr_item = len(value)
        
        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("Throughput (K Query/s)")
        plt.savefig(result_dir + f"/timeline_result_R{item}.png")

        # write-out the result
        with open(log_dir + f"/timeline_result_R{item}.txt", "w") as ofs:
            ofs.write("Iteration")
            for method in methods:
                ofs.write("\t" + method)
            ofs.write("\n")

            for i in range(nr_item):
                ofs.write(str(i))
                for method in methods:
                    ofs.write("\t" + str(all_data[method][idx][i] / 1000))
                ofs.write("\n")



