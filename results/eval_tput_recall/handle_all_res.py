import re
import matplotlib.pyplot as plt
import os
import sys
from collections import defaultdict

# this script plot the basic results of all data with one dir (i.e. benchmark), and output the numbers to txt files

def parse_file_content(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    res = {
        "dist": [],
        "filter": [],
        "tput": [],
        "recall": [],
        "p90tput": [],
        "p90recall": [],
        "p85tput": [],
        "p85recall": []
    }

    res_tput_90 = 0
    res_tput_85 = 0
    res_recall_90 = 0
    res_recall_85 = 0

    for line in lines:
        if line.startswith("[Result]"):
            res["tput"].append(float(re.search(r"Tput=(\d+\.\d+)\sops", line).group(1)))
            res["recall"].append(float(re.search(r"avg_recall=(\d+\.\d+)\s%", line).group(1)))
            res["dist"].append(int(re.search(r"avg_dist=(\d+)", line).group(1)))
            res["filter"].append(int(re.search(r"avg_filter=(\d+)", line).group(1)))
        
        if line.startswith("[PointSearch]") or line.startswith("[Result]"):
            recall = float(re.search(r"avg_recall=(\d+\.\d+)\s%", line).group(1))
            tput = float(re.search(r"Tput=(\d+\.\d+)\sops", line).group(1))
            if abs(90 - recall) < abs(90 - res_recall_90):
                res_recall_90 = recall
                res_tput_90 = tput

            if abs(85 - recall) < abs(85 - res_recall_85):
                res_recall_85 = recall
                res_tput_85 = tput

    res["p90tput"].append(res_tput_90)
    res["p90recall"].append(res_recall_90)
    res["p85tput"].append(res_tput_85)
    res["p85recall"].append(res_recall_85)

    return res

# Merge results from multiple runs.
def merge_all_res(each_res):
    nr = len(each_res)
    assert nr>=1
    res = each_res[0]
    if nr != 1:
        for key in res.keys():
            # we cut the result length to the minimal one
            min_lenghth = len(res[key])
            for i in range(1, nr):
                other_res = each_res[i]
                min_lenghth = min(min_lenghth, len(other_res[key]))
                # summarize the result
                for j in range(min_lenghth):
                    res[key][j] += other_res[key][j]
            # divide
            for i in range(min_lenghth):
                res[key][i] /= nr

            # maybe we should shrink it
            res[key] = res[key][:min_lenghth]
    
    return res

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

        # exclude the results of fix-Mexp pathseer and the incremental methods
        # if ((method.find("pathseer")!=-1 and (method!="pathseer") and method!="pathseer-opt")) or (method == "pathseer" and search_param.find("Mexp_0")==-1):
        if (method.find("pathseer")!=-1 and method!="pathseer" and method!="pathseer-opt") or (method == "pathseer" and search_param.find("Mexp_0")==-1):
            continue

        # Analyze each result.
        each_res = []
        for target_name in workload_to_files[ff]:
            # get the result
            target_file = os.path.join(sys.argv[1], target_name)
            if os.path.isdir(target_file):
                continue
            res = parse_file_content(target_file)
            each_res.append(res)

        # Merge multiple results.
        res = merge_all_res(each_res)
        res["label"] = f"{method}"

        workload_name = f"{dataset}-{method}"

        all_data[workload_name] = res
    
    # the plot order
    methods = ["filtered-ivf", "filtered-hnsw", "acorn", "navix", "pathseer"]

    datasets = ["SIFT1M", "HM", "LAION", "Paper", "Arxiv", "Amazon"]

    # plot the result
    result_dir = os.path.join(sys.argv[1], "plots")
    log_dir = os.path.join(sys.argv[1], "logs")
    plt.figure(figsize=(10, 6))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # P90 res dir
    with open(log_dir + "/recall_90_tput.txt", "w") as ofs_90, open(log_dir + "/recall_85_tput.txt", "w") as ofs_85:
        y_label_names = ["Nr_dist_comp", "Nr_filter", "Tput (OPS)"]
        for dataset in datasets:
            # output the text data
            # plot the result
            for ii, y_label in enumerate(["dist", "filter", "tput"]):
                plt.clf()
                log_path = os.path.join(log_dir, f"{dataset}-recall-{y_label}.txt")
                with open(log_path, "w") as ofs:
                    for method in methods:
                        key = f"{dataset}-{method}"
                        target = all_data[key]

                        plt.plot(target["recall"], target[y_label], label=target["label"])

                        # write the log
                        ofs.write("{}({})\t".format(y_label, target["label"]))
                        for jj in range(len(target["recall"])):
                            ofs.write("{}\t".format(target[y_label][jj]))
                        ofs.write("\n")
                        ofs.write("Recall({})\t".format(target["label"]))
                        for jj in range(len(target["recall"])):
                            ofs.write("{}\t".format(target["recall"][jj]))
                        ofs.write("\n")
    
                    plt.xlabel("MeanRecall")
                    plt.ylabel(y_label_names[ii])
                    plt.title(f"Comparing of recall-{y_label}")
                    plt.legend()
                    plt.yscale("log")

                    output_path = os.path.join(result_dir, f"{dataset}_recall-{y_label}.png")
                    plt.savefig(output_path)
                    print(f"Result figure saved to: {output_path}")
                
            # write the P90 tput of all methods
            ofs_90.write(f"## dataset: {dataset}\n")
            ofs_90.write(f"Method\tTput\tRecall\n")
            for method in methods:
                key = f"{dataset}-{method}"
                target = all_data[key]
                ofs_90.write("{}\t{}\t{}\n".format(target["label"], target["p90tput"][0], target["p90recall"][0]))

            # write the P85 tput of all methods
            ofs_85.write(f"## dataset: {dataset}\n")
            ofs_85.write(f"Method\tTput\tRecall\n")
            for method in methods:
                key = f"{dataset}-{method}"
                target = all_data[key]
                ofs_85.write("{}\t{}\t{}\n".format(target["label"], target["p85tput"][0], target["p85recall"][0]))


