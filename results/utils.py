import re
from collections import defaultdict
import os
import matplotlib.pyplot as plt

def extract_search_param(line):
    pattern = r"sleep_ns=(\d+)"
    res = re.search(pattern, line)
    if not res:
        print(line)
        assert False

    assert res
    return int(res.group(1))

def extract_time(line):
    pattern = r"MeanTime\(us\): (\d+)"
    res = re.search(pattern, line)
    assert res
    return int(res.group(1))

def extract_recall(line):
    pattern = r"MeanRecall: (\d+\.\d+)"
    res = re.search(pattern, line)
    assert res
    return float(res.group(1))
