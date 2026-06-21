# PathSeer

PathSeer is a vector retrieval system with attribute filtering. The implementation is built on top of FAISS and includes benchmark programs for evaluating PathSeer and related filtered ANN baselines.

# Directory Structure

- `faiss/`: The FAISS-based implementation used by PathSeer and the baselines. PathSeer is implemented in `faiss/faiss/IndexPathSeer.*`; related filtered ANN implementations such as ACORN, Filtered-HNSW, and NaviX are also included under `faiss/faiss/`.
- `benchmark/`: C++ benchmark programs. These programs load datasets, build or load indexes, run search workloads, and write raw benchmark logs.
- `benchmark/bencher/`: Wrapper classes that provide a unified build/search interface for different indexing methods.
- `benchmark/utils/`: Shared utilities for loading data, metadata filtering, logging, timing, statistics, recall calculation, and other benchmark helpers.
- `benchmark/build_persist_index/`: Programs for building persistent indexes.
- `benchmark/eval_tput_recall/`: Programs for throughput-recall evaluation.
- `benchmark/eval_diff_selectivity/`: Programs for evaluating performance under different filter selectivities.
- `benchmark/eval_dynamic_workload/`: Programs for dynamic workload evaluation.
- `benchmark/eval_params_sensitivity/`: Programs for PathSeer parameter sensitivity evaluation.
- `benchmark/pathseer_virtual_overhead_profilier/`: Programs for profiling PathSeer virtual overhead.
- `results/`: Scripts and generated outputs for post-processing benchmark results. Raw benchmark logs are expected to be written into subdirectories under `results/`.
- `dataset/`: Expected location of benchmark datasets. This directory is not included in the repository because datasets are usually large. Each dataset should contain `load/`, `query/`, and `golden/` subdirectories.
- `indexes/`: Expected location of built persistent indexes. This directory is not included in the repository because generated indexes are usually large.

# Dataset Download

The benchmark datasets are hosted on Hugging Face:

https://huggingface.co/datasets/lizhiyuell/filtered_ANNS_dataset

Download the dataset and place it under the repository root as `dataset/`. For example:

```bash
huggingface-cli download lizhiyuell/filtered_ANNS_dataset \
  --repo-type dataset \
  --local-dir dataset
```

After downloading, the expected layout is:

```text
dataset/
  <dataset_name>/
    load/
    query/
    golden/
```

# Compilation

1. Build FAISS.

   ```bash
   cd faiss
   ./remake.sh
   ```

2. Build the benchmarks.

   ```bash
   cd benchmark
   make -j
   ```

# Benchmark Usage

Benchmark executables are generated under their corresponding subdirectories in `benchmark/`. Each benchmark program uses paths relative to its own directory, so please enter the target benchmark directory before running it.

The main benchmark suites are:

- `benchmark/build_persist_index/`: build and save persistent indexes under `indexes/`.
- `benchmark/eval_tput_recall/`: evaluate throughput-recall trade-offs.
- `benchmark/eval_diff_selectivity/`: evaluate performance under different predicate selectivities.
- `benchmark/eval_dynamic_workload/`: evaluate dynamic workloads.
- `benchmark/eval_params_sensitivity/`: evaluate PathSeer parameter sensitivity.
- `benchmark/pathseer_virtual_overhead_profilier/`: profile PathSeer virtual overhead.

To run a benchmark suite, enter the directory and execute its `run.sh` script:

```bash
cd benchmark/build_persist_index
./run.sh
```

Similarly, other suites can be run with:

```bash
cd benchmark/eval_tput_recall
./run.sh

cd benchmark/eval_diff_selectivity
./run.sh

cd benchmark/eval_dynamic_workload
./run.sh

cd benchmark/eval_params_sensitivity
./run.sh

cd benchmark/pathseer_virtual_overhead_profilier
./run.sh
```

The benchmark scripts are simple entry points. You may edit the corresponding C++ files or `run.sh` scripts to select datasets, methods, and parameter settings.

# PathSeer Virtual Overhead Profiling

PathSeer uses a virtual overhead boundary for the third technical component described in the paper. Before running PathSeer search benchmarks, first build the persistent indexes, then run the PathSeer virtual-overhead profiler and analyze its output.

Run the profiler:

```bash
cd benchmark/pathseer_virtual_overhead_profilier
./run.sh
```

Then run the analysis script under `results/`:

```bash
cd results/pathseer_virtual_overhead_profilier
python analyze_best_virtual_overhead.py .
```

The analysis script generates the best virtual overhead boundary files under `results/pathseer_virtual_overhead_profilier/logs/`. PathSeer benchmark programs read these files when setting the virtual overhead boundary, so this step is required before running PathSeer-related benchmark cases.

Note that PathSeer requires **the virtual overhead boundary to be profiled separately for each $M_1$**, while different $M_2$ values can share the same virtual overhead boundary. You can run PathSeer without profiling the virtual overhead boundary by **setting a fixed $M_{exp}$** for PathSeer, which does not require a virtual overhead boundary. A good example can be seen in ```benchmark/eval_tput_recall/pathseer.cpp```. Setting $M_{exp}$ to non-zero value will trigger fixed-parameter evaluation that only exploits the first two techniques of PathSeer, while setting $M_{exp}$ to zero will make PathSeer find the profiled virtual overhead boundary automatically in the relevant file.

# Results

Benchmark logs and generated outputs are written automatically under `results/`, usually in the subdirectory with the same name as the benchmark suite. For example:

- `results/eval_tput_recall/`
- `results/eval_diff_selectivity/`
- `results/eval_dynamic_workload/`
- `results/pathseer_virtual_overhead_profilier/`

The repository also provides analysis scripts under `results/` for post-processing raw benchmark logs, extracting metrics, and generating summarized result files or figures. Run the analysis scripts from their corresponding result subdirectories after the benchmark logs have been generated.
