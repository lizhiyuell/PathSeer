
MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PROJECT_ROOT := $(abspath $(MAKEFILE_DIR)/..)
FAISS_ROOT := $(PROJECT_ROOT)/faiss
FAISS_BUILD_DIR := $(FAISS_ROOT)/build/faiss

# List of directories containing test scripts
bench_dirs := build_persist_index eval_diff_selectivity eval_tput_recall pathseer_virtual_overhead_profilier eval_dynamic_workload

# Common source files in each directory
sources := acorn.cpp filtered-hnsw.cpp pathseer.cpp filtered-ivf.cpp navix.cpp

# Compiler
CXX := g++

COMMON_FLAGS := -g -std=c++1z

# Compilation flags for each specific file
FAISS_FLAGS := -I$(FAISS_ROOT) -L$(FAISS_BUILD_DIR) -Wl,-rpath,$(FAISS_BUILD_DIR) -lfaiss -fopenmp $(COMMON_FLAGS) -march=native

ACORN_FLAGS := $(FAISS_FLAGS)

FILTERED_HNSW_FLAGS := $(FAISS_FLAGS)

NAVIX_FLAGS := $(FAISS_FLAGS)

FILTERED_IVF_FLAGS := $(FAISS_FLAGS)

PATHSEER_FLAGS := $(FAISS_FLAGS)

# Generate a list of all binaries
binaries := $(foreach dir,$(bench_dirs),$(foreach src,$(sources),$(if $(wildcard $(dir)/$(src)),$(dir)/$(basename $(src)))))

sensitivity_binaries := eval_params_sensitivity/eval_M_sensitivity eval_params_sensitivity/eval_M_exp_sensitivity

# Default target
all: $(binaries) $(sensitivity_binaries)

# Rule for building each source file with specific flags
$(foreach dir,$(bench_dirs), \
    $(eval $(dir)/acorn: $(dir)/acorn.cpp ; \
        $(CXX) -o $$@ $$< $(ACORN_FLAGS)) \
    $(eval $(dir)/filtered-hnsw: $(dir)/filtered-hnsw.cpp ; \
        $(CXX) -o $$@ $$< $(FILTERED_HNSW_FLAGS)) \
    $(eval $(dir)/pathseer: $(dir)/pathseer.cpp ; \
        $(CXX) -o $$@ $$< $(PATHSEER_FLAGS)) \
	$(eval $(dir)/filtered-ivf: $(dir)/filtered-ivf.cpp ; \
        $(CXX) -o $$@ $$< $(FILTERED_IVF_FLAGS)) \
	$(eval $(dir)/navix: $(dir)/navix.cpp ; \
        $(CXX) -o $$@ $$< $(NAVIX_FLAGS)) \
)

eval_params_sensitivity/eval_M_sensitivity: eval_params_sensitivity/eval_M_sensitivity.cpp
	$(CXX) -fopenmp -o $@ $^ $(PATHSEER_FLAGS) -march=native

eval_params_sensitivity/eval_M_exp_sensitivity: eval_params_sensitivity/eval_M_exp_sensitivity.cpp
	$(CXX) -fopenmp -o $@ $^ $(PATHSEER_FLAGS) -march=native


# Clean target to remove all generated binaries
clean:
	@$(foreach dir,$(bench_dirs), rm -f $(dir)/acorn $(dir)/filtered-hnsw $(dir)/vbase $(dir)/pathseer $(dir)/filtered-ivf $(dir)/navix)
	@rm -f $(sensitivity_binaries)

# Phony targets
.PHONY: all clean
