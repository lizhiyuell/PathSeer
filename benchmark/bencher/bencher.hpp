#ifndef __BENCHER_HPP__
#define __BENCHER_HPP__

#include "../utils/timing.hpp"
#include "../utils/recall.hpp"
#include "../utils/statistic.hpp"
#include "../utils/io.hpp"
#include "../utils/load.hpp"
#include "../utils/print.hpp"
#include "../utils/logger.hpp"
#include "../utils/selector.hpp"
#include <string>
#include <iostream>
#include <unordered_map>
#include <omp.h>

enum query_method {
    QUERY_NO_FILTER,
    QUERY_METHOD_BIT_VECTOR,
    QUERY_METHOD_INT_ARRAY,
    QUERY_METHOD_INT_ARRAY_SLEEP,
    QUERY_METHOD_MULTI_FILTER,
    QUERY_METHOD_COMPOSITE_FILTER,
    QUERY_METHOD_REGEX,
    QUERY_METHOD_FUZZY_FILTER,
    QUERY_METHOD_INT_LARGER,
    QUERY_METHOD_FLOAT_LOWER
};

class Bencher {
public:
    // Constructor.
    Bencher() {}
    ~Bencher() {}

    virtual void build(void* buildParameter) = 0;

    // the return value is the recall rate
    virtual float query(void* searchParameter) = 0;

    // batch query with openMP, and the returned value is the average recall rate
    virtual float batch_query(void* searchParameter) { return 0; };

    // clear at the end of each build
    virtual void clear() = 0;

    virtual void load_persist(char* persist_path) = 0;
    virtual void save_persist(char* persist_path) = 0;

    Timing timing;
    Statistic<int> stat_latency;
    Statistic<double> stat_recall;
    Statistic<int> stat_dist_comp;
    Statistic<int> stat_dist_comp_first_stage;
    Statistic<int> stat_nr_step;
    Statistic<int> stat_nr_step_first_stage;
    Statistic<int> stat_nr_filter;
    Statistic<int> stat_nr_filter_first_stage;
    Recall<int64_t> recall;
    Logger *logger{nullptr};

    // original string-format metadata
    std::vector<std::string> query_metadata;

    // load original metadata
    void set_metadata(std::string metadata_path){
        query_metadata.clear();
        std::ifstream infile(metadata_path);
        if(!infile){
            std::cerr << "Error opening metadata file: " << metadata_path << std::endl;
            assert(false);
        }
        std::string line;
        while (std::getline(infile, line)) {
            query_metadata.push_back(line);
        }
    }

    void clear_stats(){
        stat_latency.clear();
        stat_recall.clear();
        stat_dist_comp.clear();
        stat_dist_comp_first_stage.clear();
        stat_nr_filter.clear();
        stat_nr_filter_first_stage.clear();
        stat_nr_step.clear();
        stat_nr_step_first_stage.clear();
    }

    void set_logger(Logger *logger){
        this->logger = logger;
    }

    void print_results(){

        std::vector<std::string> output_strings;

        output_strings.push_back("MeanTime(us): " + std::to_string(stat_latency.get_mean())
                + ", StdTime(us): " + std::to_string(stat_latency.get_std())
                + ", MedianTime(us): " + std::to_string(stat_latency.get_percentile(50))
                + ", P90Time(us): " + std::to_string(stat_latency.get_percentile(90))
                + ", P95Time(us): " + std::to_string(stat_latency.get_percentile(95))
                + ", P99Time(us): " + std::to_string(stat_latency.get_percentile(99))
                + ", P999Time(us): " + std::to_string(stat_latency.get_percentile(99.9)));

        output_strings.push_back("MeanRecall: " + std::to_string(stat_recall.get_mean())
                + ", StdRecall: " + std::to_string(stat_recall.get_std())
                + ", MedianRecall: " + std::to_string(stat_recall.get_percentile(50))
                + ", P0.1Recall: " + std::to_string(stat_recall.get_percentile(0.1))
                + ", P1Recall: " + std::to_string(stat_recall.get_percentile(1))
                + ", P5Recall: " + std::to_string(stat_recall.get_percentile(5))
                + ", P10Recall: " + std::to_string(stat_recall.get_percentile(10)));
        
        output_strings.push_back("MeanDistCompS1: " + std::to_string(stat_dist_comp_first_stage.get_mean())
                + ", StdDistCompS1: " + std::to_string(stat_dist_comp_first_stage.get_std())
                + ", MedianDistCompS1: " + std::to_string(stat_dist_comp_first_stage.get_percentile(50))
                + ", P90DistCompS1: " + std::to_string(stat_dist_comp_first_stage.get_percentile(90))
                + ", P95DistCompS1: " + std::to_string(stat_dist_comp_first_stage.get_percentile(95))
                + ", P99DistCompS1: " + std::to_string(stat_dist_comp_first_stage.get_percentile(99)));
        
        output_strings.push_back("MeanDistComp: " + std::to_string(stat_dist_comp.get_mean())
                + ", StdDistComp: " + std::to_string(stat_dist_comp.get_std())
                + ", MedianDistComp: " + std::to_string(stat_dist_comp.get_percentile(50))
                + ", P90DistComp: " + std::to_string(stat_dist_comp.get_percentile(90))
                + ", P95DistComp: " + std::to_string(stat_dist_comp.get_percentile(95))
                + ", P99DistComp: " + std::to_string(stat_dist_comp.get_percentile(99)));

        output_strings.push_back("MeanNrStepS1: " + std::to_string(stat_nr_step_first_stage.get_mean())
                + ", StdNrStepS1: " + std::to_string(stat_nr_step_first_stage.get_std())
                + ", MedianNrStepS1: " + std::to_string(stat_nr_step_first_stage.get_percentile(50))
                + ", P90NrStepS1: " + std::to_string(stat_nr_step_first_stage.get_percentile(90))
                + ", P95NrStepS1: " + std::to_string(stat_nr_step_first_stage.get_percentile(95))
                + ", P99NrStepS1: " + std::to_string(stat_nr_step_first_stage.get_percentile(99)));


        output_strings.push_back("MeanNrStep: " + std::to_string(stat_nr_step.get_mean())
                + ", StdNrStep: " + std::to_string(stat_nr_step.get_std())
                + ", MedianNrStep: " + std::to_string(stat_nr_step.get_percentile(50))
                + ", P90NrStep: " + std::to_string(stat_nr_step.get_percentile(90))
                + ", P95NrStep: " + std::to_string(stat_nr_step.get_percentile(95))
                + ", P99NrStep: " + std::to_string(stat_nr_step.get_percentile(99)));

        output_strings.push_back("MeanNrFilterS1: " + std::to_string(stat_nr_filter_first_stage.get_mean())
                + ", StdNrFilterS1: " + std::to_string(stat_nr_filter_first_stage.get_std())
                + ", MedianNrFilterS1: " + std::to_string(stat_nr_filter_first_stage.get_percentile(50))
                + ", P90NrFilterS1: " + std::to_string(stat_nr_filter_first_stage.get_percentile(90))
                + ", P95NrFilterS1: " + std::to_string(stat_nr_filter_first_stage.get_percentile(95))
                + ", P99NrFilterS1: " + std::to_string(stat_nr_filter_first_stage.get_percentile(99)));

        output_strings.push_back("MeanNrFilter: " + std::to_string(stat_nr_filter.get_mean())
                + ", StdNrFilter: " + std::to_string(stat_nr_filter.get_std())
                + ", MedianNrFilter: " + std::to_string(stat_nr_filter.get_percentile(50))
                + ", P90NrFilter: " + std::to_string(stat_nr_filter.get_percentile(90))
                + ", P95NrFilter: " + std::to_string(stat_nr_filter.get_percentile(95))
                + ", P99NrFilter: " + std::to_string(stat_nr_filter.get_percentile(99)));


        for (auto output_string : output_strings){
            if(logger){
                logger->logging(
                    output_string
                );
            }
            else
                printf("%s\n", output_string.c_str());
        }

        fflush(stdout);
        if(logger)
            logger->flush_log();

    }

};


#endif