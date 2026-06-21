#ifndef _STATISTIC_HPP_
#define _STATISTIC_HPP_

#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>

template <typename data_t>
class Statistic {
public:

    Statistic() : amp_value(1) {}

    // Add one data_t value.
    void add(data_t value) {
        data_.push_back(value * amp_value);
    }

    void set_amp(int val){
        if(val != amp_value){
            // After amp_value changes, clear existing records and start over.
            clear();
            amp_value = val;
        }
    }

    // Remove all records.
    void clear() {
        data_.clear();
        amp_value = 1;
    }

    // Get the nr-th percentile of existing values (0 <= nr <= 100).
    data_t get_percentile(double nr) {
        if (data_.empty()) return 0;

        // Copy and sort the data.
        std::vector<data_t> sorted_data = data_;
        std::sort(sorted_data.begin(), sorted_data.end());

        // Calculate the percentile index.
        double pos = (nr / 100.0) * (sorted_data.size() - 1);
        size_t idx = static_cast<size_t>(pos);
        double frac = pos - idx;

        // Interpolate the percentile value.
        if (idx + 1 < sorted_data.size()) {
            return (data_t)(sorted_data[idx] + frac * (sorted_data[idx + 1] - sorted_data[idx]));
        } else {
            return (data_t)(sorted_data[idx]);
        }
    }

    // Get the mean of existing values.
    data_t get_mean() {
        if (data_.empty()) return 0;

        double sum = 0.0;
        for (const auto& val : data_) {
            sum += val;
        }
        return (data_t)(sum / data_.size());
    }

    data_t get_min() {
        return (data_t) data_[0];
    }

    data_t get_max() {
        return (data_t) data_[this->size()-1];
    }

    // Get the standard deviation of existing values.
    data_t get_std() {
        if (data_.empty()) return 0;

        double mean = get_mean();
        double sum_sq_diff = 0.0;
        for (const auto& val : data_) {
            sum_sq_diff += (val - mean) * (val - mean);
        }
        return (data_t)(std::sqrt(sum_sq_diff / data_.size()));
    }

    void sort(){
        std::sort(data_.begin(), data_.end());
    }

    // Check whether data has been added; do nothing if empty.
    bool has_data(){
        return !data_.empty();
    }

    int size(){
        return data_.size();
    }

    const data_t operator[](size_t index) const {
        if (index >= data_.size()) {
            throw std::out_of_range("Index out of range");
        }
        return data_[index];
    }

private:
    std::vector<data_t> data_;
    int amp_value;  // Scale factor applied to all returned results.
};

#endif