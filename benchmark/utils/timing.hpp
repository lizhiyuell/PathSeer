#ifndef _TIMING_HPP_
#define _TIMING_HPP_

#include <chrono>
#include <cstdint>

class Timing {
public:
    // Constructor: record the initialization time point.
    Timing() : start_time_(std::chrono::high_resolution_clock::now()) {}

    // Return elapsed nanoseconds relative to the initial time.
    int64_t get_ns_time() const {
        return get_time<std::chrono::nanoseconds>();
    }

    // Return elapsed microseconds relative to the initial time.
    int64_t get_us_time() const {
        return get_time<std::chrono::microseconds>();
    }

    // Return elapsed milliseconds relative to the initial time.
    int64_t get_ms_time() const {
        return get_time<std::chrono::milliseconds>();
    }

    // Return elapsed seconds relative to the initial time.
    int64_t get_s_time() const {
        return get_time<std::chrono::seconds>();
    }

private:
    // Template helper for elapsed time in different units.
    template <typename Duration>
    int64_t get_time() const {
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<Duration>(now - start_time_);
        return elapsed.count();
    }

    // Initial time point.
    const std::chrono::high_resolution_clock::time_point start_time_;
};

#endif