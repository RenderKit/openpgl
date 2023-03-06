#pragma once
#include <chrono>

class Timer {
private:
    using clock = std::chrono::high_resolution_clock;
    using time_point = clock::time_point;

public:
    Timer() {
        reset();
    }

    void reset() {
        start = clock::now();
    }

    double elapsed() {
        time_point end = clock::now();
        std::chrono::duration<double, std::micro> diff = end - start;
        return diff.count();
    }

private:
    time_point start;
};