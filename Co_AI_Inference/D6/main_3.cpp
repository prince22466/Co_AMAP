// Compare false sharing with two counters on the same cache line against
// padded counters on separate cache lines.

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <thread>

constexpr std::size_t kCacheLineSize = 64;
constexpr std::size_t kRepeats = 10'000'000;

// Both counters start on the same cache line.
struct alignas(kCacheLineSize) NormalCounters {
    //alignas(kCacheLineSize) here actually ensure  
    // a, b will on the same 64 bytes cache line
    int a{0}; // each int is 4 bytes 
    int b{0};
};

// Each counter occupies its own cache line.
struct alignas(kCacheLineSize) PaddedCounter {
    int value{0};
};

struct PaddedCounters {
    PaddedCounter a;
    PaddedCounter b;
};

void increment(int& counter) {
    for (std::size_t i = 0; i < kRepeats; ++i)
        ++counter;
}

double run(int& a, int& b) {
    const auto begin = std::chrono::steady_clock::now();

    std::thread first([&] { increment(a); });
    std::thread second([&] { increment(b); });
    first.join();
    second.join();

    const auto end = std::chrono::steady_clock::now();

    return std::chrono::duration<double, std::milli>(end - begin).count();
}

void print_result(std::string_view mode,
                  const int& a,
                  const int& b,
                  double time_ms) {
    std::cout << std::fixed << std::setprecision(3)
              << "Mode: " << mode
              << "\nCounter a: " << a
              << "\nCounter b: " << b
              << "\nTotal: " << a + b
              << "\nTime: " << time_ms << " ms\n";
}

int main(int argc, char* argv[]) {
    if (argc != 3 || std::string_view(argv[1]) != "--mode") {
        std::cerr << "Usage: " << argv[0] << " --mode normal|padded\n";
        return 1;
    }

    const std::string_view mode = argv[2];

    if (mode == "normal") {
        NormalCounters counters;
        const double time = run(counters.a, counters.b);
        print_result(mode, counters.a, counters.b, time);
        return 0;
    }

    if (mode == "padded") {
        PaddedCounters counters;
        const double time = run(counters.a.value, counters.b.value);
        print_result(mode, counters.a.value, counters.b.value, time);
        return 0;
    }

    std::cerr << "Mode must be normal or padded.\n";
    return 1;
}
