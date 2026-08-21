// Compare Array of Structures (AoS) with Structure of Arrays (SoA).

#include <array>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string_view>

constexpr std::size_t kSize = 100'000;
constexpr std::size_t kRepeats = 1'000;

struct AoS {
    int a;
    int b;
    int c;
};

struct SoA {
    std::array<int, kSize> a;
    std::array<int, kSize> b;
    std::array<int, kSize> c;
};

std::uint64_t sum_AoS(const std::array<AoS, kSize>& data) {
    std::uint64_t sum_a = 0;
    std::uint64_t sum_b = 0;
    std::uint64_t sum_c = 0;

    for (const AoS& item : data) {
        sum_a += item.a;
        sum_b += item.b;
        sum_c += item.c;
    }

    return sum_a + sum_b + sum_c;
}

std::uint64_t sum_SoA(const SoA& data) {
    std::uint64_t sum_a = 0;
    std::uint64_t sum_b = 0;
    std::uint64_t sum_c = 0;

    for (int value : data.a) sum_a += value;
    for (int value : data.b) sum_b += value;
    for (int value : data.c) sum_c += value;

    return sum_a + sum_b + sum_c;
}

int main(int argc, char* argv[]) {
    if (argc != 3 || std::string_view(argv[1]) != "--mode") {
        std::cerr << "Usage: " << argv[0] << " --mode aos|soa\n";
        return 1;
    }

    const std::string_view mode = argv[2];

    if (mode == "aos") {
        std::array<AoS, kSize> data{};
        for (std::size_t i = 0; i < kSize; ++i)
            data[i] = {static_cast<int>(i + 1), static_cast<int>(i + 2),
                       static_cast<int>(i + 3)};

        std::uint64_t sum = 0;
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t repeat = 0; repeat < kRepeats; ++repeat) {
            // Prevent the compiler from calculating the sum only once.
            ++data[repeat % kSize].a;
            sum += sum_AoS(data);
        }
        const auto end = std::chrono::steady_clock::now();
        const double time =
            std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << std::fixed << std::setprecision(3)
                  << "Mode: aos\nIntegers: " << kSize * 3
                  << "\nSum: " << sum << "\nTime: " << time
                  << " ms\n";
        return 0;
    }

    if (mode == "soa") {
        SoA data{};
        for (std::size_t i = 0; i < kSize; ++i) {
            data.a[i] = static_cast<int>(i + 1);
            data.b[i] = static_cast<int>(i + 2);
            data.c[i] = static_cast<int>(i + 3);
        }

        std::uint64_t sum = 0;
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t repeat = 0; repeat < kRepeats; ++repeat) {
            ++data.a[repeat % kSize];
            sum += sum_SoA(data);
        }
        const auto end = std::chrono::steady_clock::now();
        const double time =
            std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << std::fixed << std::setprecision(3)
                  << "Mode: soa\nIntegers: " << kSize * 3
                  << "\nSum: " << sum << "\nTime: " << time
                  << " ms\n";
        return 0;
    }

    std::cerr << "Mode must be aos or soa.\n";
    return 1;
}
