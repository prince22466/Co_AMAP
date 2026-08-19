#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#if defined(__linux__)
#include <fstream>
#include <sys/resource.h>
#include <unistd.h>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif

constexpr std::size_t kRequestCount = 100'000;
constexpr std::size_t kProcessingIterations = 1;

struct LargeRequest {
    std::uint64_t id;
    std::string prompt;
    std::string user_id;
    std::string model_name;
    std::vector<int> token_ids;
    double deadline;
    int priority;
    int result;
};

struct CompactReq {
    std::uint64_t id;
    std::uint32_t prompt_tokens;
    std::uint32_t expected_output_tokens;
    std::uint32_t user_id;
    std::uint16_t model_id;
    std::uint16_t priority;
    double deadline;
    int result;
};

namespace {

enum class Mode { large, compact };
std::vector<LargeRequest> large_requests;
std::vector<CompactReq> compact_requests;

std::string repeated_string(std::string_view prefix, std::uint64_t id,
                            std::size_t minimum_length) {
    std::string value(prefix);
    value += std::to_string(id);
    while (value.size() < minimum_length) value += "_benchmark_data";
    value.resize(minimum_length);
    return value;
}

std::size_t resident_memory_bytes() {
#if defined(__linux__)
    std::ifstream statm("/proc/self/statm");
    std::size_t total_pages = 0, resident_pages = 0;
    if (statm >> total_pages >> resident_pages) {
        const long page_size = ::sysconf(_SC_PAGESIZE);
        if (page_size > 0)
            return resident_pages * static_cast<std::size_t>(page_size);
    }
#elif defined(_WIN32)
    PROCESS_MEMORY_COUNTERS counters{};
    if (::GetProcessMemoryInfo(::GetCurrentProcess(), &counters, sizeof(counters)))
        return static_cast<std::size_t>(counters.WorkingSetSize);
#endif
    return 0;
}

std::size_t peak_memory_bytes() {
#if defined(__linux__)
    rusage usage{};
    if (::getrusage(RUSAGE_SELF, &usage) == 0)
        return static_cast<std::size_t>(usage.ru_maxrss) * 1024U;
#elif defined(_WIN32)
    PROCESS_MEMORY_COUNTERS counters{};
    if (::GetProcessMemoryInfo(::GetCurrentProcess(), &counters, sizeof(counters)))
        return static_cast<std::size_t>(counters.PeakWorkingSetSize);
#endif
    return 0;
}

double to_mib(std::size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

int folded_result(std::uint64_t value) {
    return static_cast<int>(value & static_cast<std::uint64_t>(
                                       std::numeric_limits<int>::max()));
}

void print_usage(const char* program) {
    std::cerr << "Usage: " << program
              << " --mode large|compact [--count N] [--iterations N]"
                 "\nDefault request count: "
              << kRequestCount << "\nDefault processing iterations: "
              << kProcessingIterations << '\n';
}

bool parse_count(std::string_view text, std::size_t& result) {
    if (text.empty() || text.front() == '-') return false;
    const char* first = text.data();
    const char* last = first + text.size();
    const auto parsed = std::from_chars(first, last, result);
    return parsed.ec == std::errc{} && parsed.ptr == last;
}

}  // namespace

void Make_vector_req(Mode mode, std::size_t count) {
    if (mode == Mode::large) {
        large_requests.clear();
        large_requests.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            LargeRequest request{};
            request.id = i + 1;
            // These lengths intentionally exceed typical small-string storage.
            request.prompt = repeated_string("prompt_", request.id, 96);
            request.user_id = repeated_string("user_", request.id, 32);
            request.model_name = repeated_string("model_", request.id, 24);
            request.token_ids.resize(32);
            for (std::size_t token = 0; token < request.token_ids.size(); ++token)
                request.token_ids[token] =
                    static_cast<int>((request.id + token * 17U) % 32'000U);
            request.deadline = 1'700'000'000.0 + static_cast<double>(i % 10'000U);
            request.priority = static_cast<int>(i % 8U);
            request.result = 0;
            large_requests.push_back(std::move(request));
        }
        return;
    }

    compact_requests.clear();
    compact_requests.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
        CompactReq& request = compact_requests[i];
        request.id = i + 1;
        request.prompt_tokens = 64U + static_cast<std::uint32_t>(i % 1'984U);
        request.expected_output_tokens = 32U + static_cast<std::uint32_t>(i % 480U);
        request.user_id = 10'000U + static_cast<std::uint32_t>(i % 100'000U);
        request.model_id = static_cast<std::uint16_t>(i % 16U);
        request.priority = static_cast<std::uint16_t>(i % 8U);
        request.deadline = 1'700'000'000.0 + static_cast<double>(i % 10'000U);
        request.result = 0;
    }
}

void Func_LargeReq() {
    for (LargeRequest& request : large_requests) {
        // These reads follow four pointers out of the contiguous request vector
        // into separately allocated heap blocks. The id-dependent string offsets
        // prevent the benchmark from repeatedly touching only the first byte.
        std::uint64_t sum = request.id +
                            static_cast<std::uint64_t>(request.deadline) +
                            static_cast<std::uint64_t>(request.priority);
        sum += static_cast<unsigned char>(
            request.prompt[request.id % request.prompt.size()]);
        sum += static_cast<unsigned char>(
            request.user_id[request.id % request.user_id.size()]);
        sum += static_cast<unsigned char>(
            request.model_name[request.id % request.model_name.size()]);

        const std::size_t last = request.token_ids.size() - 1;
        sum += static_cast<std::uint32_t>(request.token_ids[0]);
        sum += static_cast<std::uint32_t>(request.token_ids[last / 3]);
        sum += static_cast<std::uint32_t>(request.token_ids[(last * 2) / 3]);
        sum += static_cast<std::uint32_t>(request.token_ids[last]);
        request.result = folded_result(sum);
    }
}

void Func_CompactReq() {
    for (CompactReq& request : compact_requests) {
        const std::uint64_t sum =
            request.id + request.prompt_tokens + request.expected_output_tokens +
            request.user_id + request.model_id + request.priority +
            static_cast<std::uint64_t>(request.deadline);
        request.result = folded_result(sum);
    }
}

int main(int argc, char* argv[]) {
    Mode mode = Mode::compact;
    bool has_mode = false;
    std::size_t count = kRequestCount;
    std::size_t iterations = kProcessingIterations;

    for (int i = 1; i < argc; ++i) {
        const std::string_view argument(argv[i]);
        if (argument == "--mode" && i + 1 < argc) {
            const std::string_view value(argv[++i]);
            if (value == "large") mode = Mode::large;
            else if (value == "compact") mode = Mode::compact;
            else { print_usage(argv[0]); return EXIT_FAILURE; }
            has_mode = true;
        } else if (argument == "--count" && i + 1 < argc) {
            if (!parse_count(argv[++i], count)) {
                std::cerr << "Invalid request count.\n";
                return EXIT_FAILURE;
            }
        } else if (argument == "--iterations" && i + 1 < argc) {
            if (!parse_count(argv[++i], iterations) || iterations == 0) {
                std::cerr << "Iterations must be a positive integer.\n";
                return EXIT_FAILURE;
            }
        } else if (argument == "--help" || argument == "-h") {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        } else {
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
    }
    if (!has_mode) { print_usage(argv[0]); return EXIT_FAILURE; }

    try {
        const std::size_t before = resident_memory_bytes();
        const auto allocation_start = std::chrono::steady_clock::now();
        Make_vector_req(mode, count);
        const auto allocation_end = std::chrono::steady_clock::now();
        const std::size_t after_allocation = resident_memory_bytes();

        const auto processing_start = std::chrono::steady_clock::now();
        for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
            if (mode == Mode::large) Func_LargeReq();
            else Func_CompactReq();
        }
        const auto processing_end = std::chrono::steady_clock::now();

        const auto allocation_ms = std::chrono::duration<double, std::milli>(
            allocation_end - allocation_start);
        const auto processing_ms = std::chrono::duration<double, std::milli>(
            processing_end - processing_start);

        std::cout << std::fixed << std::setprecision(3)
                  << "Mode: " << (mode == Mode::large ? "large" : "compact")
                  << "\nRequests: " << count
                  << "\nProcessing iterations: " << iterations
                  << "\nObject size: "
                  << (mode == Mode::large ? sizeof(LargeRequest) : sizeof(CompactReq))
                  << " bytes\nAllocation/initialization time: " << allocation_ms.count()
                  << " ms\nProcessing time: " << processing_ms.count()
                  << " ms\nResident memory before allocation: " << to_mib(before)
                  << " MiB\nResident memory after allocation: "
                  << to_mib(after_allocation) << " MiB\nResident memory increase: "
                  << to_mib(after_allocation >= before ? after_allocation - before : 0)
                  << " MiB\nPeak resident memory: " << to_mib(peak_memory_bytes())
                  << " MiB\n";

        // Keep the processing loop observable to an optimizing compiler.
        if (count != 0) {
            const int sample = mode == Mode::large ? large_requests.back().result
                                                   : compact_requests.back().result;
            std::cout << "Last result: " << sample << '\n';
        }
    } catch (const std::bad_alloc&) {
        std::cerr << "Not enough memory for " << count
                  << " requests. Try a smaller --count value.\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
