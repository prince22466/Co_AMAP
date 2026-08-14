// C++20 producer/consumer mutex benchmark for a shared FIFO LLM server.
// Each scenario sends 10M requests (80% short) to 2, 4, 8, or 16 workers.
//
// Structure and data flow
// =======================
//
//                              +-------------------+
//                              |     Producer      |
//                              |10,000,000 requests|
//                              +---------+---------+
//                                        |
//                                        v
//                    +---------------------------------------+
//                    |       Shared input FIFO queue         |
//                    |  protected by mutex + condition var   |
//                    +--+---------------+-----------------+--+
//                       |               |                 |
//                       v               v                 v
//                  +---------+     +---------+       +---------+
//                  | Worker 0|     | Worker 1|  ...  |Worker N-1|
//                  | CPU work|     | CPU work|       | CPU work|
//                  +----+----+     +----+----+       +----+----+
//                       |               |                 |
//                       v               v                 v
//                  +---------+     +---------+       +---------+
//                  | Private |     | Private |       | Private |
//                  | FIFO 0  |     | FIFO 1  |       |FIFO N-1 |
//                  +----+----+     +----+----+       +----+----+
//                       |               |                 |
//                       +---------------+-----------------+
//                                       |
//                             all threads joined
//                                       |
//                                       v
//                    +---------------------------------------+
//                    | Merge into completed_requests vector |
//                    +-------------------+-------------------+
//                                        |
//                                        v
//                    +---------------------------------------+
//                    | Validate the completed-request count  |
//                    +---------------------------------------+
//
// The program runs N = 2, 4, 8, and 16 sequentially by default. For isolated
// perf measurements, select one scenario with: --workers 2|4|8|16


#include <algorithm>
#include <cstddef>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <optional>
#include <queue>
#include <random>
#include <string_view>
#include <thread>
#include <vector>

namespace {

#if defined(__GNUC__) || defined(__clang__)
#define PERF_NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#define PERF_NOINLINE __declspec(noinline)
#else
#define PERF_NOINLINE
#endif

constexpr std::size_t kRequestCount = 10'000'000;
constexpr std::size_t kShortRequestCount = kRequestCount * 80 / 100;
constexpr std::size_t kLongRequestCount =
    kRequestCount - kShortRequestCount;
constexpr unsigned int kRandomSeed = 42;
constexpr int kWorkerScenarios[]{2, 4, 8, 16};
const std::vector<int> kLongPromptTokenChoices{
    1024, 2048, 4096,
};
const std::vector<int> kShortPromptTokenChoices{
    64, 128, 256, 512,
};
enum class RequestType {
    Short,
    Long,
};

struct Request {
    std::size_t id{};
    int prompt_tokens{};
};

std::queue<Request> request_queue;
std::vector<Request> completed_requests;
std::mutex queue_mutex;
std::condition_variable request_available;
bool producer_finished = false;

thread_local volatile double cpu_work_sink = 0.0;

PERF_NOINLINE void perform_cpu_work(int input_tokens) {
    double value = 1.0;
    const int iterations = input_tokens > 0
        ? std::max(1, input_tokens / 10)
        : 0;
    for (int i = 0; i < iterations; ++i) {
        value = value + i;  // Simulate CPU-intensive work.
    }
    // Make the result observable so the optimizer retains the CPU work.
    cpu_work_sink = value;
}

PERF_NOINLINE void perform_service(int input_tokens) {
    perform_cpu_work(input_tokens);
}

void producer() {
    // Use the same workload in every worker-count scenario.
    std::mt19937 generator(kRandomSeed);
    std::uniform_int_distribution<std::size_t> long_prompt_choice(
        0, kLongPromptTokenChoices.size() - 1);
    std::uniform_int_distribution<std::size_t> short_prompt_choice(
        0, kShortPromptTokenChoices.size() - 1);
    // Build and shuffle an exact 80/20 workload mix.
    std::vector<RequestType> workload;
    workload.reserve(kRequestCount);
    workload.insert(
        workload.end(), kLongRequestCount, RequestType::Long);
    workload.insert(
        workload.end(), kShortRequestCount, RequestType::Short);
    std::shuffle(workload.begin(), workload.end(), generator);

    for (std::size_t id = 0; id < kRequestCount; ++id) {
        const bool is_long =
            workload[id] == RequestType::Long;
        const int prompt_tokens = is_long
            ? kLongPromptTokenChoices[long_prompt_choice(generator)]
            : kShortPromptTokenChoices[short_prompt_choice(generator)];

        Request request{
            .id = id,
            .prompt_tokens = prompt_tokens,
        };

        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            request_queue.push(std::move(request));
        }// queue_mutex is automatically unlocked after the scope
        request_available.notify_one();
    }

    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        producer_finished = true;
    }
    request_available.notify_all();
}

void consumer(std::queue<Request>& processed_requests) {
    while (true) {
        Request request;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            request_available.wait(lock, [] {
                return !request_queue.empty() || producer_finished;
            });

            if (request_queue.empty()) {
                break;
            }

            request = std::move(request_queue.front());
            request_queue.pop();
        }

        perform_service(request.prompt_tokens);
        // This queue belongs exclusively to the current worker.
        processed_requests.push(std::move(request));
    }
}

bool run_scenario(int worker_count) {
    request_queue = {};
    completed_requests.clear();
    completed_requests.reserve(kRequestCount);
    producer_finished = false;
    std::vector<std::queue<Request>> worker_processed_queues(
        static_cast<std::size_t>(worker_count));

    {
        std::jthread producer_thread(producer);
        std::vector<std::jthread> worker_threads;
        worker_threads.reserve(static_cast<std::size_t>(worker_count));
        for (int worker = 0; worker < worker_count; ++worker) {
            worker_threads.emplace_back(
                consumer,
                std::ref(worker_processed_queues[
                    static_cast<std::size_t>(worker)]));
        }
    }

    // All threads have joined, so the per-worker queues can now be merged
    // without synchronization into the final completed collection.
    for (auto& processed_queue : worker_processed_queues) {
        while (!processed_queue.empty()) {
            completed_requests.push_back(
                std::move(processed_queue.front()));
            processed_queue.pop();
        }
    }

    return completed_requests.size() == kRequestCount;
}

std::optional<int> parse_worker_count(int argc, char* argv[]) {
    if (argc == 1) {
        return 0;  // Run every configured scenario.
    }

    if (argc == 3 && std::string_view(argv[1]) == "--workers") {
        const std::string_view value(argv[2]);
        for (const int worker_count : kWorkerScenarios) {
            if ((worker_count == 2 && value == "2")
                || (worker_count == 4 && value == "4")
                || (worker_count == 8 && value == "8")
                || (worker_count == 16 && value == "16")) {
                return worker_count;
            }
        }
    }

    std::cerr << "Usage: " << argv[0]
              << " [--workers 2|4|8|16]\n";
    return std::nullopt;
}

}  // namespace

int main(int argc, char* argv[]) {
    const std::optional<int> selected_worker_count =
        parse_worker_count(argc, argv);
    if (!selected_worker_count) {
        return 1;
    }

    if (*selected_worker_count != 0) {
        if (run_scenario(*selected_worker_count)) {
            return 0;
        }
        std::cerr << "Completed-request count mismatch\n";
        return 2;
    }

    for (const int worker_count : kWorkerScenarios) {
        if (!run_scenario(worker_count)) {
            std::cerr << "Completed-request count mismatch for "
                      << worker_count << " workers\n";
            return 2;
        }
    }
    return 0;
}
