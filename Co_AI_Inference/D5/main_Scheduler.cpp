// C++20 scheduler benchmark for an LLM worker system.
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
//                    |         Scheduler input FIFO          |
//                    | producer/scheduler mutex (workers do  |
//                    |              not lock it)             |
//                    +-------------------+-------------------+
//                                        |
//                                        v
//                    +---------------------------------------+
//                    |   Scheduler: wait for input + idle    |
//                    | worker by polling atomic status, then |
//                    |         dispatch one request          |
//                    +--+---------------+-----------------+--+
//                       |               |                 |
//                       v               v                 v
//                  +---------+     +---------+       +---------+
//                  | Worker 0|     | Worker 1|  ...  |Worker N-1|
//                  |I/A/Work |     |I/A/Work |       |I/A/Work |
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
//                    input empty + all workers idle,
//                         then all threads joined
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
#include <atomic>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
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

enum class WorkerStatus {
    Idle,
    Assigned,
    Working,
    Stop,
};

struct WorkerState {
    std::atomic<WorkerStatus> status{WorkerStatus::Idle};
    Request assigned_request;
    std::queue<Request> processed_requests;
};

std::queue<Request> request_queue;
std::vector<Request> completed_requests;
std::mutex input_queue_mutex;
std::atomic<bool> producer_finished{false};

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
            std::lock_guard<std::mutex> lock(input_queue_mutex);
            request_queue.push(std::move(request));
        }
    }

    producer_finished.store(true, std::memory_order_release);
}

bool all_workers_idle(
    const std::vector<std::unique_ptr<WorkerState>>& workers) {
    return std::all_of(
        workers.begin(), workers.end(),
        [](const auto& worker) {
            return worker->status.load(std::memory_order_acquire)
                == WorkerStatus::Idle;
        });
}

std::optional<std::size_t> find_idle_worker(
    const std::vector<std::unique_ptr<WorkerState>>& workers,
    std::size_t first_worker) {
    for (std::size_t offset = 0; offset < workers.size(); ++offset) {
        const std::size_t index = (first_worker + offset) % workers.size();
        if (workers[index]->status.load(std::memory_order_acquire)
            == WorkerStatus::Idle) {
            return index;
        }
    }
    return std::nullopt;
}

bool input_is_finished() {
    if (!producer_finished.load(std::memory_order_acquire)) {
        return false;
    }
    std::lock_guard<std::mutex> lock(input_queue_mutex);
    return request_queue.empty();
}

void scheduler(
    std::vector<std::unique_ptr<WorkerState>>& workers) {
    std::size_t next_worker = 0;

    while (true) {
        while (true) {
            const auto worker_index =
                find_idle_worker(workers, next_worker);
            if (!worker_index) {
                break;
            }

            std::optional<Request> request;
            {
                std::lock_guard<std::mutex> lock(input_queue_mutex);
                if (request_queue.empty()) {
                    break;
                }
                request.emplace(std::move(request_queue.front()));
                request_queue.pop();
            }

            WorkerState& worker = *workers[*worker_index];
            worker.assigned_request = std::move(*request);
            worker.status.store(
                WorkerStatus::Assigned, std::memory_order_release);
            next_worker = (*worker_index + 1) % workers.size();
        }

        if (input_is_finished() && all_workers_idle(workers)) {
            for (auto& worker : workers) {
                worker->status.store(
                    WorkerStatus::Stop, std::memory_order_release);
            }
            return;
        }
        // No idle worker (or no input) means the outer loop immediately
        // scans again. Worker status is polled; workers send no notification.
    }
}

void worker(WorkerState& state) {
    while (true) {
        const WorkerStatus status =
            state.status.load(std::memory_order_acquire);
        if (status == WorkerStatus::Stop) {
            return;
        }

        if (status != WorkerStatus::Assigned) {
            continue;
        }

        state.status.store(WorkerStatus::Working, std::memory_order_release);
        Request request = std::move(state.assigned_request);
        perform_service(request.prompt_tokens);
        state.processed_requests.push(std::move(request));
        state.status.store(WorkerStatus::Idle, std::memory_order_release);
    }
}

bool run_scenario(int worker_count) {
    request_queue = {};
    completed_requests.clear();
    completed_requests.reserve(kRequestCount);
    producer_finished.store(false, std::memory_order_relaxed);

    std::vector<std::unique_ptr<WorkerState>> workers;
    workers.reserve(static_cast<std::size_t>(worker_count));
    for (int worker_index = 0;
         worker_index < worker_count;
         ++worker_index) {
        workers.push_back(std::make_unique<WorkerState>());
    }

    std::vector<std::jthread> worker_threads;
    worker_threads.reserve(static_cast<std::size_t>(worker_count));
    for (auto& state : workers) {
        worker_threads.emplace_back(worker, std::ref(*state));
    }

    std::jthread scheduler_thread(scheduler, std::ref(workers));
    std::jthread producer_thread(producer);

    producer_thread.join();
    scheduler_thread.join();
    for (std::jthread& worker_thread : worker_threads) {
        worker_thread.join();
    }

    // All threads have joined, so the per-worker queues can now be merged
    // without synchronization into the final completed collection.
    for (auto& state : workers) {
        std::queue<Request>& processed_queue = state->processed_requests;
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
