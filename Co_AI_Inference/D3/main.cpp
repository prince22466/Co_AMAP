// C++20 real-time producer/consumer simulation for a single FIFO LLM server.

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <queue>
#include <random>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kRequestCount = 10'000;
constexpr int kLongRequestCount = kRequestCount * 80 / 100;
constexpr int kShortRequestCount = kRequestCount - kLongRequestCount;
constexpr double kLatencyBudgetSeconds = 0.040;
constexpr double kPrefillSecondsPerToken = 0.000002;
constexpr double kDecodeSecondsPerToken = 0.000050;

const std::vector<int> kLongPromptTokenChoices{
    1024, 2048, 4096,
};
const std::vector<int> kShortPromptTokenChoices{
    64, 128, 256, 512,
};
const std::vector<double> kInterarrivalTimeChoices{
    0.001, 0.002, 0.003, 0.005, 0.006,
    0.007, 0.008, 0.009, 0.010,
};

struct Request {
    int id{};
    double arrival_time{};
    int prompt_tokens{};
    std::optional<int> expected_output_tokens;
    double deadline{};
    double queue_waiting_time{};
    std::optional<double> llm_response_time;
    std::optional<double> completion_time;
};

enum class RequestType {
    Short,
    Long,
};

std::queue<Request> request_queue;
std::vector<Request> completed_requests;
std::mutex queue_mutex;
std::condition_variable request_available;
bool producer_finished = false;
Clock::time_point simulation_start;

double seconds_since_start() {
    return std::chrono::duration<double>(
               Clock::now() - simulation_start)
        .count();
}

double estimate_service_time(int prompt_tokens, int output_tokens) {
    return prompt_tokens * kPrefillSecondsPerToken
        + output_tokens * kDecodeSecondsPerToken;
}

void producer() {
    std::mt19937 generator(std::random_device{}());
    std::uniform_int_distribution<std::size_t> long_prompt_choice(
        0, kLongPromptTokenChoices.size() - 1);
    std::uniform_int_distribution<std::size_t> short_prompt_choice(
        0, kShortPromptTokenChoices.size() - 1);
    std::uniform_int_distribution<std::size_t> interarrival_choice(
        0, kInterarrivalTimeChoices.size() - 1);

    // Build and shuffle an exact 80/20 workload mix.
    std::vector<RequestType> workload;
    workload.reserve(kRequestCount);
    workload.insert(
        workload.end(), kLongRequestCount, RequestType::Long);
    workload.insert(
        workload.end(), kShortRequestCount, RequestType::Short);
    std::shuffle(workload.begin(), workload.end(), generator);

    for (int id = 0; id < kRequestCount; ++id) {
        const double interarrival =
            kInterarrivalTimeChoices[interarrival_choice(generator)];
        std::this_thread::sleep_for(
            std::chrono::duration<double>(interarrival));

        const bool is_long =
            workload[static_cast<std::size_t>(id)] == RequestType::Long;
        const int prompt_tokens = is_long
            ? kLongPromptTokenChoices[long_prompt_choice(generator)]
            : kShortPromptTokenChoices[short_prompt_choice(generator)];
        const int output_tokens = prompt_tokens / 2;//assume to be half of reqeust token.
        const double arrival_time = seconds_since_start();

        Request request{
            .id = id,
            .arrival_time = arrival_time,
            .prompt_tokens = prompt_tokens,
            .expected_output_tokens = output_tokens,
            .deadline = arrival_time + kLatencyBudgetSeconds,
            .queue_waiting_time = 0.0,
            .llm_response_time = std::nullopt,
            .completion_time = std::nullopt,
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

void consumer() {
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

        const double service_start = seconds_since_start();
        request.queue_waiting_time =
            service_start - request.arrival_time;

        const double requested_service_time = estimate_service_time(
            request.prompt_tokens,
            request.expected_output_tokens.value_or(0));
        const auto service_started_at = Clock::now();
        std::this_thread::sleep_for(
            std::chrono::duration<double>(requested_service_time));
        request.llm_response_time = std::chrono::duration<double>(
                                        Clock::now() - service_started_at)
                                        .count();
        request.completion_time = seconds_since_start();
        completed_requests.push_back(std::move(request));
    }
}

double percentile(
    const std::vector<double>& sorted_values,
    double quantile) {
    if (sorted_values.empty()) {
        return 0.0;
    }

    // Linear interpolation between the two surrounding observations.
    const double position =
        quantile * static_cast<double>(sorted_values.size() - 1);
    const auto lower_index = static_cast<std::size_t>(position);
    const auto upper_index =
        std::min(lower_index + 1, sorted_values.size() - 1);
    const double fraction =
        position - static_cast<double>(lower_index);

    return sorted_values[lower_index]
        + fraction
            * (sorted_values[upper_index] - sorted_values[lower_index]);
}

void print_report() {
    std::vector<double> end_to_end_latencies;
    std::vector<double> queueing_times;
    std::vector<double> service_times;
    end_to_end_latencies.reserve(completed_requests.size());
    queueing_times.reserve(completed_requests.size());
    service_times.reserve(completed_requests.size());

    for (const Request& request : completed_requests) {
        const double service_time =
            request.llm_response_time.value_or(0.0);
        const double completion_time =
            request.completion_time.value_or(request.arrival_time);
        end_to_end_latencies.push_back(
            completion_time - request.arrival_time);
        queueing_times.push_back(request.queue_waiting_time);
        service_times.push_back(service_time);
    }

    std::sort(end_to_end_latencies.begin(), end_to_end_latencies.end());
    std::sort(queueing_times.begin(), queueing_times.end());
    std::sort(service_times.begin(), service_times.end());

    std::cout << "Completed requests: " << completed_requests.size()
              << "\nLong requests:      " << kLongRequestCount
              << " (80%)"
              << "\nShort requests:     " << kShortRequestCount
              << " (20%)\n\n";

    std::cout << std::fixed << std::setprecision(6)
              << std::left << std::setw(26) << "Metric"
              << std::right << std::setw(14) << "p50 (s)"
              << std::setw(14) << "p95 (s)"
              << std::setw(14) << "p99 (s)" << '\n'
              << std::string(68, '-') << '\n';

    const auto print_percentiles =
        [](const char* label, const std::vector<double>& values) {
            std::cout << std::left << std::setw(26) << label
                      << std::right << std::setw(14)
                      << percentile(values, 0.50)
                      << std::setw(14) << percentile(values, 0.95)
                      << std::setw(14) << percentile(values, 0.99)
                      << '\n';
        };

    print_percentiles("End-to-end latency", end_to_end_latencies);
    print_percentiles("Queueing time", queueing_times);
    print_percentiles("Service time", service_times);
}

}  // namespace

int main() {
    simulation_start = Clock::now();

    {
        // std::jthread is a C++20 joining thread. Both threads are joined
        // automatically when this scope ends.
        std::jthread producer_thread(producer);
        std::jthread consumer_thread(consumer);
    }

    print_report();
    return 0;
}
