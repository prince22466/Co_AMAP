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

constexpr int kRequestCount = 15;
constexpr double kLatencyBudgetSeconds = 0.040;
constexpr double kPrefillSecondsPerToken = 0.000002;
constexpr double kDecodeSecondsPerToken = 0.000050;

const std::vector<int> kPromptTokenChoices{
    64, 128, 256, 512, 1024, 2048, 4096,
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
    std::uniform_int_distribution<std::size_t> prompt_choice(
        0, kPromptTokenChoices.size() - 1);
    std::uniform_int_distribution<std::size_t> interarrival_choice(
        0, kInterarrivalTimeChoices.size() - 1);

    for (int id = 0; id < kRequestCount; ++id) {
        const double interarrival =
            kInterarrivalTimeChoices[interarrival_choice(generator)];
        std::this_thread::sleep_for(
            std::chrono::duration<double>(interarrival));

        const int prompt_tokens =
            kPromptTokenChoices[prompt_choice(generator)];
        const int output_tokens = prompt_tokens / 2;
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

void print_report() {
    std::cout << std::fixed << std::setprecision(3);
    std::cout
        << std::setw(3) << "id"
        << std::setw(9) << "arrival"
        << std::setw(8) << "prompt"
        << std::setw(8) << "output"
        << std::setw(9) << "queued"
        << std::setw(9) << "service"
        << std::setw(10) << "complete"
        << std::setw(10) << "deadline"
        << std::setw(5) << "met" << '\n';
    std::cout << std::string(71, '-') << '\n';

    double total_queueing_time = 0.0;
    double total_service_time = 0.0;
    double total_latency = 0.0;
    std::size_t deadline_misses = 0;

    for (const Request& request : completed_requests) {
        const double service_time =
            request.llm_response_time.value_or(0.0);
        const double completion_time =
            request.completion_time.value_or(request.arrival_time);
        const double latency = completion_time - request.arrival_time;
        const bool met_deadline = completion_time <= request.deadline;

        total_queueing_time += request.queue_waiting_time;
        total_service_time += service_time;
        total_latency += latency;
        deadline_misses += met_deadline ? 0 : 1;

        std::cout
            << std::setw(3) << request.id
            << std::setw(9) << request.arrival_time
            << std::setw(8) << request.prompt_tokens
            << std::setw(8)
            << request.expected_output_tokens.value_or(0)
            << std::setw(9) << request.queue_waiting_time
            << std::setw(9) << service_time
            << std::setw(10) << completion_time
            << std::setw(10) << request.deadline
            << std::setw(5) << (met_deadline ? "yes" : "no")
            << '\n';
    }

    const double count =
        static_cast<double>(completed_requests.size());
    double average_queueing_time = 0.0;
    double average_service_time = 0.0;
    double average_latency = 0.0;
    double throughput = 0.0;
    double deadline_miss_rate = 0.0;

    if (!completed_requests.empty()) {
        average_queueing_time = total_queueing_time / count;
        average_service_time = total_service_time / count;
        average_latency = total_latency / count;
        deadline_miss_rate =
            static_cast<double>(deadline_misses) / count;

        const double first_arrival =
            completed_requests.front().arrival_time;
        const double last_completion =
            completed_requests.back().completion_time.value_or(
                first_arrival);
        const double elapsed_time = last_completion - first_arrival;
        if (elapsed_time > 0.0) {
            throughput = count / elapsed_time;
        }
    }

    std::cout << std::setprecision(6) << '\n'
              << "Average queueing time:      "
              << average_queueing_time << " s\n"
              << "Average service time:       "
              << average_service_time << " s\n"
              << "Average end-to-end latency: "
              << average_latency << " s\n"
              << std::setprecision(3)
              << "Throughput:                 "
              << throughput << " requests/s\n"
              << std::setprecision(2)
              << "Deadline-miss rate:         "
              << deadline_miss_rate * 100.0 << "%\n";
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
