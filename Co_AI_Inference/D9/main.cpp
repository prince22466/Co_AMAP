// KV cache memory benchmark, with an 8 GiB cap and a 95% clearing threshold.
// M = 2 * batch_num * tokens size per batch * layers * kv_heads * head_depth * bytes_per_value.
// The factor 2 accounts for both keys and values. Token counts are per sequence.
//
// Default scenarios: batch size = {16, 32, 64, 128}; input/output tokens size = {2048, 4192, 8192, 16384}.
// Model layout: 10 layers, 32 KV heads per layer, depth 128 per layer, FP16 storage (2 bytes).
// This measures real vector allocation/writes, without model or attention math.
// Clearing discards context; this is a memory experiment, not valid inference.
//
// TOKEN PROCESSING ORDER
// batch is the number of sequences; input_tokens/output_tokens are per sequence.
// Each loop iteration appends ONE token position across ALL sequences in the batch:
//   Step 1: token 1's K/V for sequence 1, sequence 2, ..., sequence batch.
//   Step 2: token 2's K/V for sequence 1, sequence 2, ..., sequence batch.
//   ...continue until every sequence has received the requested token count.
// Each step writes batch * 2 * layers * kv_heads * head_depth FP16 values.
// One sequence's entire token history is not processed before the next sequence.
// This same token-by-token order is used for BOTH prefill and decoding here.
// The writes are serial CPU operations; the code does not execute sequences in parallel.
// Before each step, clear the whole batch's cache if that step would cross 95%.

// WORKFLOW
//
// main()
//   |
//   +--> positive_number() --> parse and validate supplied arguments
//   +--> gib() -------------> print cap settings and CSV header
//   |
//   v
// Select one requested scenario OR loop over all 64 combinations
//   |
//   v
// run_batch(batch, input_tokens, output_tokens, cap_bytes)
//   |
//   +--> Calculate cache limits --> start timer --> cache.reserve()
//   |
//   +--> process_tokens(input_tokens)  [prefill]
//   |      Record prefill end time, used bytes and clear count
//   |
//   +--> process_tokens(output_tokens) [decode]
//   |      Record decode end time
//   |
//   +--> gib() + milliseconds() --> print this scenario's CSV result
//   |
//   v
// Release vector storage --> return to main() --> next scenario or finish
//
// process_tokens(count) is a lambda inside run_batch():
//   |
//   v
// For each token step (one token for every sequence in the batch):
//   |
//   v
// Would appending this step exceed the 95% threshold?
//   | yes                         | no
//   v                             |
// cache.clear(); ++clears          |
//   |                             |
//   +-----------------------------+
//   |
//   v
// cache.resize() --> write synthetic FP16 K/V values for this step
//   |
//   v
// Update peak_bytes --> next token step, or return when count is reached

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <new>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <vector>

using Clock = std::chrono::steady_clock;

static_assert(sizeof(std::size_t) >= 8, "Use a 64-bit build for an 8 GiB cache.");
constexpr std::size_t GIB = 1024ULL * 1024 * 1024;
constexpr std::size_t CAP_BYTES = 8 * GIB;
constexpr std::size_t LAYERS = 10;
constexpr std::size_t KV_HEADS = 32;
constexpr std::size_t HEAD_DEPTH = 128;
constexpr std::size_t VALUES_PER_TOKEN = 2 * LAYERS * KV_HEADS * HEAD_DEPTH;
constexpr std::size_t BYTES_PER_TOKEN = VALUES_PER_TOKEN * sizeof(std::uint16_t);

double gib(std::size_t bytes) {
    return static_cast<double>(bytes) / GIB;
}

double milliseconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::size_t positive_number(std::string_view text) {
    std::size_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || value == 0) {
        throw std::invalid_argument("Batch, input tokens and output tokens must be positive integers.");
    }
    return value;
}

void run_batch(std::size_t batch, std::size_t input_tokens, std::size_t output_tokens,
               std::size_t cap_bytes) {
    const std::size_t clear_bytes = cap_bytes * 95 / 100;
    // Validate before multiplying, so even a very large batch cannot overflow.
    if (batch > clear_bytes / BYTES_PER_TOKEN) {
        throw std::invalid_argument("One token across this batch exceeds the cache threshold.");
    }
    const std::size_t step_values = batch * VALUES_PER_TOKEN;
    const std::size_t step_bytes = batch * BYTES_PER_TOKEN;
    const std::size_t max_cached_tokens = clear_bytes / step_bytes;
    const std::size_t max_values = max_cached_tokens * step_values;

    // uint16_t holds FP16 bit patterns; no nonstandard half-precision type is needed.
    // Layout: [token step][batch][K/V][layer][head][depth].
    std::vector<std::uint16_t> cache;
    std::size_t peak_bytes = 0;
    std::size_t clears = 0;

    const auto start = Clock::now();
    // Reserve once to prevent geometric vector growth and transient reallocations
    // from exceeding the cap. Size still grows by one batched token per step.
    cache.reserve(max_values);
    if (cache.capacity() > cap_bytes / sizeof(std::uint16_t)) {
        throw std::runtime_error("Vector capacity exceeds the selected cap.");
    }

    const auto process_tokens = [&](std::size_t count) {
        for (std::size_t token = 0; token < count; ++token) {
            // Clear BEFORE adding a token that would cross the 95% threshold.
            // clear() resets size to zero and retains the allocation for reuse.
            if (cache.size() > max_values - step_values) {
                cache.clear();
                ++clears;
            }
            // 0x3c00 is FP16 1.0. Filling touches every new K/V entry in memory.
            cache.resize(cache.size() + step_values, std::uint16_t{0x3c00});
            peak_bytes = std::max(peak_bytes, cache.size() * sizeof(std::uint16_t));
        }
    };

    // Stream input tokens too: some prompts alone are larger than the cap.
    process_tokens(input_tokens);
    const auto prefill_end = Clock::now();
    const std::size_t prefill_bytes = cache.size() * sizeof(std::uint16_t);
    const std::size_t prefill_clears = clears;

    process_tokens(output_tokens);
    const auto decode_end = Clock::now();

    const double uncapped_gib = (static_cast<double>(input_tokens)
                              + static_cast<double>(output_tokens)) * gib(step_bytes);
    std::cout << batch << ',' << input_tokens << ',' << output_tokens << ','
              << uncapped_gib << ',' << gib(prefill_bytes) << ','
              << gib(cache.size() * sizeof(std::uint16_t)) << ',' << gib(peak_bytes) << ','
              << gib(cache.capacity() * sizeof(std::uint16_t)) << ','
              << prefill_clears << ',' << clears - prefill_clears << ','
              << milliseconds(start, prefill_end) << ','
              << milliseconds(prefill_end, decode_end) << ','
              << milliseconds(start, decode_end) << std::endl;
}

int main(int argc, char* argv[]) {
    constexpr std::string_view usage =
        " [batch input_tokens output_tokens] [--cap-mib MiB]\n";
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << "Usage: " << argv[0] << usage
                  << "No arguments runs all 64 scenarios. Token counts are per sequence.\n"
                  << "Default cap: 8192 MiB (8 GiB); use a smaller cap for a low-memory run.\n";
        return 0;
    }

    try {
        std::size_t cap_bytes = CAP_BYTES;
        if (argc >= 3 && std::string_view(argv[argc - 2]) == "--cap-mib") {
            const std::size_t cap_mib = positive_number(argv[argc - 1]);
            if (cap_mib > 8192) {
                throw std::invalid_argument("The cap must be between 1 and 8192 MiB.");
            }
            cap_bytes = cap_mib * 1024 * 1024;
            argc -= 2;
        }
        if (argc != 1 && argc != 4) {
            std::cerr << "Usage: " << argv[0] << usage;
            return 1;
        }
        // Parse before printing results or allocating memory.
        const std::size_t batch = argc == 4 ? positive_number(argv[1]) : 0;
        const std::size_t input = argc == 4 ? positive_number(argv[2]) : 0;
        const std::size_t output = argc == 4 ? positive_number(argv[3]) : 0;

        std::cout << std::fixed << std::setprecision(3)
                  << "# KV cache writes only; 10 layers, 32 KV heads, depth 128, FP16.\n"
                  << "# Cap: " << gib(cap_bytes) << " GiB; clear before next token would exceed "
                  << gib(cap_bytes * 95 / 100) << " GiB (95%).\n"
                  << "# clear() resets used size; reserved memory stays allocated until batch ends.\n"
                  << "# Prefill includes reservation; timings exclude printing and final deallocation.\n"
                  << "batch,input_tokens,output_tokens,uncapped_GiB,prefill_GiB,final_GiB,"
                     "peak_GiB,reserved_GiB,prefill_clears,decode_clears,prefill_ms,decode_ms,total_ms\n"
                  << std::flush;

        if (argc == 4) {
            run_batch(batch, input, output, cap_bytes);
        } else {//run scenarios for all combinations of batch sizes and token counts
            for (const std::size_t batch_size : {8, 16, 20}) {
                for (const std::size_t input_tokens : {2048, 4192, 8192, 16384}) {
                    for (const std::size_t output_tokens : {2048, 4192, 8192, 16384}) {
                        run_batch(batch_size, input_tokens, output_tokens, cap_bytes);
                    }
                }
            }
        }
    } catch (const std::bad_alloc&) {
        std::cerr << "Not enough memory; try a smaller cap with --cap-mib.\n";
        return 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
