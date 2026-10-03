// DECODE SIMULATOR
// Flow: last token ID -> embedding lookup -> 10 vector-matrix layers -> next ID.
// Each layer: input[1,D] * weights[D,D] = output[1,D], with D=128 by default.
// Generate exactly ONE token per iteration, then feed it into the next iteration.
// SIMD parallelizes output columns for that token, not future generation steps.
// One active layer occupies 65 KiB, below the configured 256 KiB L2 capacity.
// This is a linear-algebra example, without attention or a real vocabulary head.
// main() benchmarks 500 sequences of 100 generated tokens and prints a summary.
//
// Build/run in WSL:
// cmake --build build-wsl --target decode_sim -j && ./build-wsl/decode_sim
// Profile the whole program, including initialization and all benchmark runs:
// perf stat -d -- ./build-wsl/decode_sim

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <span>
#include <vector>

// SIMD availability is chosen at compile time; other CPUs use scalar arithmetic.
// SSE2 computes four float values at once on x86-64 CPUs.
#if defined(__SSE2__) || defined(_M_X64)
#include <emmintrin.h>
#define DECODE_HAS_SSE2 1
#else
#define DECODE_HAS_SSE2 0
#endif

// COMPONENT 1: Experiment configuration and cache-size accounting.
// D = embedding width, L = layers, NUM_GENERATED_TOKENS = iterations per run.
// NUM_RUNS repeats the entire sequence; each iteration still generates one token.
constexpr int EMBEDDING_SIZE = 128;
constexpr int NUM_LAYERS = 10;
constexpr int NUM_GENERATED_TOKENS = 100;
constexpr int NUM_RUNS = 500;
constexpr int VOCAB_SIZE = EMBEDDING_SIZE; // Toy output scores map to these IDs.
constexpr std::size_t L2_BYTES = 256 * 1024;
constexpr std::size_t LAYER_WEIGHT_ELEMENTS =
    static_cast<std::size_t>(EMBEDDING_SIZE) * EMBEDDING_SIZE;
// Active arrays: one input vector + one layer's weights + one output vector.
// Other layers and the vocabulary embedding table are not counted here.
constexpr std::size_t LAYER_WORKING_SET_BYTES =
    (LAYER_WEIGHT_ELEMENTS + 2 * EMBEDDING_SIZE) * sizeof(float);
static_assert(EMBEDDING_SIZE > 0 && NUM_LAYERS > 0 && NUM_GENERATED_TOKENS > 0);
static_assert(LAYER_WORKING_SET_BYTES <= L2_BYTES * 3 / 4,
              "Reduce embedding size to stay within 75% of L2");

// COMPONENT 2: Flat storage and SIMD vector-matrix multiplication.
using Embedding = std::vector<float>;
// Flat row-major matrix: [row, column] is at row * EMBEDDING_SIZE + column.
using Matrix = std::vector<float>;

// Multiply one embedding [1 x D] by weights [D x D] into output [1 x D].
// Caller provides distinct input/output vectors and a flat D x D weight array.
// Keep 16 output values in four SIMD registers across the entire inner loop.
// This parallelizes output columns for ONE token; it does not batch tokens.
// Optional dimension allows testing short vectors and partial SIMD blocks.
void multiply(const Embedding& input, const Matrix& weights, Embedding& output,
              int dimension = EMBEDDING_SIZE) {
    int j = 0;
#if DECODE_HAS_SSE2
    for (; j + 16 <= dimension; j += 16) {
        // Four registers cover columns j..j+15 of the SAME output token.
        // Start fresh sums; every output column is overwritten, so no fill is needed.
        __m128 sum0 = _mm_setzero_ps();
        __m128 sum1 = _mm_setzero_ps();
        __m128 sum2 = _mm_setzero_ps();
        __m128 sum3 = _mm_setzero_ps();
        for (int k = 0; k < dimension; ++k) {
            // k walks the dot product. Load 16 adjacent weights from row k.
            const float* row = weights.data() + static_cast<std::size_t>(k) * dimension + j;
            // Broadcast one input value and apply it to all 16 output columns.
            const __m128 value = _mm_set1_ps(input[k]);
            sum0 = _mm_add_ps(sum0, _mm_mul_ps(value, _mm_loadu_ps(row)));
            sum1 = _mm_add_ps(sum1, _mm_mul_ps(value, _mm_loadu_ps(row + 4)));
            sum2 = _mm_add_ps(sum2, _mm_mul_ps(value, _mm_loadu_ps(row + 8)));
            sum3 = _mm_add_ps(sum3, _mm_mul_ps(value, _mm_loadu_ps(row + 12)));
        }
        // Unaligned stores work with std::vector's ordinary allocation.
        // No output-array loads or stores occur inside the k loop.
        _mm_storeu_ps(output.data() + j, sum0);
        _mm_storeu_ps(output.data() + j + 4, sum1);
        _mm_storeu_ps(output.data() + j + 8, sum2);
        _mm_storeu_ps(output.data() + j + 12, sum3);
    }
    // Handle remaining groups of four columns the same way.
    for (; j + 4 <= dimension; j += 4) {
        __m128 sum = _mm_setzero_ps();
        for (int k = 0; k < dimension; ++k) {
            const __m128 w = _mm_loadu_ps(
                weights.data() + static_cast<std::size_t>(k) * dimension + j);
            sum = _mm_add_ps(sum, _mm_mul_ps(_mm_set1_ps(input[k]), w));
        }
        _mm_storeu_ps(output.data() + j, sum);
    }
#endif
    // Scalar tail, or all columns when SSE2 is unavailable.
    for (; j < dimension; ++j) {
        float sum = 0.0f;
        for (int k = 0; k < dimension; ++k) {
            sum += input[k] * weights[static_cast<std::size_t>(k) * dimension + j];
        }
        output[j] = sum;
    }
}

// COMPONENT 3: One-token layer workflow and toy token selection.
// Decode one token: pass its embedding through all 10 layers.
// A span views one row of the flat embedding table without copying that row.
int decode(std::span<const float> input, const std::vector<Matrix>& layer_weights) {
    Embedding hidden(input.begin(), input.end());
    Embedding output(EMBEDDING_SIZE);
    for (const Matrix& weights : layer_weights) {
        multiply(hidden, weights, output);
        hidden.swap(output); // Reuse just two working vectors across layers.
    }

    // Toy selection: treat the final D values as scores for IDs 0..D-1.
    // Pick the highest score. A real model would use a vocabulary projection.
    const auto best = std::max_element(hidden.begin(), hidden.end());
    return static_cast<int>(best - hidden.begin());
}

// COMPONENT 4: Model setup, autoregressive benchmark, and final reporting.
int main() {
    using Clock = std::chrono::steady_clock;
    const auto initialization_start = Clock::now();

    // 1. Make up embeddings for token IDs 0..D-1.
    // Each row of the flat table contains that token's D-number embedding.
    // A fixed random seed makes the example repeatable.
    std::mt19937 random(42);
    std::uniform_real_distribution<float> embedding_value(-1.0f, 1.0f);
    Matrix embeddings(static_cast<std::size_t>(VOCAB_SIZE) * EMBEDDING_SIZE);
    for (float& value : embeddings) {
        value = embedding_value(random);
    }

    // 2. Make up 10 flat weight matrices, each D x D.
    // Small random weights keep values at a reasonable scale across layers.
    const float scale = std::sqrt(3.0f / EMBEDDING_SIZE);
    std::uniform_real_distribution<float> weight_value(-scale, scale);
    std::vector<Matrix> layer_weights(
        NUM_LAYERS, Matrix(LAYER_WEIGHT_ELEMENTS));
    for (Matrix& weights : layer_weights) {
        for (float& value : weights) {
            value = weight_value(random);
        }
    }

    // 3. Make up an initial sequence. Only its last token enters decode.
    std::vector<int> tokens = {100 % VOCAB_SIZE, 101 % VOCAB_SIZE, 102 % VOCAB_SIZE};
    tokens.reserve(tokens.size() + NUM_GENERATED_TOKENS);
    const auto initialization_end = Clock::now();
    // Configuration output is outside both initialization and decode timing.
    std::cout << "Decode: 1 input token per step, embedding size " << EMBEDDING_SIZE
              << ", " << NUM_LAYERS << " layers\n"
              << "Multiplication backend: "
              << (DECODE_HAS_SSE2 ? "SSE2 SIMD (1 token x 16 output columns, register accumulation)"
                                  : "scalar fallback")
              << "\nConfigured L2 capacity: " << L2_BYTES / 1024 << " KiB\n"
              << "One layer's weights: " << LAYER_WEIGHT_ELEMENTS * sizeof(float) / 1024
              << " KiB\n"
              << "Input + output: " << 2.0 * EMBEDDING_SIZE * sizeof(float) / 1024
              << " KiB\n"
              << "Active layer working set: " << LAYER_WORKING_SET_BYTES / 1024.0
              << " KiB (" << 100.0 * LAYER_WORKING_SET_BYTES / L2_BYTES << "% of L2)\n";
    std::cout << "Initial token IDs: ";
    for (int token : tokens) {
        std::cout << token << ' ';
    }
    std::cout << '\n';

    // 4. Generate 100 tokens autoregressively, one token per iteration.
    // Look up the last token's embedding, decode it, then append the new ID.
    // On the next iteration, this new token becomes the input.
    // Keep printing outside the timed loop so terminal I/O is excluded.
    // Repeat the 100-token sequence 500 times using the same model and inputs.
    // Reset to the original three tokens before every run, outside timing.
    // Caches are not flushed; the first run is included in the statistics.
    const std::size_t initial_token_count = tokens.size();
    std::vector<double> run_seconds(NUM_RUNS);
    double total_seconds = 0.0;
    unsigned long long token_checksum = 0;
    for (int run = 0; run < NUM_RUNS; ++run) {
        // Preserve the first three IDs and discard the previous run's generated IDs.
        // Capacity was reserved, so resetting does not allocate a new token array.
        tokens.resize(initial_token_count);
        const auto start = Clock::now();
        for (int step = 0; step < NUM_GENERATED_TOKENS; ++step) {
            const int last_token = tokens.back();
            // A span views just this token's row in the flat embedding table.
            const std::span<const float> input(
                embeddings.data() + static_cast<std::size_t>(last_token) * EMBEDDING_SIZE,
                EMBEDDING_SIZE);
            const int next_token = decode(input, layer_weights);
            // Exactly one append per step. The next step consumes this new token.
            tokens.push_back(next_token);
        }
        const auto end = Clock::now();
        run_seconds[run] = std::chrono::duration<double>(end - start).count();
        total_seconds += run_seconds[run];
        // Consume every run's result without printing inside the repeat loop.
        token_checksum += static_cast<unsigned long long>(tokens.back());
    }

    // Show one sample sequence; all 500 runs are summarized below.
    std::cout << "Generated token IDs (last run):\n";
    const std::size_t first_generated = tokens.size() - NUM_GENERATED_TOKENS;
    for (int step = 0; step < NUM_GENERATED_TOKENS; ++step) {
        std::cout << tokens[first_generated + step]
                  << ((step + 1) % 10 == 0 ? '\n' : ' ');
    }

    std::cout << "Generated " << NUM_GENERATED_TOKENS << " tokens through "
              << NUM_LAYERS << " layers per step.\n";

    // 5. Compute analytical work/traffic estimates for ONE 100-token run.
    // These estimates are not collected from hardware performance counters.
    // Each weight participates in one multiply and one add: 2 FLOPs.
    // A fused multiply-add also counts as 2 FLOPs, regardless of instructions.
    const double weight_count = static_cast<double>(NUM_LAYERS)
                                * EMBEDDING_SIZE * EMBEDDING_SIZE;
    const double flops_per_token = 2.0 * weight_count;
    const double weight_bytes = weight_count * sizeof(float);
    const double total_flops = flops_per_token * NUM_GENERATED_TOKENS;
    const double total_weight_bytes = weight_bytes * NUM_GENERATED_TOKENS;
    // Ideal transfer model: weights/input read once and output written once
    // per layer per generated token; excludes zeroing and write allocation.
    const double ideal_transfer_bytes = static_cast<double>(NUM_LAYERS)
                                       * NUM_GENERATED_TOKENS * LAYER_WORKING_SET_BYTES;
    const double initialization_seconds =
        std::chrono::duration<double>(initialization_end - initialization_start).count();
    // Use mean run duration so rates account for all NUM_RUNS repetitions.
    const double decode_seconds = total_seconds / NUM_RUNS;
    const auto [minimum, maximum] = std::minmax_element(run_seconds.begin(), run_seconds.end());

    // 6. Print timing statistics and estimated intensities/rates.
    // Initialization and output are excluded; per-call vector allocation is timed.
    // Logical weight scans may be served by cache rather than RAM.
    // Neither the scan count nor the ideal transfer model measures RAM traffic.
    std::cout << std::fixed << std::setprecision(3)
              << "\nDecode performance (MB/GB use decimal units):\n"
              << "Initialization time: " << initialization_seconds << " s\n"
              << "Measured runs: " << NUM_RUNS << " (no cache flush)\n"
              << "Last token checksum (all runs): " << token_checksum << '\n'
              << "Total timed decode time: " << total_seconds << " s\n"
              << "Mean decode time per run: " << decode_seconds * 1000.0 << " ms\n"
              << "Minimum / maximum run time: " << *minimum * 1000.0 << " / "
              << *maximum * 1000.0 << " ms\n"
              << "Work and traffic estimates below are per run; rates use mean time.\n"
              << "Average decode latency: "
              << decode_seconds * 1000.0 / NUM_GENERATED_TOKENS << " ms/token\n"
              << "Token throughput: " << NUM_GENERATED_TOKENS / decode_seconds
              << " tokens/s\n"
              << "Weight storage: " << weight_bytes / 1e6 << " MB\n"
              << "Estimated work per token: " << flops_per_token / 1e6 << " MFLOPs\n"
              << "Estimated work per run: " << total_flops / 1e9 << " GFLOPs\n"
              << "Logical weight bytes scanned (including cache reads): "
              << total_weight_bytes / 1e9
              << " GB\n"
              << "Logical weight-scan intensity: "
              << flops_per_token / weight_bytes << " FLOPs/byte\n"
              << "Ideal transfer bytes (weights + input + output, all steps): "
              << ideal_transfer_bytes / 1e6 << " MB\n"
              << "Ideal transfer arithmetic intensity: "
              << total_flops / ideal_transfer_bytes << " FLOPs/byte\n"
              << "Achieved compute rate: " << total_flops / decode_seconds / 1e9
              << " GFLOP/s\n"
              << "Effective logical weight-scan rate (not RAM bandwidth): "
              << total_weight_bytes / decode_seconds / 1e9 << " GB/s\n"
              << "Only the active layer is counted; the embedding table, other layers "
                 "and cache conflicts can affect residency.\n"
              << "Cache capacity and these estimates alone cannot confirm the bottleneck.\n";
    return 0;
}
