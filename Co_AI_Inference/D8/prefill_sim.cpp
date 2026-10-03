// PREFILL SIMULATOR
// Flow: made-up token embeddings -> 10 matrix-matrix layers -> first output token.
// Each layer: A[T,D] * B[D,D] = C[T,D], with T=128 and D=128 by default.
// Four tokens share weight loads; partial output sums stay in SIMD registers.
// One active layer occupies 192 KiB; its multiplication tiles occupy 5 KiB.
// This is a linear-algebra example, without attention or a real vocabulary head.
// main() initializes data once, benchmarks 500 runs, then prints a summary.

// Build in WSL (C++20, optimized, with symbols for Linux perf):
// cmake -S . -B build-wsl -DCMAKE_BUILD_TYPE=RelWithDebInfo
// cmake --build build-wsl --target prefill_sim -j
// Profile the whole program, including input/weight initialization:
// perf stat -d -- ./build-wsl/prefill_sim

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

// SIMD availability: choose an implementation at compile time, not by timing.
// SSE2 is available on x86-64 CPUs. Other architectures use the scalar loop.
#if defined(__SSE2__) || defined(_M_X64)
#include <emmintrin.h>
#define PREFILL_HAS_SSE2 1
#else
#define PREFILL_HAS_SSE2 0
#endif

// COMPONENT 1: Experiment configuration and cache-size accounting.
// T = input tokens, D = embedding width, L = layer count.
// NUM_RUNS repeats an entire prefill call; it is not a token or layer count.
constexpr int NUM_TOKENS = 128;
constexpr int EMBEDDING_SIZE = 128;
constexpr int NUM_LAYERS = 10;
constexpr int NUM_RUNS = 500;
constexpr int BLOCK_TOKENS = 4; // Kernel explicitly uses four rows; keep this at 4.
constexpr int BLOCK_COLUMNS = 32;
constexpr int BLOCK_INNER = 32;
constexpr std::size_t L2_BYTES = 256 * 1024; // Per-cache capacity reported by WSL.
constexpr std::size_t ACTIVATION_ELEMENTS =
    static_cast<std::size_t>(NUM_TOKENS) * EMBEDDING_SIZE;
constexpr std::size_t LAYER_WEIGHT_ELEMENTS =
    static_cast<std::size_t>(EMBEDDING_SIZE) * EMBEDDING_SIZE;
// Active arrays: input[T,D] + weights[D,D] + output[T,D].
// Count only the current layer, not all model weights or allocator overhead.
constexpr std::size_t LAYER_WORKING_SET_BYTES =
    (LAYER_WEIGHT_ELEMENTS + 2 * ACTIVATION_ELEMENTS) * sizeof(float);
constexpr std::size_t TILE_WORKING_SET_BYTES =
    (BLOCK_TOKENS * BLOCK_INNER + BLOCK_INNER * BLOCK_COLUMNS
     + BLOCK_TOKENS * BLOCK_COLUMNS) * sizeof(float);
static_assert(NUM_TOKENS > 0 && EMBEDDING_SIZE > 0 && NUM_LAYERS > 0);
// Leave 25% of L2 capacity for other data. Capacity does not guarantee residency.
static_assert(LAYER_WORKING_SET_BYTES <= L2_BYTES * 3 / 4,
              "Reduce token count or embedding size to stay within 75% of L2");
static_assert(TILE_WORKING_SET_BYTES <= L2_BYTES * 3 / 4);

// COMPONENT 2: Storage and helper for leftover token rows.
// One contiguous row-major array. Element [row, column] is at row * D + column.
// No matrix transpose or padding is required by this example.
using Matrix = std::vector<float>;

// Add value * weights to one output row. SIMD computes four columns at once.
// Unaligned loads/stores work with std::vector's ordinary memory allocation.
void add_scaled_row(float value, const float* weights, float* output, int columns) {
    int j = 0;
#if PREFILL_HAS_SSE2
    const __m128 values = _mm_set1_ps(value); // Broadcast value to all four lanes.
    for (; j + 4 <= columns; j += 4) {
        const __m128 weight_values = _mm_loadu_ps(weights + j);
        const __m128 previous = _mm_loadu_ps(output + j);
        const __m128 products = _mm_mul_ps(values, weight_values);
        _mm_storeu_ps(output + j, _mm_add_ps(previous, products));
    }
#endif
    // Handle any remaining columns, or the whole row on a non-SSE2 CPU.
    for (; j < columns; ++j) {
        output[j] += value * weights[j];
    }
}

// COMPONENT 3: Blocked SIMD matrix-matrix multiplication.
// Multiply embeddings [tokens x D] by weights [D x D] into output [tokens x D].
// Caller provides three separate arrays; output must not alias either input.
// Blocked matrix multiplication. Four token rows share each weight-vector load.
// A 4 x 4 output microtile stays in registers across 32 inner-dimension steps.
// Column/inner tiles limit the active data; accumulation keeps k in order.
// Optional dimensions also allow checking partial blocks with small matrices.
void multiply(const Matrix& embeddings, const Matrix& weights, Matrix& output,
              int token_count = NUM_TOKENS, int dimension = EMBEDDING_SIZE) {
    std::fill(output.begin(), output.end(), 0.0f);

    int token = 0;
#if PREFILL_HAS_SSE2
    for (; token + BLOCK_TOKENS <= token_count; token += BLOCK_TOKENS) {
        // a0..a3 are four input rows; c0..c3 are their four output rows.
        const float* a0 = embeddings.data() + static_cast<std::size_t>(token) * dimension;
        const float* a1 = a0 + dimension;
        const float* a2 = a1 + dimension;
        const float* a3 = a2 + dimension;
        float* c0 = output.data() + static_cast<std::size_t>(token) * dimension;
        float* c1 = c0 + dimension;
        float* c2 = c1 + dimension;
        float* c3 = c2 + dimension;

        // j chooses output columns; k is the shared dot-product dimension.
        // Walk column tiles, then accumulate contributions from each k tile.
        for (int j0 = 0; j0 < dimension; j0 += BLOCK_COLUMNS) {
            const int j_end = std::min(j0 + BLOCK_COLUMNS, dimension);
            for (int k0 = 0; k0 < dimension; k0 += BLOCK_INNER) {
                const int k_end = std::min(k0 + BLOCK_INNER, dimension);
                int j = j0;
                for (; j + 4 <= j_end; j += 4) {
                    // Each __m128 contains four columns for ONE token row.
                    // Load partial sums from earlier k tiles (initially zero).
                    __m128 sum0 = _mm_loadu_ps(c0 + j);
                    __m128 sum1 = _mm_loadu_ps(c1 + j);
                    __m128 sum2 = _mm_loadu_ps(c2 + j);
                    __m128 sum3 = _mm_loadu_ps(c3 + j);
                    for (int k = k0; k < k_end; ++k) {
                        // One weight load serves four different tokens.
                        const __m128 w = _mm_loadu_ps(
                            weights.data() + static_cast<std::size_t>(k) * dimension + j);
                        // Broadcast one embedding value into all four lanes.
                        // Four independent updates reuse the SAME weight vector.
                        sum0 = _mm_add_ps(sum0, _mm_mul_ps(_mm_set1_ps(a0[k]), w));
                        sum1 = _mm_add_ps(sum1, _mm_mul_ps(_mm_set1_ps(a1[k]), w));
                        sum2 = _mm_add_ps(sum2, _mm_mul_ps(_mm_set1_ps(a2[k]), w));
                        sum3 = _mm_add_ps(sum3, _mm_mul_ps(_mm_set1_ps(a3[k]), w));
                    }
                    // Store only after finishing this inner block.
                    _mm_storeu_ps(c0 + j, sum0);
                    _mm_storeu_ps(c1 + j, sum1);
                    _mm_storeu_ps(c2 + j, sum2);
                    _mm_storeu_ps(c3 + j, sum3);
                }
                // Scalar column tail for dimensions not divisible by four.
                for (; j < j_end; ++j) {
                    float sum0 = c0[j], sum1 = c1[j], sum2 = c2[j], sum3 = c3[j];
                    for (int k = k0; k < k_end; ++k) {
                        const float w = weights[static_cast<std::size_t>(k) * dimension + j];
                        sum0 += a0[k] * w;
                        sum1 += a1[k] * w;
                        sum2 += a2[k] * w;
                        sum3 += a3[k] * w;
                    }
                    c0[j] = sum0; c1[j] = sum1; c2[j] = sum2; c3[j] = sum3;
                }
            }
        }
    }
#endif
    // Remaining token rows, or every row when SSE2 is unavailable.
    for (; token < token_count; ++token) {
        const std::size_t token_offset = static_cast<std::size_t>(token) * dimension;
        for (int k = 0; k < dimension; ++k) {
            const float value = embeddings[token_offset + k];
            const std::size_t weight_offset = static_cast<std::size_t>(k) * dimension;
            add_scaled_row(value, weights.data() + weight_offset,
                           output.data() + token_offset, dimension);
        }
    }
}

// COMPONENT 4: Layer workflow and toy output-token selection.
struct PrefillResult {
    std::vector<float> embedding;
    int token;
};

// Process all input embeddings through the 10 layers, then select a token.
// Token IDs identify the inputs; their corresponding embeddings do the math.
PrefillResult prefill(const std::vector<int>& tokens,
                      Matrix hidden,
                      const std::vector<Matrix>& layer_weights) {
    assert(tokens.size() == NUM_TOKENS && hidden.size() == ACTIVATION_ELEMENTS);
    // Only two activation arrays stay live. Reuse them across layers.
    Matrix output(ACTIVATION_ELEMENTS);

    for (std::size_t layer = 0; layer < layer_weights.size(); ++layer) {
        multiply(hidden, layer_weights[layer], output);
        // The current output becomes the next input without copying its data.
        hidden.swap(output);
    }

    // Use the last input position to produce the first output token.
    const auto last_row = hidden.begin() + (tokens.size() - 1) * EMBEDDING_SIZE;
    std::vector<float> final_embedding(last_row, hidden.end());

    // Toy token selection: treat the D values as scores for token IDs
    // 0..D-1 and pick the largest. A real model would use a vocabulary head.
    const auto best = std::max_element(final_embedding.begin(), final_embedding.end());
    const int output_token = static_cast<int>(best - final_embedding.begin());
    return {final_embedding, output_token};
}

// COMPONENT 5: Input setup, repeated benchmark, and final reporting.
int main() {
    using Clock = std::chrono::steady_clock;
    const auto initialization_start = Clock::now();

    // 1. Make up token IDs: 100, 101, ...
    std::vector<int> tokens(NUM_TOKENS);
    for (int i = 0; i < NUM_TOKENS; ++i) {
        tokens[i] = 100 + i;
    }

    // 2. Make up an embedding for each token, with values between -1 and 1.
    // A fixed seed makes the example repeatable.
    std::mt19937 random(42);
    std::uniform_real_distribution<float> embedding_value(-1.0f, 1.0f);
    Matrix embeddings(ACTIVATION_ELEMENTS);
    for (float& value : embeddings) {
        value = embedding_value(random);
    }

    // 3. Make up 10 independent weight matrices of size D x D.
    // Small weights keep the values from growing excessively across layers.
    const float scale = std::sqrt(3.0f / EMBEDDING_SIZE);
    std::uniform_real_distribution<float> weight_value(-scale, scale);
    std::vector<Matrix> layer_weights(
        NUM_LAYERS, Matrix(LAYER_WEIGHT_ELEMENTS));
    for (auto& weights : layer_weights) {
        for (float& value : weights) {
            value = weight_value(random);
        }
    }

    const auto initialization_end = Clock::now();

    // 4. Print configuration, then benchmark the complete prefill call.
    // Only timed call durations are summed; setup and terminal output are excluded.
    std::cout << "Prefill: " << NUM_TOKENS << " tokens, embedding size "
              << EMBEDDING_SIZE << ", " << NUM_LAYERS << " layers\n";
    std::cout << "Multiplication backend: "
              << (PREFILL_HAS_SSE2 ? "blocked SSE2 SIMD (4 tokens x 4 output columns)"
                                   : "scalar fallback")
              << '\n';
    std::cout << "Configured L2 capacity: " << L2_BYTES / 1024 << " KiB\n"
              << "One layer's weights: " << LAYER_WEIGHT_ELEMENTS * sizeof(float) / 1024
              << " KiB\n"
              << "Input + output: " << 2 * ACTIVATION_ELEMENTS * sizeof(float) / 1024
              << " KiB\n"
              << "Active layer working set: " << LAYER_WORKING_SET_BYTES / 1024
              << " KiB (" << 100.0 * LAYER_WORKING_SET_BYTES / L2_BYTES << "% of L2)\n"
              << "Tile dimensions (tokens x columns x inner): " << BLOCK_TOKENS << " x "
              << BLOCK_COLUMNS << " x " << BLOCK_INNER << '\n'
              << "Tile array footprint: " << TILE_WORKING_SET_BYTES / 1024.0 << " KiB\n";
    // Repeat the same workload 500 times; initialize weights only once.
    // After each move, recreate the identical input outside the timed call.
    // This avoids keeping a third activation array during multiplication.
    // Caches are not flushed; the first run is included in the statistics.
    std::vector<double> run_seconds(NUM_RUNS);
    double total_seconds = 0.0;
    unsigned long long token_checksum = 0;
    PrefillResult result;
    for (int run = 0; run < NUM_RUNS; ++run) {
        if (run > 0) {
            // Reusing seed 42 restores the original embeddings, not layer output.
            embeddings.resize(ACTIVATION_ELEMENTS);
            std::mt19937 input_random(42);
            for (float& value : embeddings) value = embedding_value(input_random);
        }
        const auto start = Clock::now();
        result = prefill(tokens, std::move(embeddings), layer_weights);
        const auto end = Clock::now();
        run_seconds[run] = std::chrono::duration<double>(end - start).count();
        total_seconds += run_seconds[run];
        // Consume every run's result without printing inside the repeat loop.
        token_checksum += static_cast<unsigned long long>(result.token);
    }

    // 5. Print the first generated token and a few final embedding values.
    std::cout << "First output token ID: " << result.token << '\n';
    std::cout << "Final embedding (first 8 of " << result.embedding.size() << "): ";
    std::cout << std::fixed << std::setprecision(4);
    for (int i = 0; i < std::min(8, EMBEDDING_SIZE); ++i) {
        std::cout << result.embedding[i] << ' ';
    }
    std::cout << '\n';

    // 6. Compute analytical work/traffic estimates for ONE complete run.
    // These estimates are not collected from hardware performance counters.
    // Count the main multiplication work: one multiply + one add per weight
    // per input token. A fused multiply-add also counts as 2 FLOPs.
    const double weight_count = static_cast<double>(NUM_LAYERS)
                                * EMBEDDING_SIZE * EMBEDDING_SIZE;
    const double flops_per_token = 2.0 * weight_count;
    const double weight_bytes = weight_count * sizeof(float);
    const double total_flops = flops_per_token * NUM_TOKENS;
    // Full four-token blocks load each weight once for all four tokens.
    // Leftover token rows scan weights separately. This counts source-level
    // weight loads; compiler changes and cache effects can change actual traffic.
    const int weight_scans = PREFILL_HAS_SSE2
        ? NUM_TOKENS / BLOCK_TOKENS + NUM_TOKENS % BLOCK_TOKENS : NUM_TOKENS;
    const double total_weight_bytes = weight_bytes * weight_scans;
    // Ideal external transfers: weights/input read once, output written once
    // per layer. Excludes zeroing and write-allocate traffic; NOT measured bytes.
    const double ideal_transfer_bytes = static_cast<double>(NUM_LAYERS)
                                       * LAYER_WORKING_SET_BYTES;
    const double initialization_seconds =
        std::chrono::duration<double>(initialization_end - initialization_start).count();
    // Rate = work_per_run / mean_seconds = all_runs_work / all_runs_seconds.
    const double prefill_seconds = total_seconds / NUM_RUNS;
    const auto [minimum, maximum] = std::minmax_element(run_seconds.begin(), run_seconds.end());

    // 7. Print benchmark statistics and estimated intensities/rates.
    // Work is per run, whereas total_seconds covers all NUM_RUNS repetitions.
    // Logical scans include repeated cache reads, not just RAM reads.
    // With cache reuse, RAM arithmetic intensity can exceed scan intensity.
    std::cout << std::fixed << std::setprecision(3)
              << "\nPrefill performance (MB/GB use decimal units):\n"
              << "Initialization time: " << initialization_seconds << " s\n"
              << "Measured runs: " << NUM_RUNS << " (no cache flush)\n"
              << "Output token checksum (all runs): " << token_checksum << '\n'
              << "Total timed prefill time: " << total_seconds << " s\n"
              << "Mean prefill time per run: " << prefill_seconds * 1000.0 << " ms\n"
              << "Minimum / maximum run time: " << *minimum * 1000.0 << " / "
              << *maximum * 1000.0 << " ms\n"
              << "Work and traffic estimates below are per run; rates use mean time.\n"
              << "Average time per input token: "
              << prefill_seconds * 1000.0 / NUM_TOKENS << " ms/token\n"
              << "Input token throughput: " << NUM_TOKENS / prefill_seconds
              << " tokens/s\n"
              << "Weight storage: " << weight_bytes / 1e6 << " MB\n"
              << "Estimated work per input token: " << flops_per_token / 1e6
              << " MFLOPs\n"
              << "Estimated work per run: " << total_flops / 1e9 << " GFLOPs\n"
              << "Estimated logical weight bytes loaded (including cache reads): "
              << total_weight_bytes / 1e9 << " GB\n"
              << "Estimated weight-load intensity: "
              << total_flops / total_weight_bytes << " FLOPs/byte\n"
              << "Ideal weights-only intensity (weights read once per layer): "
              << total_flops / weight_bytes << " FLOPs/byte\n"
              << "Ideal transfer bytes (weights + input + output, all layers): "
              << ideal_transfer_bytes / 1e6 << " MB\n"
              << "Ideal transfer arithmetic intensity: "
              << total_flops / ideal_transfer_bytes << " FLOPs/byte\n"
              << "Achieved compute rate: " << total_flops / prefill_seconds / 1e9
              << " GFLOP/s\n"
              << "Effective estimated weight-load rate (not RAM bandwidth): "
              << total_weight_bytes / prefill_seconds / 1e9 << " GB/s\n"
              << "Only the active layer is counted; other layers and cache conflicts "
                 "can affect residency.\n"
              << "Cache capacity and these estimates alone cannot confirm the bottleneck.\n";
    return 0;
}
