// WORKFLOW
// 1. Make trivial embeddings for the only request: "Will it be Waylander in future?".
// 2. prefill(): pass these embeddings through 20 attention + feed-forward layers.
// 3. decode(): generate and process 50 token embeddings in 50 sequential iterations.
// 4. Print tensor shapes, token IDs, cache sizes, elapsed times, and a checksum.
//
// This is a small CPU teaching example: synthetic weights, no sleeps,
// no trained model. Positional encoding and learned normalization are omitted.
// PARALLEL comments mark opportunities; the implementation still uses serial loops.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

constexpr std::array<std::string_view, 7> DEMO_TOKENS{
    "Will", "it", "be", "Waylander", "in", "future", "?"};


constexpr std::size_t WIDTH = 64;  // Also the size of our toy vocabulary.
constexpr std::size_t NUM_LAYERS = 20;
constexpr std::size_t NUM_HEADS = 4;
constexpr std::size_t HEAD_WIDTH = WIDTH / NUM_HEADS;
constexpr std::size_t FFN_WIDTH = 2 * WIDTH;
constexpr std::string_view DEMO_PROMPT = "Will it be Waylander in future?";

static_assert(WIDTH % NUM_HEADS == 0);
static_assert(DEMO_TOKENS.size() <= WIDTH);
constexpr std::size_t MAX_TOKENS = 8192;
constexpr std::size_t OUTPUT_TOKENS = 50;

using Clock = std::chrono::steady_clock;
using Features = std::array<float, WIDTH>;
using EmbeddingTable = std::array<Features, WIDTH>;

struct LayerWeights {
    std::vector<float> query, key, value, output;
    std::vector<float> ffn_up, ffn_down;
};
using Model = std::array<LayerWeights, NUM_LAYERS>;

struct KVCache {
    // Flattened [cached tokens, NUM_HEADS, HEAD_WIDTH]. Q is not cached.
    std::vector<float> keys, values;
};
using ModelCache = std::array<KVCache, NUM_LAYERS>;

struct QKV {
    std::vector<float> queries, keys, values;
};

struct TokenTensor {
    std::array<std::size_t, 2> shape;  // [batch size = 1, number of tokens]
    std::vector<std::int64_t> data;

    explicit TokenTensor(std::size_t length) : shape{1, length}, data(length) {}
};

struct Result {
    TokenTensor response;
    std::vector<float> response_embeddings{};  // [1, response length, WIDTH]
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double checksum = 0.0;  // Keeps all calculated results observable.
};

struct DecodeResult {
    TokenTensor tokens;
    std::vector<float> embeddings;  // Selected tokens' embeddings, not hidden states.
    double ms = 0.0;
};

double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// Deterministic synthetic numbers, used only to initialize the toy model.
float sample_value(std::size_t row, std::size_t col) {
    return static_cast<float>(static_cast<int>((row * 17 + col * 13) % 101) - 50)
           / 50.0F;
}

// Hand-assigned toy token embeddings: Will=e0, it=e1, be=e2, Waylander=e3,
// in=e4, future=e5, ?=e6. Each e has one 1 and 63 zeros; no learned semantics.
// The seven rows follow DEMO_TOKENS order, giving shape [1, 7, 64].
std::vector<float> make_prompt_embeddings() {
    std::vector<float> embeddings(DEMO_TOKENS.size() * WIDTH, 0.0F);
    for (std::size_t token = 0; token < DEMO_TOKENS.size(); ++token)
        embeddings[token * WIDTH + token] = 1.0F;
    return embeddings;
}

Model make_model() {
    Model model;
    for (std::size_t layer = 0; layer < NUM_LAYERS; ++layer) {
        auto initialize = [layer](std::vector<float>& matrix, std::size_t in,
                                  std::size_t out, std::size_t salt) {
            matrix.resize(in * out);
            for (std::size_t row = 0; row < in; ++row)
                for (std::size_t col = 0; col < out; ++col)
                    matrix[row * out + col] =
                        sample_value(row + layer * 37 + salt, col + salt * 11)
                        / std::sqrt(static_cast<float>(in));
        };
        auto& w = model[layer];
        initialize(w.query, WIDTH, WIDTH, 1);
        initialize(w.key, WIDTH, WIDTH, 2);
        initialize(w.value, WIDTH, WIDTH, 3);
        initialize(w.output, WIDTH, WIDTH, 4);
        initialize(w.ffn_up, WIDTH, FFN_WIDTH, 5);
        initialize(w.ffn_down, FFN_WIDTH, WIDTH, 6);
    }
    return model;
}

// Matrix multiplication: [rows, input_width] x [input_width, output_width].
void project(const float* input, const std::vector<float>& weights,
             float* output, std::size_t rows,
             std::size_t input_width = WIDTH, std::size_t output_width = WIDTH) {
    std::fill(output, output + rows * output_width, 0.0F);
    // PARALLEL: token rows write separate output rows after initialization.
    // Output columns are independent too; the k loop accumulates into the same
    // elements, so parallelizing k requires a reduction rather than shared +=.
    for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t k = 0; k < input_width; ++k)
            for (std::size_t col = 0; col < output_width; ++col)
                output[row * output_width + col] +=
                    input[row * input_width + k] * weights[k * output_width + col];
}

// Per-token RMS normalization, with unit scale (no learned gain).
std::vector<float> normalize(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    // PARALLEL: each token row has its own sum, scale, and output slice.
    // Within a row, the sum is a reduction; scaling must wait for that sum.
    for (std::size_t row = 0; row < input.size() / WIDTH; ++row) {
        float sum = 0.0F;
        for (std::size_t col = 0; col < WIDTH; ++col) {
            const float x = input[row * WIDTH + col];
            sum += x * x;
        }
        const float scale = 1.0F / std::sqrt(sum / WIDTH + 1e-5F);
        for (std::size_t col = 0; col < WIDTH; ++col)
            output[row * WIDTH + col] = input[row * WIDTH + col] * scale;
    }
    return output;
}

QKV calculate_qkv(const std::vector<float>& input, const LayerWeights& weights) {
    QKV qkv{std::vector<float>(input.size()), std::vector<float>(input.size()),
            std::vector<float>(input.size())};
    const std::size_t rows = input.size() / WIDTH;
    // PARALLEL: Q, K, and V projections read the same input and write separate
    // buffers. Each projection can also process token rows in parallel.
    project(input.data(), weights.query, qkv.queries.data(), rows);
    project(input.data(), weights.key, qkv.keys.data(), rows);
    project(input.data(), weights.value, qkv.values.data(), rows);
    return qkv;
}

void store_kv_cache(const QKV& qkv, KVCache& cache) {
    // PARALLEL: K and V can be copied independently once their projections finish.
    // For parallel token copies, resize each vector first and write disjoint
    // slices; concurrent insert() calls on the same vector are not safe.
    cache.keys.insert(cache.keys.end(), qkv.keys.begin(), qkv.keys.end());
    cache.values.insert(cache.values.end(), qkv.values.begin(), qkv.values.end());
}

// One query/head: softmax(Q K^T / sqrt(head width)). Only visible positions
// are included, so future prompt tokens are causally masked.
void calculate_attention_weights(const float* query, const KVCache& cache,
                                 std::size_t head, std::size_t visible_tokens,
                                 std::vector<float>& scores) {
    scores.resize(visible_tokens);
    const std::size_t offset = head * HEAD_WIDTH;
    // PARALLEL: each visible key produces an independent score after resize().
    // Each dot product is a reduction over head features.
    for (std::size_t token = 0; token < visible_tokens; ++token) {
        float dot = 0.0F;
        for (std::size_t col = 0; col < HEAD_WIDTH; ++col)
            dot += query[offset + col] * cache.keys[token * WIDTH + offset + col];
        scores[token] = dot / std::sqrt(static_cast<float>(HEAD_WIDTH));
    }
    // PARALLEL reductions: first find the maximum, then exponentiate and sum.
    // Each stage waits for the preceding reduction; final divisions are independent.
    const float maximum = *std::max_element(scores.begin(), scores.end());
    float sum = 0.0F;
    for (float& score : scores) {
        score = std::exp(score - maximum);
        sum += score;
    }
    for (float& score : scores) score /= sum;
}

// One query/head: attention weights times V. Heads occupy adjacent columns,
// so writing each head's context here also concatenates the heads.
void calculate_attention_context(const std::vector<float>& scores,
                                 const KVCache& cache, std::size_t head,
                                 float* context) {
    const std::size_t offset = head * HEAD_WIDTH;
    std::fill(context + offset, context + offset + HEAD_WIDTH, 0.0F);
    // PARALLEL: context columns can be computed independently. The token loop
    // accumulates into shared columns, so parallel tokens require a reduction.
    for (std::size_t token = 0; token < scores.size(); ++token)
        for (std::size_t col = 0; col < HEAD_WIDTH; ++col)
            context[offset + col] +=
                scores[token] * cache.values[token * WIDTH + offset + col];
}

void add_residual(std::vector<float>& output, const std::vector<float>& input) {
    // PARALLEL: each element is independent once the sublayer output is ready.
    for (std::size_t i = 0; i < output.size(); ++i) output[i] += input[i];
}

std::vector<float> attention_output(const std::vector<float>& context,
                                    const LayerWeights& weights,
                                    const std::vector<float>& input) {
    std::vector<float> output(input.size());
    project(context.data(), weights.output, output.data(), input.size() / WIDTH);
    add_residual(output, input);
    return output;
}

// Sublayer 1: normalize -> Q/K/V -> cache -> weights -> context -> output.
std::vector<float> multihead_sublayer(const std::vector<float>& input,
                                     const LayerWeights& weights, KVCache& cache) {
    const auto normalized = normalize(input);
    const QKV qkv = calculate_qkv(normalized, weights);
    const std::size_t previous_tokens = cache.keys.size() / WIDTH;
    store_kv_cache(qkv, cache);

    std::vector<float> context(input.size());
    std::vector<float> scores;
    scores.reserve(cache.keys.size() / WIDTH);
    // PARALLEL: query rows and heads can run independently once Q and the
    // required cached K/V are ready. Give each worker its own scores buffer;
    // the current shared scratch vector cannot be used concurrently.
    // Context slices are disjoint. Causality restricts which K/V a query reads,
    // but does not require earlier query outputs to finish first.
    for (std::size_t row = 0; row < input.size() / WIDTH; ++row)
        for (std::size_t head = 0; head < NUM_HEADS; ++head) {
            calculate_attention_weights(qkv.queries.data() + row * WIDTH, cache,
                                        head, previous_tokens + row + 1, scores);
            calculate_attention_context(scores, cache, head, context.data() + row * WIDTH);
        }
    // A token's output projection needs context from all of its heads.
    // For this batched call, finish all context writes before reading context.
    return attention_output(context, weights, input);
}

void activation(std::vector<float>& hidden) {
    // ReLU, a simple feed-forward activation for this teaching model.
    // PARALLEL: each feature of each token is independent.
    for (float& value : hidden) value = std::max(0.0F, value);
}

// Sublayer 2: normalize -> expand -> activation -> project back -> residual.
// PARALLEL: tokens are independent throughout this sublayer. For each token,
// preserve the stage order: normalization -> up projection -> ReLU -> down projection.
std::vector<float> feed_forward_sublayer(const std::vector<float>& input,
                                        const LayerWeights& weights) {
    const auto normalized = normalize(input);
    const std::size_t rows = input.size() / WIDTH;
    std::vector<float> hidden(rows * FFN_WIDTH);
    project(normalized.data(), weights.ffn_up, hidden.data(), rows, WIDTH, FFN_WIDTH);
    activation(hidden);
    std::vector<float> output(input.size());
    project(hidden.data(), weights.ffn_down, output.data(), rows, FFN_WIDTH, WIDTH);
    add_residual(output, input);
    return output;
}

// Input is already embeddings: flattened [prompt_length, WIDTH], batch size 1.
// Return every prompt position's final-layer output and fill all 20 KV caches.
std::vector<float> prefill(const std::vector<float>& input_embeddings,
                           const Model& model, ModelCache& caches) {
    if (input_embeddings.empty() || input_embeddings.size() % WIDTH != 0 ||
        input_embeddings.size() / WIDTH > MAX_TOKENS)
        throw std::invalid_argument("Expected embeddings with shape [1, length, WIDTH], length 1..8192.");
    for (auto& cache : caches) {
        cache.keys.clear();
        cache.values.clear();
    }
    std::vector<float> output = input_embeddings;
    // SEQUENTIAL: each layer consumes the previous layer's output, and its
    // feed-forward sublayer consumes its attention output. Token/head parallelism
    // is inside the sublayers; do not parallelize this layer loop.
    for (std::size_t layer = 0; layer < NUM_LAYERS; ++layer) {
        output = multihead_sublayer(output, model[layer], caches[layer]);
        output = feed_forward_sublayer(output, model[layer]);
    }
    return output;
}

// Process one new embedding through the same layers, appending one K/V per layer.
std::vector<float> decode_token(const Features& embedding, const Model& model,
                                ModelCache& caches) {
    std::vector<float> output(embedding.begin(), embedding.end());
    for (std::size_t layer = 0; layer < NUM_LAYERS; ++layer) {
        output = multihead_sublayer(output, model[layer], caches[layer]);
        output = feed_forward_sublayer(output, model[layer]);
    }
    return output;
}

// Toy vocabulary projection using the embedding table as the output weights.
std::int64_t select_token(const std::vector<float>& output, const EmbeddingTable& embeddings) {
    const auto state = normalize(output);
    Features logits{};
    for (std::size_t id = 0; id < WIDTH; ++id)
        for (std::size_t col = 0; col < WIDTH; ++col)
            logits[id] += state[col] * embeddings[id][col];
    return static_cast<std::int64_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

// The final prompt output predicts the first response token. Each iteration saves
// its selected embedding, then processes it through all layers and grows the caches.
// We also process the last token so all emitted tokens are cached: N iterations,
// N layer-stack passes. If stopping immediately, that last pass could be skipped.
DecodeResult decode(const std::vector<float>& prefill_output, ModelCache& caches,
                    const EmbeddingTable& embeddings, const Model& model,
                    std::size_t output_length = OUTPUT_TOKENS) 
{
    
    if (prefill_output.empty() || prefill_output.size() % WIDTH != 0 ||
        prefill_output.size() / WIDTH > MAX_TOKENS ||
        output_length == 0 || output_length > MAX_TOKENS)
        throw std::invalid_argument("Expected nonempty prefill output and 1..8192 output tokens.");
    
        for (const auto& cache : caches)
        if (cache.keys.size() != prefill_output.size() || cache.values.size() != prefill_output.size())
            throw std::invalid_argument("Decode requires the matching prompt KV caches in every layer.");

    DecodeResult result{TokenTensor(output_length), std::vector<float>(output_length * WIDTH)};
    const auto start = Clock::now();
    std::vector<float> state(prefill_output.end() - WIDTH, prefill_output.end());
    // SEQUENTIAL: the next generated token depends on the token selected here.
    // Parallel work is possible within a token's layer computations, as above.
    for (std::size_t token = 0; token < output_length; ++token) {
        const auto next_id = select_token(state, embeddings);
        result.tokens.data[token] = next_id;
        const Features& next_embedding = embeddings[static_cast<std::size_t>(next_id)];
        std::copy(next_embedding.begin(), next_embedding.end(),
                  result.embeddings.data() + token * WIDTH);
        // Each layer forms Q/K/V, appends K/V, attends over the growing cache,
        // then applies its feed-forward sublayer. The final state predicts the next ID.
        state = decode_token(next_embedding, model, caches);
    }
    result.ms = elapsed_ms(start);
    return result;
}

void run_prompt_demo(const Model& model) {
    const auto input_embeddings = make_prompt_embeddings();
    // Use the same one-hot mapping for prompt tokens and generated token IDs.
    EmbeddingTable embedding_table{};
    for (std::size_t id = 0; id < WIDTH; ++id) embedding_table[id][id] = 1.0F;

    ModelCache caches;
    const auto start = Clock::now();
    const auto output = prefill(input_embeddings, model, caches);
    const double prefill_ms = elapsed_ms(start);
    const std::size_t prompt_cache_size = caches.front().keys.size() / WIDTH;
    const auto response = decode(output, caches, embedding_table, model, OUTPUT_TOKENS);

    double checksum = 0.0;
    for (float value : output) checksum += value;
    for (float value : response.embeddings) checksum += value;
    for (const auto id : response.tokens.data) checksum += static_cast<double>(id);
    for (const auto& cache : caches)
        for (std::size_t i = 0; i < cache.keys.size(); ++i)
            checksum += cache.keys[i] + cache.values[i];

    std::cout << "\nPrompt: " << DEMO_PROMPT << "\nToy tokens:";
    for (const auto token : DEMO_TOKENS) std::cout << " [" << token << ']';
    std::cout << "\nOne-hot embeddings [1, " << DEMO_TOKENS.size() << ", " << WIDTH
              << "] -> prefill output [1, " << output.size() / WIDTH << ", " << WIDTH
              << "]\nPrefill cache: " << prompt_cache_size << " K/V pairs per layer.\n"
              << "Decode: " << response.tokens.data.size() << " iterations -> token embeddings [1, "
              << response.embeddings.size() / WIDTH << ", " << WIDTH << "]\n"
              << "Final cache: " << caches.front().keys.size() / WIDTH
              << " K/V pairs in each of " << caches.size() << " layers.\n"
              << "Generated token IDs (each ID identifies its one-hot embedding):";
    for (const auto id : response.tokens.data) std::cout << ' ' << id;
    std::cout << "\nPrefill: " << prefill_ms << " ms; decode: " << response.ms
              << " ms; checksum: " << checksum << '\n';
}


int main() {
    const Model model = make_model();
    std::cout << std::fixed << std::setprecision(3)
              << "20 layers, 4 heads: causal attention + feed-forward.\n";
    run_prompt_demo(model);
    return 0;
}
