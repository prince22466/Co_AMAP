// Sequence-aware KV cache memory benchmark: 8 GiB cap, 95% usable KV budget.
// M = 2 * batch_num * tokens size per batch * layers * kv_heads * head_depth * bytes_per_value.
// The factor 2 accounts for both keys and values. Token counts are per sequence.
//
// Default scenarios: batch size = {8, 16, 20}; input/output tokens = {2048, 4192, 8192}.
// Model layout: 10 layers, 32 KV heads per layer, depth 128 per layer, FP16 storage (2 bytes).
// This measures real vector allocation/writes, without model or attention math.
// Each sequence retains its complete prompt and generated-token history until
// generation finishes. Only completed sequences are eligible for eviction.
//
// TOKEN PROCESSING ORDER
// batch is the number of sequences; input_tokens/output_tokens are per sequence.
// Admit a group whose complete input+output histories fit within the KV budget.
// Remaining requests wait for the next group; batch still counts ALL requests.
// Each step appends ONE token position across every sequence in the active group.
// This same token-by-token order is used for BOTH prefill and decoding here.
// The writes are serial CPU operations; the code does not execute sequences in parallel.
// Under pressure, reuse only slots belonging to completed sequences. If even
// one complete sequence cannot fit, reject the scenario without dropping KV.
//
// WORKFLOW: validate -> check available RAM -> reserve a fitting token pool ->
// for each admitted group:
//   start sequences -> prefill -> decode -> mark sequences completed.
// Append: take a free slot, grow into the reserved pool, or evict a completed
// sequence. Every other active sequence remains pinned, even between appends.
// Finally report memory, whole-sequence evictions and accumulated phase timings.

// ============================================================================
// 1. Standard library dependencies
// ============================================================================

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

// ============================================================================
// 2. Model layout and memory limits
// ============================================================================

using Clock = std::chrono::steady_clock;

static_assert(sizeof(std::size_t) >= 8, "Use a 64-bit build for an 8 GiB cache.");
constexpr std::size_t GIB = 1024ULL * 1024 * 1024;
constexpr std::size_t MIB = 1024ULL * 1024;
constexpr std::size_t CAP_BYTES = 8 * GIB;
constexpr std::size_t LAYERS = 10;
constexpr std::size_t KV_HEADS = 32;
constexpr std::size_t HEAD_DEPTH = 128;
constexpr std::size_t VALUES_PER_TOKEN = 2 * LAYERS * KV_HEADS * HEAD_DEPTH;
constexpr std::size_t BYTES_PER_TOKEN = VALUES_PER_TOKEN * sizeof(std::uint16_t);

// ============================================================================
// 3. Sequence-aware KV cache
// ============================================================================

// Each token occupies one fixed-size slot. Sequence-owned slot lists preserve
// token order even when slots belonging to completed requests are reused.
// All running sequences are pinned through both prefill and decoding.
class SequenceKVCache {
public:
    SequenceKVCache(std::size_t token_capacity, std::size_t values_per_token)
        : token_capacity_(token_capacity), values_per_token_(values_per_token) {
        if (token_capacity == 0 || values_per_token == 0 ||
            token_capacity > storage_.max_size() / values_per_token) {
            throw std::invalid_argument("Invalid KV cache capacity.");
        }
        const std::size_t max_values = token_capacity * values_per_token;
        // Reserve once, avoiding geometric growth and transient KV copies.
        storage_.reserve(max_values);
        if (storage_.capacity() > max_values) {
            throw std::runtime_error("KV allocation exceeds the cache budget.");
        }
        free_slots_.reserve(token_capacity);
    }

    // Sequence lifecycle: starting pins KV; finishing makes it reclaimable.
    void start_sequence(std::size_t id) {
        if (!sequences_.try_emplace(id).second) {
            throw std::invalid_argument("Sequence already has a KV cache.");
        }
    }

    void finish_sequence(std::size_t id) {
        auto& sequence = sequences_.at(id);
        if (!sequence.active) {
            throw std::invalid_argument("Sequence has already finished.");
        }
        // Completion makes its slots eligible for eviction, but retains them
        // until memory pressure actually requires their reuse.
        sequence.active = false;
    }

    // Token storage: reuse free slots and evict completed sequences as needed.
    void append_token(std::size_t id, std::uint16_t value = 0x3c00) {
        auto& sequence = sequences_.at(id);
        if (!sequence.active) {
            throw std::invalid_argument("Cannot append KV to a completed sequence.");
        }

        if (free_slots_.empty() && storage_.size() / values_per_token_ == token_capacity_) {
            if (!evict_completed_sequence()) {
                // No mutation: every resident token still belongs to a running
                // request. Its context must survive a failed allocation.
                throw std::runtime_error(
                    "KV cache is full of active sequences; their histories were preserved.");
            }
        }

        const std::size_t slot = free_slots_.empty()
            ? storage_.size() / values_per_token_ : free_slots_.back();
        sequence.slots.push_back(slot);
        if (free_slots_.empty()) {
            // uint16_t holds FP16 bit patterns; 0x3c00 represents FP16 1.0.
            storage_.resize(storage_.size() + values_per_token_, value);
        } else {
            std::fill_n(storage_.data() + slot * values_per_token_, values_per_token_, value);
            free_slots_.pop_back();
        }
        ++used_tokens_;
        peak_tokens_ = std::max(peak_tokens_, used_tokens_);
    }

    // Cache inspection and memory accounting.
    bool contains(std::size_t id) const { return sequences_.contains(id); }

    std::size_t cached_tokens(std::size_t id) const {
        return sequences_.at(id).slots.size();
    }

    // A real attention implementation would read a sequence's K/V via these
    // positions. Spans remain valid while their owning sequence is active.
    std::span<const std::uint16_t> token(std::size_t id, std::size_t position) const {
        const auto slot = sequences_.at(id).slots.at(position);
        return {storage_.data() + slot * values_per_token_, values_per_token_};
    }

    std::size_t used_bytes() const {
        return used_tokens_ * values_per_token_ * sizeof(std::uint16_t);
    }
    std::size_t peak_bytes() const {
        return peak_tokens_ * values_per_token_ * sizeof(std::uint16_t);
    }
    std::size_t reserved_bytes() const { return storage_.capacity() * sizeof(std::uint16_t); }
    std::size_t evictions() const { return evictions_; }

private:
    struct Sequence {
        bool active = true;
        std::vector<std::size_t> slots;
    };

    // Eviction releases one completed sequence's slots for other requests.
    bool evict_completed_sequence() {
        const auto victim = std::find_if(sequences_.begin(), sequences_.end(),
            [](const auto& entry) { return !entry.second.active && !entry.second.slots.empty(); });
        if (victim == sequences_.end()) {
            return false;
        }
        const auto& slots = victim->second.slots;
        free_slots_.insert(free_slots_.end(), slots.begin(), slots.end());
        used_tokens_ -= slots.size();
        sequences_.erase(victim);
        ++evictions_;
        return true;
    }

    std::size_t token_capacity_;
    std::size_t values_per_token_;
    std::vector<std::uint16_t> storage_;
    std::vector<std::size_t> free_slots_;
    std::map<std::size_t, Sequence> sequences_;
    std::size_t used_tokens_ = 0;
    std::size_t peak_tokens_ = 0;
    std::size_t evictions_ = 0;
};

// ============================================================================
// 4. Reporting and argument helpers
// ============================================================================

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

// ============================================================================
// 5. Available memory and sequence admission
// ============================================================================

std::optional<std::size_t> available_memory_bytes() {
#if defined(__linux__)
    // WSL reports the VM's available RAM here, rather than the host's RAM.
    std::ifstream meminfo("/proc/meminfo");
    std::string name, unit;
    std::size_t kib = 0;
    while (meminfo >> name >> kib >> unit) {
        if (name == "MemAvailable:" && unit == "kB" &&
            kib <= std::numeric_limits<std::size_t>::max() / 1024) {
            return kib * 1024;
        }
    }
#endif
    // If RAM information is unavailable, allocation failures still cause
    // smaller groups to be tried before any request begins.
    return std::nullopt;
}

struct CachePlan {
    std::size_t max_active;
    std::size_t reserved_tokens;
};

CachePlan plan_cache(std::size_t batch, std::size_t input_tokens, std::size_t output_tokens,
                     std::size_t cap_bytes, std::optional<std::size_t> available_bytes) {
    const std::size_t configured_budget = cap_bytes * 95 / 100;
    const std::size_t configured_tokens = configured_budget / BYTES_PER_TOKEN;
    // Validate before adding or multiplying: a running sequence's full history
    // must fit, even when it is the only admitted request.
    if (input_tokens > configured_tokens || output_tokens > configured_tokens - input_tokens) {
        throw std::invalid_argument(
            "One complete sequence (input + output tokens) exceeds the 95% KV budget. "
            "Increase --cap-mib or reduce token counts; active KV is never evicted.");
    }
    const std::size_t sequence_tokens = input_tokens + output_tokens;

    // The configured cap is an upper bound, not a requirement to allocate it.
    // Leave 10% of available RAM for bookkeeping and other system activity.
    const std::size_t budget_bytes = available_bytes
        ? std::min(configured_budget, *available_bytes - *available_bytes / 10)
        : configured_budget;
    const std::size_t token_capacity = budget_bytes / BYTES_PER_TOKEN;
    if (sequence_tokens > token_capacity) {
        throw std::runtime_error(
            "One complete sequence needs " + std::to_string(sequence_tokens * BYTES_PER_TOKEN / MIB)
            + " MiB of KV, but the available-memory budget is only "
            + std::to_string(budget_bytes / MIB)
            + " MiB. Reduce token counts or increase WSL/container memory; active KV is never evicted.");
    }
    const std::size_t max_active = std::min(batch, token_capacity / sequence_tokens);
    const std::size_t reserved_tokens = batch <= token_capacity / sequence_tokens
        ? batch * sequence_tokens : token_capacity;
    return {max_active, reserved_tokens};
}

// ============================================================================
// 6. Batch scheduling, prefill and decoding
// ============================================================================

void run_batch(std::size_t batch, std::size_t input_tokens, std::size_t output_tokens,
               std::size_t cap_bytes) {
    const auto plan = plan_cache(batch, input_tokens, output_tokens, cap_bytes,
                                 available_memory_bytes());
    const std::size_t sequence_tokens = input_tokens + output_tokens;
    std::size_t max_active = plan.max_active;
    std::size_t reserved_tokens = plan.reserved_tokens;

    const auto start = Clock::now();
    // Layout: [slot][K/V][layer][head][depth]; sequence metadata owns slots.
    std::optional<SequenceKVCache> allocation;
    while (!allocation) {
        try {
            allocation.emplace(reserved_tokens, VALUES_PER_TOKEN);
        } catch (const std::bad_alloc&) {
            // Commit/address-space limits can be tighter than available RAM.
            // First drop space for retaining completed KV, then reduce active
            // requests. These retries occur before any sequence has started.
            if (reserved_tokens > max_active * sequence_tokens) {
                reserved_tokens = max_active * sequence_tokens;
            } else if (max_active > 1) {
                max_active /= 2;
                reserved_tokens = max_active * sequence_tokens;
            } else {
                throw std::runtime_error(
                    "Cannot allocate even one complete sequence's "
                    + std::to_string(sequence_tokens * BYTES_PER_TOKEN / MIB)
                    + " MiB KV history. Reduce token counts or increase WSL/container memory.");
            }
        }
    }
    auto& cache = *allocation;
    double prefill_ms = milliseconds(start, Clock::now());
    double decode_ms = 0.0;
    std::size_t prefill_peak_bytes = 0;
    std::size_t prefill_evictions = 0;
    std::size_t decode_evictions = 0;
    std::size_t waves = 0;

    for (std::size_t completed = 0; completed < batch;) {
        const std::size_t active_count = std::min(max_active, batch - completed);
        ++waves;
        const auto prefill_start = Clock::now();
        for (std::size_t sequence = 0; sequence < active_count; ++sequence) {
            cache.start_sequence(completed + sequence);
        }
        const auto process_tokens = [&](std::size_t count) {
            for (std::size_t token = 0; token < count; ++token) {
                for (std::size_t sequence = 0; sequence < active_count; ++sequence) {
                    cache.append_token(completed + sequence);
                }
            }
        };

        // Prefill: store prompt KV while all admitted sequences stay active.
        std::size_t previous_evictions = cache.evictions();
        process_tokens(input_tokens);
        const auto prefill_end = Clock::now();
        prefill_ms += milliseconds(prefill_start, prefill_end);
        prefill_peak_bytes = std::max(prefill_peak_bytes, cache.used_bytes());
        prefill_evictions += cache.evictions() - previous_evictions;

        // Decode: keep prompt and generated KV pinned until generation finishes.
        previous_evictions = cache.evictions();
        process_tokens(output_tokens);
        for (std::size_t sequence = 0; sequence < active_count; ++sequence) {
            cache.finish_sequence(completed + sequence);
        }
        decode_ms += milliseconds(prefill_end, Clock::now());
        decode_evictions += cache.evictions() - previous_evictions;
        completed += active_count;
    }
    const auto end = Clock::now();

    // Reporting: summarize memory, completed-sequence evictions and phase times.
    const double uncapped_gib = (static_cast<double>(input_tokens)
                              + static_cast<double>(output_tokens))
                              * static_cast<double>(batch) * gib(BYTES_PER_TOKEN);
    std::cout << batch << ',' << input_tokens << ',' << output_tokens << ','
              << max_active << ',' << waves << ','
              << uncapped_gib << ',' << gib(prefill_peak_bytes) << ','
              << gib(cache.used_bytes()) << ',' << gib(cache.peak_bytes()) << ','
              << gib(cache.reserved_bytes()) << ','
              << prefill_evictions << ',' << decode_evictions << ','
              << prefill_ms << ',' << decode_ms << ','
              << milliseconds(start, end) << std::endl;
}

// ============================================================================
// 7. Command-line entry point and scenario sweep
// ============================================================================

// Unit tests include this file with the benchmark entry point disabled.
#ifndef KV_CACHE_NO_MAIN
int main(int argc, char* argv[]) {
    constexpr std::string_view usage =
        " [batch input_tokens output_tokens] [--cap-mib MiB]\n";
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << "Usage: " << argv[0] << usage
                  << "No arguments runs all 27 scenarios. Token counts are per sequence.\n"
                  << "Default cap: 8192 MiB (8 GiB); use a smaller cap for a low-memory run.\n"
                  << "Requests run in groups whose complete histories fit within 95% of the cap.\n"
                  << "Available Linux RAM and allocation limits can reduce group sizes further.\n"
                  << "A single oversized sequence is rejected; running KV is never evicted.\n";
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
                  << "# Cap: " << gib(cap_bytes) << " GiB; configured KV budget: "
                  << gib(cap_bytes * 95 / 100) << " GiB (95%).\n"
                  << "# Admission also respects 90% of available Linux RAM; allocation failures reduce groups before processing.\n"
                  << "# Active histories stay pinned through prefill and decode; only completed sequences are evicted.\n"
                  << "# Batch requests run in groups; max_active_sequences and waves describe admission.\n"
                  << "# Evicted slots are reused; the reserved KV pool is released at scenario end.\n"
                  << "# Prefill peak is the maximum resident KV at group prefill ends; timings sum all groups.\n"
                  << "# KV budget excludes bookkeeping; prefill includes reservation; timings exclude printing and final deallocation.\n"
                  << "batch,input_tokens,output_tokens,max_active_sequences,waves,uncapped_GiB,"
                     "prefill_peak_GiB,final_GiB,peak_GiB,reserved_GiB,prefill_evictions,decode_evictions,"
                     "prefill_ms,decode_ms,total_ms\n"
                  << std::flush;

        if (argc == 4) {
            run_batch(batch, input, output, cap_bytes);
        } else {//run scenarios for all combinations of batch sizes and token counts
            for (const std::size_t batch_size : {8, 16, 20}) {
                for (const std::size_t input_tokens : {2048, 4192, 8192, }) {
                    for (const std::size_t output_tokens : {2048, 4192, 8192, }) {
                        run_batch(batch_size, input_tokens, output_tokens, cap_bytes);
                    }
                }
            }
        }
    } catch (const std::bad_alloc&) {
        std::cerr << "Memory allocation failed; reduce batch/token counts or increase WSL/container memory.\n";
        return 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
#endif // KV_CACHE_NO_MAIN
