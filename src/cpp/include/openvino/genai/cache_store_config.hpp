// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <sstream>
#include <string>

#include "openvino/core/except.hpp"

namespace ov::genai {

/**
 * @brief When a KV cache block is persisted by a tier-to-tier movement of the cache store.
 */
enum class WriteTrigger {
    OnEvict,           /**< When the block leaves the faster tier (its device block is reused), and for every block
                            still cached when the pipeline is destroyed. */
    Immediate,         /**< As soon as the block is complete (all its tokens are computed). */
    OnAccessSelective, /**< When a later request reuses the block from the faster tier, i.e. its reuse is proven. */
    ExplicitOnly       /**< Only blocks published by `ContinuousBatchingPipeline::push_cache`. */
};

inline std::string write_trigger_to_string(WriteTrigger trigger) {
    switch (trigger) {
    case WriteTrigger::OnEvict:
        return "on_evict";
    case WriteTrigger::Immediate:
        return "immediate";
    case WriteTrigger::OnAccessSelective:
        return "on_access_selective";
    case WriteTrigger::ExplicitOnly:
        return "explicit_only";
    }
    OPENVINO_THROW("Unknown WriteTrigger value ", static_cast<int>(trigger));
}

inline WriteTrigger write_trigger_from_string(const std::string& name) {
    for (WriteTrigger trigger : {WriteTrigger::OnEvict,
                                 WriteTrigger::Immediate,
                                 WriteTrigger::OnAccessSelective,
                                 WriteTrigger::ExplicitOnly}) {
        if (write_trigger_to_string(trigger) == name) {
            return trigger;
        }
    }
    OPENVINO_THROW("Unknown cache store write trigger '", name,
                   "', expected on_evict, immediate, on_access_selective or explicit_only");
}

/**
 * @brief Cache store tier: persists KV cache blocks through a store provider (a plugin library implementing
 * openvino/genai/cache_store/provider_abi.hpp) and serves them back to later requests, also across process restarts.
 *
 * The pipeline owns the content identity of every block (tokenization, 128-bit content keys, witnesses), the prefix
 * decision, verification and admission into the device cache; the provider only stores opaque blocks. A provider that
 * fails to load or open disables the store tier with a warning, and store failures never abort a request: they make
 * the affected blocks a cache miss.
 *
 * A provider without asynchronous I/O is called on the inference thread, which waits for every transfer; an
 * asynchronous provider writes in the background.
 */
struct CacheStoreConfig {
    enum class Durability {
        Relaxed, /**< Pending writes are flushed when the pipeline is destroyed. */
        Strict   /**< Every write batch is flushed before it counts as written. */
    };

    /** @brief Write policy of one tier-to-tier movement. */
    struct MovementPolicy {
        /** Immediate by default: a block is in the store as soon as it is complete. */
        WriteTrigger trigger = WriteTrigger::Immediate;

        bool operator==(const MovementPolicy& other) const {
            return trigger == other.trigger;
        }
    };

    /** @brief Engine-level write policy: when a movement between tiers is persisted. */
    struct WritePolicy {
        MovementPolicy device_to_store;
        /** Accepted for completeness; this pipeline has no host tier, so the setting has no effect. */
        MovementPolicy host_to_store;

        bool operator==(const WritePolicy& other) const {
            return device_to_store == other.device_to_store && host_to_store == other.host_to_store;
        }
    };

    /** Master switch of the store tier. */
    bool enable = false;
    /** Location passed to the provider's `open`. */
    std::string directory;
    /** Capacity budget passed to the provider's `open`; 0 lets the provider decide. */
    std::uint64_t capacity_bytes = 0;
    /** Provider to select from the loaded module, e.g. "vendor-x"; empty selects the module's first one. */
    std::string provider;
    /** Path of the store plugin library. Required. */
    std::string library_path;
    /** Size of the host staging pool every store transfer goes through. */
    std::size_t staging_bytes = 32 * 1024 * 1024;
    /** Number of host I/O threads that call the provider. */
    std::size_t io_threads = 4;
    Durability durability = Durability::Relaxed;
    /** Verify every block read back against the tokens its key was derived from. Must stay true. */
    bool verify_key_tokens = true;
    /**
     * Also store a checksum (CRC32C) of every block's bytes and verify it on read. On by default: the witness proves
     * which content a block belongs to, not that its bytes are intact, so without the checksum a store that returns
     * damaged bytes under an intact witness is served as a hit. Detects accidental damage, not a deliberate forgery.
     */
    bool verify_payload_crc = true;
    /**
     * Bytes copied between the device cache and staging for store writes in one pipeline step, for a provider with
     * asynchronous I/O. A synchronous provider is written in full when the blocks are due.
     */
    std::size_t step_transfer_budget_bytes = 16 * 1024 * 1024;
    WritePolicy write;
    /** Free-form provider settings, forwarded to the provider's `open` as key / value strings. */
    std::map<std::string, std::string> properties;

    void validate() const {
        if (!enable) {
            return;
        }
        OPENVINO_ASSERT(verify_key_tokens,
                        "CacheStoreConfig verify_key_tokens must be true: blocks read from a store are always verified "
                        "against the tokens their key was derived from");
        OPENVINO_ASSERT(io_threads > 0, "CacheStoreConfig io_threads must be greater than 0");
        OPENVINO_ASSERT(staging_bytes > 0, "CacheStoreConfig staging_bytes must be greater than 0");
        OPENVINO_ASSERT(step_transfer_budget_bytes > 0,
                        "CacheStoreConfig step_transfer_budget_bytes must be greater than 0");
        OPENVINO_ASSERT(!library_path.empty(),
                        "CacheStoreConfig library_path is required: the store provider is a plugin library");
    }

    bool operator==(const CacheStoreConfig& other) const {
        return enable == other.enable && directory == other.directory && capacity_bytes == other.capacity_bytes &&
               provider == other.provider && library_path == other.library_path &&
               staging_bytes == other.staging_bytes && io_threads == other.io_threads &&
               durability == other.durability && verify_key_tokens == other.verify_key_tokens &&
               verify_payload_crc == other.verify_payload_crc &&
               step_transfer_budget_bytes == other.step_transfer_budget_bytes && write == other.write &&
               properties == other.properties;
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "CacheStoreConfig { \n";
        oss << "  enable: " << std::boolalpha << enable << "\n";
        oss << "  directory: " << directory << "\n";
        oss << "  capacity_bytes: " << capacity_bytes << "\n";
        oss << "  provider: " << provider << "\n";
        oss << "  library_path: " << library_path << "\n";
        oss << "  staging_bytes: " << staging_bytes << "\n";
        oss << "  io_threads: " << io_threads << "\n";
        oss << "  durability: " << (durability == Durability::Strict ? "strict" : "relaxed") << "\n";
        oss << "  verify_key_tokens: " << std::boolalpha << verify_key_tokens << "\n";
        oss << "  verify_payload_crc: " << std::boolalpha << verify_payload_crc << "\n";
        oss << "  step_transfer_budget_bytes: " << step_transfer_budget_bytes << "\n";
        oss << "  write.device_to_store.trigger: " << write_trigger_to_string(write.device_to_store.trigger) << "\n";
        oss << "  write.host_to_store.trigger: " << write_trigger_to_string(write.host_to_store.trigger) << "\n";
        oss << "  properties: {";
        const char* separator = "";
        for (const auto& [key, value] : properties) {
            oss << separator << key << ": " << value;
            separator = ", ";
        }
        oss << "}\n";
        oss << " }";
        return oss.str();
    }
};

}  // namespace ov::genai
