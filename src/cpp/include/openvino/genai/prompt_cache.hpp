// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <variant>

#include "openvino/core/except.hpp"

namespace ov::genai {

/** @brief Prompt cache mode: the request's KV cache blocks are written to the cache store by the store's write policy. */
struct ImplicitCache {
    bool operator==(const ImplicitCache&) const {
        return true;
    }
};

/** @brief Prompt cache mode: the request's blocks are written only when published explicitly with `push_cache`.
 * Reading from the store is not affected by the mode. */
struct ExplicitCache {
    bool operator==(const ExplicitCache&) const {
        return true;
    }
};

using PromptCacheMode = std::variant<ImplicitCache, ExplicitCache>;

/** @brief Prompt cache scope: blocks are written to the namespace shared by every request of the model. */
struct SharedScope {
    bool operator==(const SharedScope&) const {
        return true;
    }
};

/**
 * @brief Prompt cache scope: blocks are written to a namespace of their own for this session. The scope partitions
 * namespaces (and their capacity) but does not by itself restrict hits: a session request also reuses blocks of the
 * shared namespace.
 */
struct SessionScope {
    std::string session_id;

    bool operator==(const SessionScope& other) const {
        return session_id == other.session_id;
    }
};

using PromptCacheScope = std::variant<SharedScope, SessionScope>;

/** @brief Time-to-live of the store entries a request writes; an expired entry is a cache miss. */
class CacheTtl {
public:
    /** @return The default time-to-live, one hour. */
    static CacheTtl standard() {
        return CacheTtl(3600);
    }

    /** @return A time-to-live of @p seconds, which must be greater than 0. */
    static CacheTtl seconds(std::uint64_t seconds) {
        OPENVINO_ASSERT(seconds > 0, "CacheTtl::seconds requires a positive number of seconds");
        return CacheTtl(seconds);
    }

    /** @return A time-to-live that never expires. */
    static CacheTtl unlimited() {
        return CacheTtl(0);
    }

    bool is_unlimited() const {
        return m_seconds == 0;
    }

    /** @return Seconds an entry lives, 0 when unlimited. */
    std::uint64_t get_seconds() const {
        return m_seconds;
    }

    bool operator==(const CacheTtl& other) const {
        return m_seconds == other.m_seconds;
    }

private:
    explicit CacheTtl(std::uint64_t seconds) : m_seconds(seconds) {}

    std::uint64_t m_seconds;
};

/** @brief Request-level prompt cache policy. Always present and always valid; the default is implicit caching in the
 * shared scope with the standard time-to-live. It has an effect only when the pipeline has a cache store. */
struct PromptCachePolicy {
    PromptCacheMode mode = ImplicitCache{};
    PromptCacheScope scope = SharedScope{};
    CacheTtl ttl = CacheTtl::standard();
    /** Opt this request out of write-through: its blocks are persisted only when they are evicted from the device. */
    bool opt_out_write_through = false;

    bool is_explicit_only() const {
        return std::holds_alternative<ExplicitCache>(mode);
    }

    bool is_session_scoped() const {
        return std::holds_alternative<SessionScope>(scope);
    }

    void validate() const {
        if (const auto* session = std::get_if<SessionScope>(&scope)) {
            OPENVINO_ASSERT(!session->session_id.empty(), "prompt_cache SessionScope requires a non-empty session_id");
        }
    }

    bool operator==(const PromptCachePolicy& other) const {
        return mode == other.mode && scope == other.scope && ttl == other.ttl &&
               opt_out_write_through == other.opt_out_write_through;
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "PromptCachePolicy { mode: " << (is_explicit_only() ? "explicit" : "implicit") << ", scope: ";
        if (const auto* session = std::get_if<SessionScope>(&scope)) {
            oss << "session(" << session->session_id << ")";
        } else {
            oss << "shared";
        }
        oss << ", ttl: ";
        if (ttl.is_unlimited()) {
            oss << "unlimited";
        } else {
            oss << ttl.get_seconds() << " s";
        }
        oss << ", opt_out_write_through: " << std::boolalpha << opt_out_write_through << " }";
        return oss.str();
    }
};

/** @brief Cache tier a request's reused prompt tokens were served from. */
enum class CacheTier {
    None,   /**< No prompt token was reused. */
    Device, /**< The pipeline's own prefix cache in device memory. */
    Host,   /**< Host memory tier (not present in this pipeline). */
    Store   /**< The cache store provider. */
};

/**
 * @brief Cache facts of one request and their interpretation.
 *
 * Hit counts do not overlap: device + host + store hit tokens is the request's total reuse. Counts of several
 * requests add up when their performance metrics are combined.
 */
struct CacheDiagnostics {
    std::size_t prompt_tokens = 0;
    std::size_t device_hit_tokens = 0;
    std::size_t host_hit_tokens = 0;
    /** Prompt tokens served by the cache store; nonzero proves the store was exercised. */
    std::size_t store_hit_tokens = 0;
    /** Reconciliation rounds and keys probed with `has_blocks`. */
    std::size_t store_probe_rounds = 0;
    std::size_t store_probed_keys = 0;
    /** Blocks and bytes read back from the store (verified or not). */
    std::size_t store_read_blocks = 0;
    std::uint64_t store_read_bytes = 0;
    /** Blocks and bytes of this request queued for writing to the store. */
    std::size_t pushed_blocks = 0;
    std::uint64_t pushed_bytes = 0;
    /** Batched transfers that failed as a whole and were retried block by block. */
    std::size_t degraded_chunks = 0;
    /** Deepest tier that served tokens (the store extends a device hit), or None. */
    CacheTier served_by = CacheTier::None;
    /** Why the store served nothing, or why its reuse stopped early; empty when nothing limited it. */
    std::string miss_reason;

    /** Adds the counts of another request; keeps the deepest serving tier and the first reason. */
    CacheDiagnostics& operator+=(const CacheDiagnostics& other) {
        prompt_tokens += other.prompt_tokens;
        device_hit_tokens += other.device_hit_tokens;
        host_hit_tokens += other.host_hit_tokens;
        store_hit_tokens += other.store_hit_tokens;
        store_probe_rounds += other.store_probe_rounds;
        store_probed_keys += other.store_probed_keys;
        store_read_blocks += other.store_read_blocks;
        store_read_bytes += other.store_read_bytes;
        pushed_blocks += other.pushed_blocks;
        pushed_bytes += other.pushed_bytes;
        degraded_chunks += other.degraded_chunks;
        if (static_cast<int>(other.served_by) > static_cast<int>(served_by)) {
            served_by = other.served_by;
        }
        if (miss_reason.empty()) {
            miss_reason = other.miss_reason;
        }
        return *this;
    }

    /** @return One-line reading of the facts, e.g. "hit: store served 3584 of 4096 prompt tokens (device 0)". */
    std::string interpretation() const {
        std::ostringstream oss;
        const std::size_t reused = device_hit_tokens + host_hit_tokens + store_hit_tokens;
        if (served_by == CacheTier::None) {
            oss << "miss";
        } else {
            oss << "hit: " << (served_by == CacheTier::Store ? "store" : served_by == CacheTier::Host ? "host" : "device")
                << " served " << reused << " of " << prompt_tokens << " prompt tokens (device " << device_hit_tokens
                << ", store " << store_hit_tokens << ")";
        }
        if (!miss_reason.empty()) {
            oss << (served_by == CacheTier::None ? ": " : "; store stopped: ") << miss_reason;
        }
        return oss.str();
    }
};

/**
 * @brief What `push_cache` published into the cache store.
 *
 * Counts blocks this push wrote; blocks the store already held are not written again (the first write of a block stays).
 */
struct PushCacheOutcome {
    /** Channel groups that received blocks (e.g. "kv" and "linear_attention_state"). */
    std::size_t groups = 0;
    /** Blocks written, all groups. */
    std::size_t blocks = 0;
    /** Payload bytes written. */
    std::uint64_t bytes = 0;
    /** Order of this push among the pipeline's pushes, starting at 1. */
    std::uint64_t epoch = 0;
    /** Cache facts of the push's prefill (what the device and the store already supplied). */
    CacheDiagnostics diagnostics;
};

/**
 * @brief What `prefetch_cache` warmed: store blocks moved into the device prefix cache.
 */
struct PrefetchCacheOutcome {
    /** Prompt tokens the store supplied into the device prefix cache. */
    std::size_t tokens = 0;
    /** Blocks read from the store. */
    std::size_t blocks = 0;
    /** Cache facts of the warm-up; `miss_reason` says why nothing was warmed. */
    CacheDiagnostics diagnostics;
};

}  // namespace ov::genai
