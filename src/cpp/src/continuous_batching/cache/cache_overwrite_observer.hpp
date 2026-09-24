// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#include "continuous_batching/cache/cache_type.hpp"

namespace ov::genai {

/**
 * @brief Observes cached blocks being reused for other content, i.e. evicted from the prefix cache.
 */
class ICacheOverwriteObserver {
public:
    virtual ~ICacheOverwriteObserver() = default;

    /// A cached block of @p type is reused for other content: its prefix hash and physical block index. Called during
    /// block allocation; must not call back into the cache orchestrator.
    virtual void on_overwrite(CacheType type, uint64_t hash, size_t block_index) = 0;

    /// Called before the scheduler's device writes of a step (state zeroing, block copies) and before the forward: the
    /// last moment the blocks reported by on_overwrite still hold their old content.
    virtual void before_block_writes() = 0;
};

}  // namespace ov::genai
