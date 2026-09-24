// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "continuous_batching/cache/cache_type.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::genai {
class CacheOrchestrator;
}

namespace ov::genai::cache_store {

/** @brief How a channel's blocks relate to token positions (the proposal's ChannelDescriptor selection semantics). */
enum class ChannelKind {
    PagedBlock, /**< One block per `block_size_tokens` tokens; a prefix needs every block from 0. */
    WholeState  /**< One state per `k * block_size_tokens` tokens; a prefix needs only the state at its end. */
};

/** @brief One cache tensor of a channel and its share of a block's canonical payload. */
struct ChannelTensor {
    std::string name;
    ov::element::Type element_type;
    ov::Shape block_shape;         ///< cache tensor shape without dimension 0 (the block index)
    std::size_t payload_offset = 0;  ///< offset of this tensor's bytes in a block's payload
    std::size_t block_bytes = 0;     ///< bytes of one block of this tensor
};

/**
 * @brief A cache group persisted by the store. To a plugin a channel exists only as the `group` string of its keys.
 *
 * The canonical payload of one block is the tensors' bytes of that block, concatenated in `tensors` order: key and
 * value of each layer for "kv", every state table for "linear_attention_state".
 */
struct StoreChannel {
    std::string group;
    ChannelKind kind = ChannelKind::PagedBlock;
    CacheType cache_type = CacheType::KV_CACHE;
    std::size_t block_size_tokens = 0;
    std::size_t block_bytes = 0;
    std::vector<ChannelTensor> tensors;
};

inline constexpr char KV_GROUP[] = "kv";
inline constexpr char LINEAR_ATTENTION_GROUP[] = "linear_attention_state";

/** @return The channels of the pipeline's caches: "kv", plus "linear_attention_state" for hybrid models. */
std::vector<StoreChannel> describe_channels(const CacheOrchestrator& orchestrator);

/** @return A canonical text of the channels, part of the spec fingerprint. */
std::string channels_to_string(const std::vector<StoreChannel>& channels);

}  // namespace ov::genai::cache_store
