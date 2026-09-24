// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/cache_store/store_channels.hpp"

#include <sstream>

#include "continuous_batching/cache/cache_orchestrator.hpp"
#include "continuous_batching/cache/kv_cache_manager.hpp"
#include "continuous_batching/cache/linear_attention_cache_manager.hpp"

namespace ov::genai::cache_store {

namespace {

ov::Shape get_block_shape(const ov::PartialShape& cache_shape, const std::string& name) {
    OPENVINO_ASSERT(cache_shape.rank().is_static() && cache_shape.rank().get_length() >= 2,
                    "Cache store: cache input ", name, " must have a static rank of at least 2");
    ov::Shape block_shape;
    for (std::int64_t dim = 1; dim < cache_shape.rank().get_length(); ++dim) {
        OPENVINO_ASSERT(cache_shape[dim].is_static(),
                        "Cache store: dimension ", dim, " of cache input ", name, " must be static");
        block_shape.push_back(static_cast<std::size_t>(cache_shape[dim].get_length()));
    }
    return block_shape;
}

void add_tensor(StoreChannel& channel,
                const std::string& name,
                ov::element::Type element_type,
                const ov::PartialShape& cache_shape) {
    ChannelTensor tensor;
    tensor.name = name;
    tensor.element_type = element_type;
    tensor.block_shape = get_block_shape(cache_shape, name);
    tensor.block_bytes = (ov::shape_size(tensor.block_shape) * element_type.bitwidth() + 7) / 8;
    tensor.payload_offset = channel.block_bytes;
    channel.block_bytes += tensor.block_bytes;
    channel.tensors.push_back(std::move(tensor));
}

}  // namespace

std::vector<StoreChannel> describe_channels(const CacheOrchestrator& orchestrator) {
    std::vector<StoreChannel> channels;
    if (orchestrator.has_kv_cache()) {
        const auto& kv_manager =
            static_cast<const KVCacheManager&>(orchestrator.get_cache_manager(CacheType::KV_CACHE));
        StoreChannel channel;
        channel.group = KV_GROUP;
        channel.kind = ChannelKind::PagedBlock;
        channel.cache_type = CacheType::KV_CACHE;
        channel.block_size_tokens = orchestrator.get_block_size(CacheType::KV_CACHE);
        for (std::size_t layer = 0; layer < kv_manager.get_num_layers(); ++layer) {
            add_tensor(channel, "key_cache." + std::to_string(layer), kv_manager.get_key_cache_precision(layer),
                       kv_manager.get_key_cache_shape(layer));
            add_tensor(channel, "value_cache." + std::to_string(layer), kv_manager.get_value_cache_precision(layer),
                       kv_manager.get_value_cache_shape(layer));
        }
        channels.push_back(std::move(channel));
    }
    if (orchestrator.has_linear_attention_cache()) {
        const auto& la_manager = static_cast<const LinearAttentionCacheManager&>(
            orchestrator.get_cache_manager(CacheType::LINEAR_ATTENTION_CACHE));
        StoreChannel channel;
        channel.group = LINEAR_ATTENTION_GROUP;
        channel.kind = ChannelKind::WholeState;
        channel.cache_type = CacheType::LINEAR_ATTENTION_CACHE;
        // the linear attention checkpoint interval: a state is committed every block_size_tokens tokens
        channel.block_size_tokens = orchestrator.get_block_size(CacheType::LINEAR_ATTENTION_CACHE);
        for (const auto& state_table : la_manager.get_state_table_infos()) {
            add_tensor(channel, state_table.name, state_table.precision, state_table.partial_shape);
        }
        channels.push_back(std::move(channel));
    }
    return channels;
}

std::string channels_to_string(const std::vector<StoreChannel>& channels) {
    std::ostringstream text;
    for (const StoreChannel& channel : channels) {
        text << "channel " << channel.group << " " << (channel.kind == ChannelKind::PagedBlock ? "paged" : "whole_state")
             << " tokens=" << channel.block_size_tokens << " bytes=" << channel.block_bytes << "\n";
        for (const ChannelTensor& tensor : channel.tensors) {
            text << "  " << tensor.name << " " << tensor.element_type.get_type_name() << " " << tensor.block_shape
                 << " @" << tensor.payload_offset << "\n";
        }
    }
    return text.str();
}

}  // namespace ov::genai::cache_store
