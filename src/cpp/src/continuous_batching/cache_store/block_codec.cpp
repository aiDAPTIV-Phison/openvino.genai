// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/cache_store/block_codec.hpp"

#include <algorithm>
#include <cstring>
#include <numeric>

#include "openvino/core/except.hpp"

namespace ov::genai::cache_store {

namespace {

/// Indices of @p blocks sorted by device index, so runs of consecutive device blocks are adjacent.
std::vector<std::size_t> sorted_by_device_index(const std::vector<BlockTransfer>& blocks) {
    std::vector<std::size_t> order(blocks.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&blocks](std::size_t lhs, std::size_t rhs) {
        return blocks[lhs].device_index < blocks[rhs].device_index;
    });
    return order;
}

std::size_t run_length(const std::vector<BlockTransfer>& blocks, const std::vector<std::size_t>& order, std::size_t start) {
    std::size_t run = 1;
    while (start + run < order.size() &&
           blocks[order[start + run]].device_index == blocks[order[start + run - 1]].device_index + 1) {
        ++run;
    }
    return run;
}

/// The device blocks [first, first + run) of a cache tensor, as a view.
ov::Tensor block_range(const ov::Tensor& cache, std::size_t first, std::size_t run) {
    ov::Coordinate begin(cache.get_shape().size(), 0);
    ov::Coordinate end = cache.get_shape();
    begin[0] = first;
    end[0] = first + run;
    return ov::Tensor(cache, begin, end);
}

std::size_t checked_block_bytes(const ov::Tensor& cache, const ChannelTensor& tensor) {
    const std::size_t bytes_per_block = cache.get_byte_size() / cache.get_shape()[0];
    OPENVINO_ASSERT(bytes_per_block == tensor.block_bytes,
                    "Cache store: ", tensor.name, " holds ", bytes_per_block, " bytes per block, but its channel "
                    "describes ", tensor.block_bytes);
    return bytes_per_block;
}

}  // namespace

void BlockCodec::download(const StoreChannel& channel,
                          const std::vector<ov::Tensor>& cache_tensors,
                          const std::vector<BlockTransfer>& blocks) {
    if (blocks.empty()) {
        return;
    }
    OPENVINO_ASSERT(cache_tensors.size() == channel.tensors.size(),
                    "Cache store: channel ", channel.group, " has ", channel.tensors.size(), " tensors, the cache ",
                    cache_tensors.size());
    const std::vector<std::size_t> order = sorted_by_device_index(blocks);
    for (std::size_t t = 0; t < cache_tensors.size(); ++t) {
        const ov::Tensor& cache = cache_tensors[t];
        const ChannelTensor& tensor = channel.tensors[t];
        const std::size_t bytes_per_block = checked_block_bytes(cache, tensor);
        for (std::size_t start = 0; start < order.size();) {
            const std::size_t run = run_length(blocks, order, start);
            ov::Shape run_shape = cache.get_shape();
            run_shape[0] = run;
            const ov::Tensor source = block_range(cache, blocks[order[start]].device_index, run);
            if (run == 1) {
                ov::Tensor destination(cache.get_element_type(), run_shape, blocks[order[start]].payload + tensor.payload_offset);
                source.copy_to(destination);
            } else {
                // payloads are block-major, the device range is tensor-major: copy the range out, then scatter it
                m_scratch.resize(std::max(m_scratch.size(), run * bytes_per_block));
                ov::Tensor destination(cache.get_element_type(), run_shape, m_scratch.data());
                source.copy_to(destination);
                for (std::size_t i = 0; i < run; ++i) {
                    std::memcpy(blocks[order[start + i]].payload + tensor.payload_offset,
                                m_scratch.data() + i * bytes_per_block,
                                bytes_per_block);
                }
            }
            start += run;
        }
    }
}

void BlockCodec::upload(const StoreChannel& channel,
                        const std::vector<ov::Tensor>& cache_tensors,
                        const std::vector<BlockTransfer>& blocks) {
    if (blocks.empty()) {
        return;
    }
    OPENVINO_ASSERT(cache_tensors.size() == channel.tensors.size(),
                    "Cache store: channel ", channel.group, " has ", channel.tensors.size(), " tensors, the cache ",
                    cache_tensors.size());
    const std::vector<std::size_t> order = sorted_by_device_index(blocks);
    for (std::size_t t = 0; t < cache_tensors.size(); ++t) {
        const ov::Tensor& cache = cache_tensors[t];
        const ChannelTensor& tensor = channel.tensors[t];
        const std::size_t bytes_per_block = checked_block_bytes(cache, tensor);
        for (std::size_t start = 0; start < order.size();) {
            const std::size_t run = run_length(blocks, order, start);
            ov::Shape run_shape = cache.get_shape();
            run_shape[0] = run;
            ov::Tensor destination = block_range(cache, blocks[order[start]].device_index, run);
            if (run == 1) {
                const ov::Tensor source(cache.get_element_type(), run_shape, blocks[order[start]].payload + tensor.payload_offset);
                source.copy_to(destination);
            } else {
                m_scratch.resize(std::max(m_scratch.size(), run * bytes_per_block));
                for (std::size_t i = 0; i < run; ++i) {
                    std::memcpy(m_scratch.data() + i * bytes_per_block,
                                blocks[order[start + i]].payload + tensor.payload_offset,
                                bytes_per_block);
                }
                const ov::Tensor source(cache.get_element_type(), run_shape, m_scratch.data());
                source.copy_to(destination);
            }
            start += run;
        }
    }
}

}  // namespace ov::genai::cache_store
