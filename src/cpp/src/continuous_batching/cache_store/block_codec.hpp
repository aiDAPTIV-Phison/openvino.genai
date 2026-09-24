// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "continuous_batching/cache_store/store_channels.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::genai::cache_store {

/** @brief One block to copy: its index in the device cache tensors (block or state row) and its host payload. */
struct BlockTransfer {
    std::size_t device_index = 0;
    std::uint8_t* payload = nullptr;
};

/**
 * @brief Moves whole blocks of one channel between the device cache tensors and canonical host payloads (every tensor's
 * slice of the block, concatenated in channel order). Runs of consecutive device indices move with one copy per tensor.
 * Call only on the pipeline's step thread: cache tensor handles are only stable there.
 */
class BlockCodec {
public:
    /// Copies the listed device blocks into their payloads.
    void download(const StoreChannel& channel,
                  const std::vector<ov::Tensor>& cache_tensors,
                  const std::vector<BlockTransfer>& blocks);

    /// Copies the payloads into the listed device blocks.
    void upload(const StoreChannel& channel,
                const std::vector<ov::Tensor>& cache_tensors,
                const std::vector<BlockTransfer>& blocks);

private:
    std::vector<std::uint8_t> m_scratch;
};

}  // namespace ov::genai::cache_store
