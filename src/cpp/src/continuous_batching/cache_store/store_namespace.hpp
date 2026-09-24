// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "continuous_batching/cache_store/hashing.hpp"
#include "continuous_batching/cache_store/store_channels.hpp"
#include "openvino/genai/prompt_cache.hpp"

namespace ov::genai::cache_store {

/** @brief What the host resolves a namespace from, besides its scope. */
struct StoreFingerprints {
    Digest128 model{0, 0};      ///< model configuration, graphs and weight sizes
    Digest128 tokenizer{0, 0};  ///< tokenizer graphs and configuration
    Digest128 spec{0, 0};       ///< cache layout, device, runtime version, hash and witness algorithms
};

/**
 * @brief Fingerprints the model folder and the cache specification.
 * @return False with @p reason set when the model has no folder to fingerprint (a model created in memory).
 */
bool compute_fingerprints(const std::filesystem::path& models_path,
                          const std::string& device,
                          const std::vector<StoreChannel>& channels,
                          StoreFingerprints& fingerprints,
                          std::string& reason);

/** @return "shared" or "session:<id>", the namespace key of a prompt cache scope. */
std::string scope_key(const PromptCacheScope& scope);

/**
 * @brief Canonical namespace bytes: "OVKVNS1\0", the scope key (32-bit length and bytes), then the model, tokenizer and
 * spec fingerprints (16 bytes each, lo then hi, little-endian). Opaque to providers.
 */
std::vector<std::uint8_t> make_namespace(const std::string& scope, const StoreFingerprints& fingerprints);

}  // namespace ov::genai::cache_store
