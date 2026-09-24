// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "continuous_batching/cache_store/hashing.hpp"
#include "continuous_batching/cache_store/store_channels.hpp"
#include "continuous_batching/cache_store/store_namespace.hpp"
#include "continuous_batching/cache_store/store_provider.hpp"
#include "openvino/core/any.hpp"
#include "openvino/genai/scheduler_config.hpp"

namespace ov::genai {
class CacheOrchestrator;
}

namespace ov::genai::cache_store {

/**
 * @brief The cache store tier of one pipeline: the host side of the KV cache store proposal ("the connector" of
 * provider_abi.hpp). It owns content identity, namespaces, staging and I/O, and talks to one store provider.
 */
class CacheStoreTier {
public:
    /**
     * @brief Sets up the tier for a pipeline whose cache orchestrator exists.
     * @return The tier, or nullptr when `config.cache_store.enable` is false, or (with a warning) when no store can be
     * used for this model: it has no folder to fingerprint, or neither the configured nor the built-in provider opens.
     */
    static std::unique_ptr<CacheStoreTier> create(const SchedulerConfig& config,
                                                  std::shared_ptr<CacheOrchestrator> orchestrator,
                                                  const ov::AnyMap& properties,
                                                  const std::string& device);

    ~CacheStoreTier();

    CacheStoreTier(const CacheStoreTier&) = delete;
    CacheStoreTier& operator=(const CacheStoreTier&) = delete;

private:
    /// A namespace opened in the provider: its bytes, its store instance and the root of its key chains.
    struct OpenNamespace {
        std::string scope;
        std::vector<std::uint8_t> bytes;
        std::shared_ptr<StoreInstance> store;
        Digest128 root{0, 0};
    };

    CacheStoreTier(const CacheStoreConfig& config,
                   std::shared_ptr<CacheOrchestrator> orchestrator,
                   std::vector<StoreChannel> channels,
                   StoreFingerprints fingerprints,
                   std::shared_ptr<StoreProvider> provider);

    /// Opens the namespace of a scope; when the provider fails, falls back to the built-in "file" provider for good.
    bool open_namespace(const std::string& scope, OpenNamespace& opened);

    CacheStoreConfig m_config;
    std::shared_ptr<CacheOrchestrator> m_orchestrator;
    std::vector<StoreChannel> m_channels;
    StoreFingerprints m_fingerprints;
    std::shared_ptr<StoreProvider> m_provider;
    OpenNamespace m_shared;
};

}  // namespace ov::genai::cache_store
