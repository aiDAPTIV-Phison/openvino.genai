// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/cache_store/cache_store_tier.hpp"

#include <filesystem>

#include "continuous_batching/cache/cache_orchestrator.hpp"
#include "logger.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov::genai::cache_store {

namespace {

Digest128 chain_root(const std::vector<std::uint8_t>& ns, const StoreProvider::OpenedStore& opened) {
    Murmur3Builder root;
    root.add("ov.genai.kvstore.root.v1").add(ns.data(), ns.size());
    root.add_u64(opened.has_semantics_fingerprint ? 1 : 0).add(opened.semantics_fingerprint);
    return root.finish();
}

}  // namespace

std::unique_ptr<CacheStoreTier> CacheStoreTier::create(const SchedulerConfig& config,
                                                       std::shared_ptr<CacheOrchestrator> orchestrator,
                                                       const ov::AnyMap& properties,
                                                       const std::string& device) {
    const CacheStoreConfig& store_config = config.cache_store;
    if (!store_config.enable) {
        return nullptr;
    }
    OPENVINO_ASSERT(config.enable_prefix_caching,
                    "The cache store requires enable_prefix_caching: stored blocks extend the pipeline's prefix cache");
    OPENVINO_ASSERT(!config.use_cache_eviction,
                    "The cache store cannot be combined with cache eviction: evicting tokens changes a block after it "
                    "was stored");
    OPENVINO_ASSERT(!orchestrator->needs_per_layer_kv_block_indices(),
                    "The cache store requires the same KV cache block index in every layer");

    std::vector<StoreChannel> channels = describe_channels(*orchestrator);
    std::filesystem::path models_path;
    const auto models_path_it = properties.find(ov::cache_model_path.name());
    if (models_path_it != properties.end()) {
        models_path = models_path_it->second.as<std::filesystem::path>();
    }
    StoreFingerprints fingerprints;
    std::string reason;
    if (!compute_fingerprints(models_path, device, channels, fingerprints, reason)) {
        GENAI_WARN("The cache store is disabled: " + reason + ".");
        return nullptr;
    }

    std::string fallback_reason;
    std::shared_ptr<StoreProvider> provider = StoreProvider::create(store_config, fallback_reason);
    std::unique_ptr<CacheStoreTier> tier(
        new CacheStoreTier(store_config, std::move(orchestrator), std::move(channels), fingerprints, std::move(provider)));
    if (!tier->open_namespace(scope_key(SharedScope{}), tier->m_shared)) {
        return nullptr;
    }
    GENAI_INFO("Cache store: provider '" + tier->m_provider->name() + "' (" + tier->m_provider->origin() + "), " +
               std::to_string(tier->m_channels.size()) + " channel(s)\n" + channels_to_string(tier->m_channels));
    return tier;
}

CacheStoreTier::CacheStoreTier(const CacheStoreConfig& config,
                               std::shared_ptr<CacheOrchestrator> orchestrator,
                               std::vector<StoreChannel> channels,
                               StoreFingerprints fingerprints,
                               std::shared_ptr<StoreProvider> provider)
    : m_config(config),
      m_orchestrator(std::move(orchestrator)),
      m_channels(std::move(channels)),
      m_fingerprints(fingerprints),
      m_provider(std::move(provider)) {}

CacheStoreTier::~CacheStoreTier() {
    if (m_shared.store) {
        const OvKVCacheStatus status = m_shared.store->flush();
        if (status != OV_KVCACHE_OK) {
            GENAI_WARN("Cache store flush failed: " + status_to_string(status));
        }
    }
}

bool CacheStoreTier::open_namespace(const std::string& scope, OpenNamespace& opened) {
    opened.scope = scope;
    opened.bytes = make_namespace(scope, m_fingerprints);
    std::string error;
    StoreProvider::OpenedStore store = m_provider->open(opened.bytes, m_config, error);
    if (!store.store && !m_provider->is_builtin()) {
        GENAI_WARN("Cache store provider '" + m_provider->name() + "' from " + m_provider->origin() +
                   " failed to open namespace '" + scope + "': " + error +
                   ". Falling back to the built-in \"file\" provider.");
        m_provider = StoreProvider::create_builtin();
        store = m_provider->open(opened.bytes, m_config, error);
    }
    if (!store.store) {
        GENAI_WARN("The cache store is disabled: the built-in \"file\" provider failed to open namespace '" + scope +
                   "' in '" + m_config.directory + "': " + error + ".");
        return false;
    }
    opened.store = store.store;
    opened.root = chain_root(opened.bytes, store);
    return true;
}

}  // namespace ov::genai::cache_store
