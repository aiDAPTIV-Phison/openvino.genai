// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "openvino/genai/cache_store/provider_abi.hpp"
#include "openvino/genai/cache_store_config.hpp"

namespace ov::genai::cache_store {

/** @return Readable name of a store ABI status, e.g. "ERR_NOT_FOUND". */
std::string status_to_string(OvKVCacheStatus status);

/**
 * @brief One open store instance of a provider (one namespace): the host-side seam over OvKVCacheStoreVTable.
 *
 * Every call goes straight to the plugin, so callers must not hold a host lock. Tail entries are only reachable when
 * the plugin's `struct_bytes` covers them and the matching capability is advertised; this is checked once, when the
 * instance is created.
 */
class StoreInstance {
public:
    StoreInstance(const OvKVCacheStoreVTable* vtable,
                  void* store,
                  OvKVCacheCapabilities capabilities,
                  std::string provider_name);
    ~StoreInstance();

    StoreInstance(const StoreInstance&) = delete;
    StoreInstance& operator=(const StoreInstance&) = delete;

    const std::string& provider_name() const {
        return m_provider_name;
    }

    /** @return Whether `has_blocks` may be called (OV_KVCACHE_CAP_EXISTS). */
    bool can_probe() const {
        return m_can_probe;
    }

    /** @return Whether the asynchronous entries may be called (OV_KVCACHE_CAP_ASYNC_IO). */
    bool is_async() const {
        return m_is_async;
    }

    OvKVCacheStatus has_blocks(const OvKVCacheBlockKey* keys, std::uint8_t* out_present, std::size_t count) const;
    OvKVCacheStatus read_blocks(const OvKVCacheStoreReadRequest* requests,
                                OvKVCacheStoreReadResult* out,
                                std::size_t count) const;
    OvKVCacheStatus write_blocks(const OvKVCacheStoreWriteRequest* requests,
                                 OvKVCacheStoreWriteResult* out,
                                 std::size_t count) const;
    OvKVCacheStatus drop_blocks(const OvKVCacheBlockKey* keys, std::size_t count) const;
    OvKVCacheStatus flush() const;
    OvKVCacheStatus stats(OvKVCacheStoreStats& out) const;

    OvKVCacheStatus read_blocks_async(const OvKVCacheStoreReadRequest* requests,
                                      OvKVCacheStoreReadResult* out,
                                      std::size_t count,
                                      OvKVCacheAsyncToken& token) const;
    OvKVCacheStatus write_blocks_async(const OvKVCacheStoreWriteRequest* requests,
                                       OvKVCacheStoreWriteResult* out,
                                       std::size_t count,
                                       OvKVCacheAsyncToken& token) const;
    OvKVCacheStatus wait(OvKVCacheAsyncToken token, std::uint32_t timeout_ms) const;
    OvKVCacheStatus cancel(OvKVCacheAsyncToken token) const;

private:
    const OvKVCacheStoreVTable* m_vtable;
    void* m_store;
    bool m_can_probe;
    bool m_is_async;
    std::string m_provider_name;
};

/**
 * @brief A store provider selected from a module that passed the handshake: a plugin library, or the built-in "file"
 * provider. Modules are loaded once and never unloaded, and provider instances are never destroyed, because several
 * pipelines of the process may use them.
 */
class StoreProvider {
public:
    /**
     * @brief Loads `config.library_path` (or takes the built-in module when it is empty), checks the handshake and
     * selects `config.provider`. Never throws for a plugin problem: when anything fails it logs a warning and returns
     * the built-in "file" provider. @p fallback_reason receives the reason, empty when no fallback happened.
     */
    static std::shared_ptr<StoreProvider> create(const CacheStoreConfig& config, std::string& fallback_reason);

    /** @return The built-in "file" provider. */
    static std::shared_ptr<StoreProvider> create_builtin();

    const std::string& name() const {
        return m_name;
    }

    /** @return The library the provider came from, or "built-in". */
    const std::string& origin() const {
        return m_origin;
    }

    bool is_builtin() const {
        return m_is_builtin;
    }

    OvKVCacheCapabilities capabilities() const {
        return m_capabilities;
    }

    struct OpenedStore {
        std::shared_ptr<StoreInstance> store;
        bool has_semantics_fingerprint = false;
        OvKVCacheContentHash semantics_fingerprint{0, 0};
    };

    /**
     * @brief Opens (or creates) the store instance of a namespace. Instances are shared: opening a namespace the process
     * already has open in the same directory returns the same instance.
     * @param error Receives the reason when the result holds no store.
     */
    OpenedStore open(const std::vector<std::uint8_t>& ns, const CacheStoreConfig& config, std::string& error) const;

    /** @return Result of the provider's `purge` for a namespace. */
    OvKVCacheStatus purge(const std::vector<std::uint8_t>& ns) const;

    /** @return Result of `register_host_memory`, or ERR_UNSUPPORTED when the capability is not advertised. */
    OvKVCacheStatus register_host_memory(void* base, std::uint64_t bytes) const;

private:
    StoreProvider(const OvKVCacheStoreProviderDescriptor* descriptor, std::string name, std::string origin, bool is_builtin);

    const OvKVCacheStoreProviderDescriptor* m_descriptor;
    std::string m_name;
    std::string m_origin;
    bool m_is_builtin;
    OvKVCacheCapabilities m_capabilities;
};

}  // namespace ov::genai::cache_store
