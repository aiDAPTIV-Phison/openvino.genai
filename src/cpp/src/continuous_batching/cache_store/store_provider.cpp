// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/cache_store/store_provider.hpp"

#include <cstddef>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <tuple>

#include "continuous_batching/cache_store/file_provider.hpp"
#include "gguf_utils/gguf_tokenizer.hpp"
#include "logger.hpp"
#include "openvino/genai/cache_store/sdk_fingerprint.h"

namespace ov::genai::cache_store {

namespace {

using EntryFunction = const OvKVCacheStoreModuleInfo* (*)();

/// A vtable entry exists for the host only when the producer's struct_bytes covers it (guide section 3).
template <typename VTable>
bool covers(const VTable* vtable, std::size_t entry_offset) {
    return vtable->struct_bytes >= entry_offset + sizeof(void*);
}

std::shared_ptr<void> load_module_library(const std::filesystem::path& path) {
    static std::mutex mutex;
    // Intentionally never destroyed: the host never unloads a store module (guide section 3)
    static auto* libraries = new std::map<std::filesystem::path, std::shared_ptr<void>>();
    std::lock_guard<std::mutex> lock(mutex);
    std::shared_ptr<void>& library = (*libraries)[path];
    if (!library) {
        library = load_shared_object(path);
    }
    return library;
}

std::string check_module(const OvKVCacheStoreModuleInfo* info) {
    if (!info) {
        return "the entry point returned no module descriptor";
    }
    if (info->abi_major != OV_KVCACHE_STORE_ABI_MAJOR) {
        return "store ABI major version " + std::to_string(info->abi_major) + " differs from the host's " +
               std::to_string(OV_KVCACHE_STORE_ABI_MAJOR);
    }
    if (info->abi_minor > OV_KVCACHE_STORE_ABI_MINOR) {
        return "store ABI minor version " + std::to_string(info->abi_minor) + " is newer than the host's " +
               std::to_string(OV_KVCACHE_STORE_ABI_MINOR);
    }
    if (!info->sdk_build_fingerprint ||
        std::strcmp(info->sdk_build_fingerprint, OV_KVCACHE_STORE_SDK_BUILD_FINGERPRINT) != 0) {
        return std::string("SDK build fingerprint '") +
               (info->sdk_build_fingerprint ? info->sdk_build_fingerprint : "(null)") + "' differs from the host's '" +
               OV_KVCACHE_STORE_SDK_BUILD_FINGERPRINT + "'";
    }
    if (info->provider_count == 0 || !info->get_provider) {
        return "the module has no providers";
    }
    return {};
}

std::string descriptor_name(const OvKVCacheStoreProviderDescriptor& descriptor) {
    if (descriptor.name) {
        return descriptor.name;
    }
    const char* name = descriptor.vtable->name ? descriptor.vtable->name(descriptor.self) : nullptr;
    return name ? name : "";
}

std::string check_provider(const OvKVCacheStoreProviderDescriptor* descriptor) {
    if (!descriptor || !descriptor->vtable) {
        return "provider descriptor without a vtable";
    }
    const OvKVCacheStoreProviderVTable* vtable = descriptor->vtable;
    if (!(descriptor->capabilities & OV_KVCACHE_CAP_KEY_IO)) {
        return "the required capability OV_KVCACHE_CAP_KEY_IO is not advertised";
    }
    if (!covers(vtable, offsetof(OvKVCacheStoreProviderVTable, list)) || !vtable->name || !vtable->open ||
        !vtable->purge || !vtable->list) {
        return "the provider vtable lacks mandatory entries (struct_bytes " + std::to_string(vtable->struct_bytes) + ")";
    }
    if ((descriptor->capabilities & OV_KVCACHE_CAP_REGISTER_HOST_MEMORY) &&
        (!covers(vtable, offsetof(OvKVCacheStoreProviderVTable, register_host_memory)) ||
         !vtable->register_host_memory)) {
        return "OV_KVCACHE_CAP_REGISTER_HOST_MEMORY is advertised without register_host_memory";
    }
    return {};
}

std::string check_store_vtable(const OvKVCacheStoreVTable* vtable, OvKVCacheCapabilities capabilities) {
    if (!covers(vtable, offsetof(OvKVCacheStoreVTable, stats)) || !vtable->destroy || !vtable->read_blocks ||
        !vtable->write_blocks || !vtable->drop_blocks || !vtable->flush || !vtable->stats) {
        return "the store vtable lacks synchronous entries (struct_bytes " + std::to_string(vtable->struct_bytes) + ")";
    }
    if ((capabilities & OV_KVCACHE_CAP_EXISTS) && !vtable->has_blocks) {
        return "OV_KVCACHE_CAP_EXISTS is advertised without has_blocks";
    }
    if ((capabilities & OV_KVCACHE_CAP_ASYNC_IO) &&
        (!covers(vtable, offsetof(OvKVCacheStoreVTable, cancel)) || !vtable->read_blocks_async ||
         !vtable->write_blocks_async || !vtable->wait || !vtable->cancel)) {
        return "OV_KVCACHE_CAP_ASYNC_IO is advertised without the asynchronous entries";
    }
    return {};
}

/// Store instances shared by every pipeline of the process, per provider, directory and namespace.
struct OpenStoreKey {
    const OvKVCacheStoreProviderDescriptor* descriptor;
    std::string directory;
    std::vector<std::uint8_t> ns;

    bool operator<(const OpenStoreKey& other) const {
        return std::tie(descriptor, directory, ns) < std::tie(other.descriptor, other.directory, other.ns);
    }
};

struct OpenStoreEntry {
    std::weak_ptr<StoreInstance> store;
    bool has_semantics_fingerprint = false;
    OvKVCacheContentHash semantics_fingerprint{0, 0};
};

std::mutex& open_stores_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::map<OpenStoreKey, OpenStoreEntry>& open_stores() {
    static auto* stores = new std::map<OpenStoreKey, OpenStoreEntry>();
    return *stores;
}

}  // namespace

std::string status_to_string(OvKVCacheStatus status) {
    switch (status) {
    case OV_KVCACHE_OK:
        return "OK";
    case OV_KVCACHE_PARTIAL:
        return "PARTIAL";
    case OV_KVCACHE_PENDING:
        return "PENDING";
    case OV_KVCACHE_ERR_INVALID_ARG:
        return "ERR_INVALID_ARG";
    case OV_KVCACHE_ERR_NOT_OPEN:
        return "ERR_NOT_OPEN";
    case OV_KVCACHE_ERR_NOT_FOUND:
        return "ERR_NOT_FOUND";
    case OV_KVCACHE_ERR_CAPACITY:
        return "ERR_CAPACITY";
    case OV_KVCACHE_ERR_IO:
        return "ERR_IO";
    case OV_KVCACHE_ERR_TIMEOUT:
        return "ERR_TIMEOUT";
    case OV_KVCACHE_ERR_UNSUPPORTED:
        return "ERR_UNSUPPORTED";
    case OV_KVCACHE_ERR_ABI_MISMATCH:
        return "ERR_ABI_MISMATCH";
    case OV_KVCACHE_ERR_BUFFER_TOO_SMALL:
        return "ERR_BUFFER_TOO_SMALL";
    case OV_KVCACHE_ERR_INTERNAL:
        return "ERR_INTERNAL";
    case OV_KVCACHE_ERR_CANCELLED:
        return "ERR_CANCELLED";
    default:
        return "unknown status " + std::to_string(status);
    }
}

StoreInstance::StoreInstance(const OvKVCacheStoreVTable* vtable,
                             void* store,
                             OvKVCacheCapabilities capabilities,
                             std::string provider_name)
    : m_vtable(vtable),
      m_store(store),
      m_can_probe((capabilities & OV_KVCACHE_CAP_EXISTS) != 0),
      m_is_async((capabilities & OV_KVCACHE_CAP_ASYNC_IO) != 0),
      m_provider_name(std::move(provider_name)) {}

StoreInstance::~StoreInstance() {
    // blocks until the plugin is done with every host buffer (guide section 7)
    m_vtable->destroy(m_store);
}

OvKVCacheStatus StoreInstance::has_blocks(const OvKVCacheBlockKey* keys, std::uint8_t* out_present, std::size_t count) const {
    OPENVINO_ASSERT(m_can_probe, "Cache store provider '", m_provider_name, "' does not implement has_blocks");
    return m_vtable->has_blocks(m_store, keys, out_present, count);
}

OvKVCacheStatus StoreInstance::read_blocks(const OvKVCacheStoreReadRequest* requests,
                                           OvKVCacheStoreReadResult* out,
                                           std::size_t count) const {
    return m_vtable->read_blocks(m_store, requests, out, count);
}

OvKVCacheStatus StoreInstance::write_blocks(const OvKVCacheStoreWriteRequest* requests,
                                            OvKVCacheStoreWriteResult* out,
                                            std::size_t count) const {
    return m_vtable->write_blocks(m_store, requests, out, count);
}

OvKVCacheStatus StoreInstance::drop_blocks(const OvKVCacheBlockKey* keys, std::size_t count) const {
    return m_vtable->drop_blocks(m_store, keys, count);
}

OvKVCacheStatus StoreInstance::flush() const {
    return m_vtable->flush(m_store);
}

OvKVCacheStatus StoreInstance::stats(OvKVCacheStoreStats& out) const {
    return m_vtable->stats(m_store, &out);
}

OvKVCacheStatus StoreInstance::read_blocks_async(const OvKVCacheStoreReadRequest* requests,
                                                 OvKVCacheStoreReadResult* out,
                                                 std::size_t count,
                                                 OvKVCacheAsyncToken& token) const {
    OPENVINO_ASSERT(m_is_async, "Cache store provider '", m_provider_name, "' has no asynchronous I/O");
    token = 0;
    return m_vtable->read_blocks_async(m_store, requests, out, count, &token);
}

OvKVCacheStatus StoreInstance::write_blocks_async(const OvKVCacheStoreWriteRequest* requests,
                                                  OvKVCacheStoreWriteResult* out,
                                                  std::size_t count,
                                                  OvKVCacheAsyncToken& token) const {
    OPENVINO_ASSERT(m_is_async, "Cache store provider '", m_provider_name, "' has no asynchronous I/O");
    token = 0;
    return m_vtable->write_blocks_async(m_store, requests, out, count, &token);
}

OvKVCacheStatus StoreInstance::wait(OvKVCacheAsyncToken token, std::uint32_t timeout_ms) const {
    OPENVINO_ASSERT(m_is_async, "Cache store provider '", m_provider_name, "' has no asynchronous I/O");
    return m_vtable->wait(m_store, token, timeout_ms);
}

OvKVCacheStatus StoreInstance::cancel(OvKVCacheAsyncToken token) const {
    OPENVINO_ASSERT(m_is_async, "Cache store provider '", m_provider_name, "' has no asynchronous I/O");
    return m_vtable->cancel(m_store, token);
}

StoreProvider::StoreProvider(const OvKVCacheStoreProviderDescriptor* descriptor,
                             std::string name,
                             std::string origin,
                             bool is_builtin)
    : m_descriptor(descriptor),
      m_name(std::move(name)),
      m_origin(std::move(origin)),
      m_is_builtin(is_builtin),
      m_capabilities(descriptor->capabilities) {}

std::shared_ptr<StoreProvider> StoreProvider::create_builtin() {
    const OvKVCacheStoreModuleInfo* info = builtin_file_store_module();
    const std::string module_error = check_module(info);
    OPENVINO_ASSERT(module_error.empty(), "Built-in cache store module: ", module_error);
    const OvKVCacheStoreProviderDescriptor* descriptor = info->get_provider(0);
    const std::string provider_error = check_provider(descriptor);
    OPENVINO_ASSERT(provider_error.empty(), "Built-in cache store provider: ", provider_error);
    return std::shared_ptr<StoreProvider>(new StoreProvider(descriptor, descriptor_name(*descriptor), "built-in", true));
}

std::shared_ptr<StoreProvider> StoreProvider::create(const CacheStoreConfig& config, std::string& fallback_reason) {
    fallback_reason.clear();
    if (config.uses_builtin_provider()) {
        return create_builtin();
    }

    try {
        std::shared_ptr<void> library = load_module_library(config.library_path);
        const auto entry = reinterpret_cast<EntryFunction>(get_symbol(library, OV_KVCACHE_STORE_ENTRY_SYMBOL));
        const OvKVCacheStoreModuleInfo* info = entry();
        fallback_reason = check_module(info);
        if (fallback_reason.empty()) {
            const OvKVCacheStoreProviderDescriptor* selected = nullptr;
            for (std::uint32_t index = 0; index < info->provider_count && !selected; ++index) {
                const OvKVCacheStoreProviderDescriptor* descriptor = info->get_provider(index);
                if (descriptor && descriptor->vtable &&
                    (config.provider.empty() || descriptor_name(*descriptor) == config.provider)) {
                    selected = descriptor;
                }
            }
            if (!selected) {
                fallback_reason = "the module has no provider named '" + config.provider + "'";
            } else {
                fallback_reason = check_provider(selected);
                if (fallback_reason.empty()) {
                    return std::shared_ptr<StoreProvider>(
                        new StoreProvider(selected, descriptor_name(*selected), config.library_path, false));
                }
            }
        }
    } catch (const std::exception& error) {
        fallback_reason = error.what();
    }

    GENAI_WARN("Cache store provider '" + config.provider + "' from " + config.library_path +
               " cannot be used: " + fallback_reason + ". Falling back to the built-in \"file\" provider.");
    return create_builtin();
}

StoreProvider::OpenedStore StoreProvider::open(const std::vector<std::uint8_t>& ns,
                                               const CacheStoreConfig& config,
                                               std::string& error) const {
    error.clear();
    const OpenStoreKey key{m_descriptor, config.directory, ns};
    {
        std::lock_guard<std::mutex> lock(open_stores_mutex());
        auto it = open_stores().find(key);
        if (it != open_stores().end()) {
            if (auto store = it->second.store.lock()) {
                return {store, it->second.has_semantics_fingerprint, it->second.semantics_fingerprint};
            }
        }
    }

    // The provider is called without any host lock held (guide section 9)
    std::vector<OvKVCacheProperty> properties;
    properties.reserve(config.properties.size());
    for (const auto& [property_key, property_value] : config.properties) {
        properties.push_back({property_key.c_str(), property_value.c_str()});
    }
    const OvKVCacheNamespace ns_view{ns.data(), ns.size()};
    OvKVCacheStoreOpenOptions options;
    std::memset(&options, 0, sizeof(options));
    options.directory = config.directory.c_str();
    options.capacity_bytes = config.capacity_bytes;
    options.properties = properties.empty() ? nullptr : properties.data();
    options.property_count = properties.size();
    OvKVCacheStoreOpenResult result;
    std::memset(&result, 0, sizeof(result));

    const OvKVCacheStatus status = m_descriptor->vtable->open(m_descriptor->self, &ns_view, &options, &result);
    if (status != OV_KVCACHE_OK || result.status != OV_KVCACHE_OK || !result.store || !result.vtable) {
        error = "open returned " + status_to_string(status != OV_KVCACHE_OK ? status : result.status);
        if (result.store && result.vtable && covers(result.vtable, offsetof(OvKVCacheStoreVTable, destroy)) &&
            result.vtable->destroy) {
            result.vtable->destroy(result.store);
        }
        return {};
    }
    error = check_store_vtable(result.vtable, m_capabilities);
    if (!error.empty()) {
        if (covers(result.vtable, offsetof(OvKVCacheStoreVTable, destroy)) && result.vtable->destroy) {
            result.vtable->destroy(result.store);
        }
        return {};
    }

    OpenedStore opened;
    opened.store = std::make_shared<StoreInstance>(result.vtable, result.store, m_capabilities, m_name);
    opened.has_semantics_fingerprint =
        (m_capabilities & OV_KVCACHE_CAP_SEMANTICS_FINGERPRINT) && result.has_semantics_fingerprint;
    opened.semantics_fingerprint =
        opened.has_semantics_fingerprint ? result.store_semantics_fingerprint : OvKVCacheContentHash{0, 0};

    std::shared_ptr<StoreInstance> duplicate;
    {
        std::lock_guard<std::mutex> lock(open_stores_mutex());
        OpenStoreEntry& entry = open_stores()[key];
        if (auto existing = entry.store.lock()) {
            // another pipeline opened the same namespace meanwhile: use its instance, release ours outside the lock
            duplicate = std::move(opened.store);
            opened = {existing, entry.has_semantics_fingerprint, entry.semantics_fingerprint};
        } else {
            entry = {opened.store, opened.has_semantics_fingerprint, opened.semantics_fingerprint};
        }
    }
    duplicate.reset();
    return opened;
}

OvKVCacheStatus StoreProvider::purge(const std::vector<std::uint8_t>& ns) const {
    const OvKVCacheNamespace ns_view{ns.data(), ns.size()};
    return m_descriptor->vtable->purge(m_descriptor->self, &ns_view);
}

OvKVCacheStatus StoreProvider::register_host_memory(void* base, std::uint64_t bytes) const {
    if (!(m_capabilities & OV_KVCACHE_CAP_REGISTER_HOST_MEMORY)) {
        return OV_KVCACHE_ERR_UNSUPPORTED;
    }
    return m_descriptor->vtable->register_host_memory(m_descriptor->self, base, bytes);
}

}  // namespace ov::genai::cache_store
