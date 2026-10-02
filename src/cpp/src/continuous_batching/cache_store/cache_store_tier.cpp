// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/cache_store/cache_store_tier.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <future>
#include <optional>
#include <tuple>

#include "continuous_batching/cache/cache_orchestrator.hpp"
#include "continuous_batching/cache/kv_cache_manager.hpp"
#include "continuous_batching/cache/linear_attention_cache_manager.hpp"
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

std::uint64_t unix_seconds_now() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count());
}

OvKVCacheBlockKey make_key(const StoreChannel& channel, const Digest128& hash) {
    OvKVCacheBlockKey key{};
    key.group = channel.group.c_str();
    key.group_len = static_cast<std::uint32_t>(channel.group.size());
    key.hash = hash;
    return key;
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
    OPENVINO_ASSERT(orchestrator->has_kv_cache(), "The cache store requires a model with a KV cache");

    std::vector<StoreChannel> channels = describe_channels(*orchestrator);
    for (const StoreChannel& channel : channels) {
        OPENVINO_ASSERT(channel.block_bytes <= store_config.staging_bytes,
                        "The cache store's staging_bytes (", store_config.staging_bytes, ") cannot hold one '",
                        channel.group, "' block of this model (", channel.block_bytes, " bytes)");
    }
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
    // Under any trigger but explicit_only a request can end up writing on eviction (its own policy can downgrade to it)
    if (store_config.write.device_to_store.trigger != WriteTrigger::ExplicitOnly) {
        tier->m_orchestrator->set_overwrite_observer(tier.get());
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
      m_provider(std::move(provider)),
      m_witness_key(load_witness_key()),
      m_staging(config.staging_bytes),
      m_io_pool(config.io_threads) {
    m_block_layout.witness_bytes_max = WitnessRecord::SIZE;
    for (const StoreChannel& channel : m_channels) {
        m_block_layout.groups.emplace_back(channel.group, channel.block_bytes);
        if (channel.cache_type == CacheType::KV_CACHE) {
            m_kv_channel = &channel;
        } else if (channel.cache_type == CacheType::LINEAR_ATTENTION_CACHE) {
            m_la_channel = &channel;
        }
    }
    if (m_provider->capabilities() & OV_KVCACHE_CAP_REGISTER_HOST_MEMORY) {
        const OvKVCacheStatus status = m_provider->register_host_memory(m_staging.base(), m_staging.capacity());
        if (status != OV_KVCACHE_OK) {
            GENAI_WARN("Cache store provider '" + m_provider->name() +
                       "' did not register the staging pool: " + status_to_string(status));
        }
    }
}

CacheStoreTier::~CacheStoreTier() {
    m_orchestrator->set_overwrite_observer(nullptr);
    // the device cache still exists: what on_evict still holds there is written now
    write_back_cached_blocks();
    m_io_pool.drain();
    // every batch is submitted now; the completion thread waits for the last ones to reach a terminal status
    {
        std::lock_guard<std::mutex> lock(m_inflight_mutex);
        m_stop_completion = true;
    }
    m_inflight_added.notify_all();
    if (m_completion_thread.joinable()) {
        m_completion_thread.join();
    }
    std::vector<const OpenNamespace*> namespaces{&m_shared};
    for (const auto& [scope, session] : m_sessions) {
        namespaces.push_back(session.get());
    }
    for (const OpenNamespace* ns : namespaces) {
        if (!ns->store) {
            continue;
        }
        const OvKVCacheStatus status = ns->store->flush();
        if (status != OV_KVCACHE_OK) {
            GENAI_WARN("Cache store flush of namespace '" + ns->scope + "' failed: " + status_to_string(status));
        }
    }
    const std::size_t failed_writes = m_failed_writes.load();
    if (failed_writes > 0) {
        GENAI_WARN("Cache store: " + std::to_string(failed_writes) + " block writes failed in this pipeline");
    }
}

bool CacheStoreTier::open_namespace(const std::string& scope, OpenNamespace& opened) {
    opened.scope = scope;
    opened.bytes = make_namespace(scope, m_fingerprints);
    std::string error;
    StoreProvider::OpenedStore store = m_provider->open(opened.bytes, m_config, m_block_layout, error);
    if (!store.store && !m_provider->is_builtin()) {
        GENAI_WARN("Cache store provider '" + m_provider->name() + "' from " + m_provider->origin() +
                   " failed to open namespace '" + scope + "': " + error +
                   ". Falling back to the built-in \"file\" provider.");
        m_provider = StoreProvider::create_builtin();
        store = m_provider->open(opened.bytes, m_config, m_block_layout, error);
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

const CacheStoreTier::OpenNamespace* CacheStoreTier::session_namespace(const std::string& scope, std::string& error) {
    {
        std::lock_guard<std::mutex> lock(m_sessions_mutex);
        const auto it = m_sessions.find(scope);
        if (it != m_sessions.end()) {
            return it->second.get();
        }
    }
    // Opened outside the lock; two requests opening the same new session get the provider's shared instance
    auto opened = std::make_unique<OpenNamespace>();
    opened->scope = scope;
    opened->bytes = make_namespace(scope, m_fingerprints);
    const StoreProvider::OpenedStore store = m_provider->open(opened->bytes, m_config, m_block_layout, error);
    if (!store.store) {
        return nullptr;
    }
    opened->store = store.store;
    opened->root = chain_root(opened->bytes, store);
    std::lock_guard<std::mutex> lock(m_sessions_mutex);
    std::unique_ptr<OpenNamespace>& entry = m_sessions[scope];
    if (!entry) {
        entry = std::move(opened);
    }
    return entry.get();
}

void CacheStoreTier::on_add_request(const SequenceGroup::Ptr& group) {
    CacheDiagnostics& diagnostics = group->get_cache_diagnostics();
    const GenerationConfig& sampling_params = group->get_sampling_parameters();
    ExpectedRequest expected = take_expected(group->get_request_id());
    if (group->num_total_seqs() != 1 || sampling_params.is_beam_search() || sampling_params.num_return_sequences != 1) {
        diagnostics.miss_reason = "unsupported request: the cache store serves single-sequence requests";
        RequestState unsupported(group, m_kv_channel->block_size_tokens, m_witness_key);
        unsupported.expected = std::move(expected);
        finish_expected(unsupported);
        return;
    }

    const std::size_t block_size = m_kv_channel->block_size_tokens;
    auto state = std::make_unique<RequestState>(group, block_size, m_witness_key);
    const PromptCachePolicy& policy = sampling_params.prompt_cache;
    select_namespaces(*state, policy);
    state->expected = std::move(expected);
    if (state->expected.push) {
        // push_cache publishes the whole sequence, including what the device prefix cache supplied (IfAbsent)
        state->write_mode = state->write_namespace ? WriteMode::Push : WriteMode::None;
    } else if (state->expected.prefetch) {
        state->write_mode = WriteMode::None;  // a warm-up creates nothing in the store
    } else {
        state->write_mode = state->write_namespace ? write_mode_of(policy) : WriteMode::None;
    }
    // A block the device cache supplied is never written again, even when its last token is recomputed: that rewrites
    // the block's quantization data, so its bytes would differ from the ones already stored under the same key
    state->write_cursor =
        state->write_mode == WriteMode::Push ? 0 : (group->get_num_processed_tokens() + block_size - 1) / block_size;
    // on_access_selective: the complete blocks the device cache supplied are the ones proven reused
    state->reused_blocks = group->get_num_processed_tokens() / block_size;
    plan_reload(*state);

    const std::uint64_t seq_id = (*group)[0]->get_id();
    std::lock_guard<std::mutex> lock(m_requests_mutex);
    m_requests[seq_id] = std::move(state);
}

void CacheStoreTier::select_namespaces(RequestState& state, const PromptCachePolicy& policy) {
    state.read_namespaces = {&m_shared};
    state.write_namespace = &m_shared;
    if (!policy.is_session_scoped()) {
        return;
    }
    std::string error;
    const std::string scope = scope_key(policy.scope);
    if (const OpenNamespace* session = session_namespace(scope, error)) {
        state.read_namespaces.insert(state.read_namespaces.begin(), session);
        state.write_namespace = session;
        return;
    }
    // a session's blocks never go to the shared namespace: the request only reads shared blocks
    state.write_namespace = nullptr;
    if (!m_reported_session_failure.exchange(true)) {
        GENAI_WARN("Cache store: the namespace of '" + scope + "' cannot be opened (" + error +
                   "); session requests read the shared namespace and write nothing. Further failures are not "
                   "reported.");
    }
}

CacheStoreTier::WriteMode CacheStoreTier::write_mode_of(const PromptCachePolicy& policy) const {
    if (policy.is_explicit_only()) {
        return WriteMode::None;  // an explicit-only request is written by push_cache only
    }
    switch (m_config.write.device_to_store.trigger) {
    case WriteTrigger::Immediate:
        return policy.opt_out_write_through ? WriteMode::OnEvict : WriteMode::Immediate;  // opt-out downgrades
    case WriteTrigger::OnAccessSelective:
        return policy.opt_out_write_through ? WriteMode::OnEvict : WriteMode::OnAccess;
    case WriteTrigger::OnEvict:
        return WriteMode::OnEvict;
    case WriteTrigger::ExplicitOnly:
        return WriteMode::None;
    }
    return WriteMode::None;
}

std::future<PushCacheOutcome> CacheStoreTier::expect_push(std::uint64_t request_id) {
    auto push = std::make_shared<PushTracker>();
    push->outcome.epoch = ++m_push_epoch;
    std::future<PushCacheOutcome> outcome = push->promise.get_future();
    std::lock_guard<std::mutex> lock(m_expected_mutex);
    m_expected[request_id].push = std::move(push);
    return outcome;
}

std::future<PrefetchCacheOutcome> CacheStoreTier::expect_prefetch(std::uint64_t request_id) {
    auto prefetch = std::make_shared<std::promise<PrefetchCacheOutcome>>();
    std::future<PrefetchCacheOutcome> outcome = prefetch->get_future();
    std::lock_guard<std::mutex> lock(m_expected_mutex);
    m_expected[request_id].prefetch = std::move(prefetch);
    return outcome;
}

void CacheStoreTier::cancel_expected(std::uint64_t request_id, std::exception_ptr error) {
    const ExpectedRequest expected = take_expected(request_id);
    if (expected.push) {
        expected.push->promise.set_exception(error);
    }
    if (expected.prefetch) {
        expected.prefetch->set_exception(error);
    }
}

CacheStoreTier::ExpectedRequest CacheStoreTier::take_expected(std::uint64_t request_id) {
    std::lock_guard<std::mutex> lock(m_expected_mutex);
    const auto it = m_expected.find(request_id);
    if (it == m_expected.end()) {
        return {};
    }
    ExpectedRequest expected = std::move(it->second);
    m_expected.erase(it);
    return expected;
}

void CacheStoreTier::settle_push(const std::shared_ptr<PushTracker>& push,
                                 const std::map<std::string, std::pair<std::size_t, std::uint64_t>>& written) {
    if (!push) {
        return;
    }
    std::lock_guard<std::mutex> lock(push->mutex);
    for (const auto& [group, counts] : written) {
        push->outcome.blocks += counts.first;
        push->outcome.bytes += counts.second;
        push->groups.insert(group);
    }
    push->pending_batches -= 1;
    if (push->request_freed && push->pending_batches == 0) {
        push->outcome.groups = push->groups.size();
        push->promise.set_value(push->outcome);
    }
}

void CacheStoreTier::finish_expected(RequestState& state) {
    const CacheDiagnostics& diagnostics = state.group->get_cache_diagnostics();
    if (state.expected.prefetch) {
        PrefetchCacheOutcome outcome;
        outcome.tokens = diagnostics.store_hit_tokens;
        outcome.blocks = diagnostics.store_read_blocks;
        outcome.diagnostics = diagnostics;
        if (outcome.tokens == 0 && outcome.diagnostics.miss_reason.empty()) {
            // the warm-up was cut to one token: the probes that sized it know why
            outcome.diagnostics.miss_reason = state.expected.miss_reason;
        }
        state.expected.prefetch->set_value(outcome);
        state.expected.prefetch.reset();
    }
    if (state.expected.push) {
        const std::shared_ptr<PushTracker> push = std::move(state.expected.push);
        std::lock_guard<std::mutex> lock(push->mutex);
        push->outcome.diagnostics = diagnostics;
        push->request_freed = true;
        // the writes queued so far complete it, or nothing is left to wait for
        if (push->pending_batches == 0) {
            push->outcome.groups = push->groups.size();
            push->promise.set_value(push->outcome);
        }
    }
}

std::size_t CacheStoreTier::prompt_length_to_keep(const SequenceGroup::Ptr& group) {
    const std::size_t prompt_len = group->get_prompt_len();
    {
        std::lock_guard<std::mutex> lock(m_expected_mutex);
        const auto it = m_expected.find(group->get_request_id());
        if (it == m_expected.end() || !it->second.prefetch) {
            return prompt_len;
        }
    }
    // Probes only: the prefix the store can serve from the first token, then the one token that is always computed
    RequestState probe_state(group, m_kv_channel->block_size_tokens, m_witness_key);
    select_namespaces(probe_state, group->get_sampling_parameters().prompt_cache);
    const std::optional<Discovery> found = discover(probe_state, 0);
    const std::size_t servable = found ? found->position : 0;
    if (servable == 0) {
        std::string reason = !found ? "the prompt holds no complete block"
                             : found->probe_failure.empty() ? "not in the store"
                                                            : "the store is unavailable: " + found->probe_failure;
        std::lock_guard<std::mutex> lock(m_expected_mutex);
        const auto it = m_expected.find(group->get_request_id());
        if (it != m_expected.end()) {
            it->second.miss_reason = std::move(reason);
        }
    }
    return std::min(prompt_len, servable + 1);
}

std::shared_ptr<const CacheStoreTier::KeyChain> CacheStoreTier::key_chain(RequestState& state,
                                                                           const OpenNamespace& ns,
                                                                           std::size_t num_blocks) const {
    auto chain = std::make_shared<KeyChain>();
    const std::size_t block_size = m_kv_channel->block_size_tokens;
    chain->kv_index.reserve(num_blocks);
    for (std::size_t block = 0; block < num_blocks; ++block) {
        chain->kv_index.push_back(chain->keys.size());
        chain->keys.push_back(make_key(*m_kv_channel, state.identity.key(ns.root, block, m_kv_channel->group)));
        if (m_la_channel && (block + 1) * block_size % m_la_channel->block_size_tokens == 0) {
            chain->la_index[block] = chain->keys.size();
            chain->keys.push_back(make_key(*m_la_channel, state.identity.key(ns.root, block, m_la_channel->group)));
        }
    }
    return chain;
}

std::vector<bool> CacheStoreTier::probe(RequestState& state,
                                        const OpenNamespace& ns,
                                        const std::vector<OvKVCacheBlockKey>& keys,
                                        bool& probed,
                                        std::string& failure) {
    std::vector<bool> present(keys.size(), true);
    probed = false;
    if (keys.empty() || !ns.store->can_probe()) {
        return present;  // without a probe every candidate is a candidate for the fetch, which verifies it
    }
    std::vector<std::uint8_t> out_present(keys.size(), 0);
    OvKVCacheStatus status = OV_KVCACHE_OK;
    m_io_pool
        .submit([&] {
            status = ns.store->has_blocks(keys.data(), out_present.data(), keys.size());
        })
        .get();
    state.group->get_cache_diagnostics().store_probed_keys += keys.size();
    if (status != OV_KVCACHE_OK && failure.empty()) {
        failure = "has_blocks returned " + status_to_string(status);
    }
    probed = status == OV_KVCACHE_OK;
    for (std::size_t i = 0; i < keys.size(); ++i) {
        present[i] = probed && out_present[i] == 1;
    }
    return present;
}

std::size_t CacheStoreTier::fetch(RequestState& state, std::vector<BlockRead>& reads) {
    for (BlockRead& item : reads) {
        item.request.key = make_key(*item.channel, item.request.key.hash);
        item.request.target.kind = OV_KVCACHE_TARGET_HOST;
        item.request.target.base = item.payload.data();
        item.request.target.bytes = item.channel->block_bytes;
        item.request.witness_out = {item.witness.data(), item.witness.size()};
    }
    // Batches of consecutive items of one namespace, one per I/O thread; each is an in-order run of the sequence, and
    // a state stays in the call of the KV block it ends in
    const std::size_t batch_size = std::max<std::size_t>(1, (reads.size() + m_config.io_threads - 1) / m_config.io_threads);
    std::vector<std::pair<std::size_t, std::size_t>> batches;
    for (std::size_t start = 0; start < reads.size();) {
        std::size_t end = start + 1;
        while (end < reads.size() && reads[end].ns == reads[start].ns &&
               (end - start < batch_size || reads[end].channel != m_kv_channel)) {
            ++end;
        }
        batches.emplace_back(start, end);
        start = end;
    }

    std::size_t degraded = 0;
    const bool is_async = std::all_of(reads.begin(), reads.end(), [](const BlockRead& item) {
        return item.ns->store->is_async();
    });
    if (!is_async) {
        std::vector<std::uint8_t> batch_degraded(batches.size(), 0);
        std::vector<std::future<void>> done;
        for (std::size_t b = 0; b < batches.size(); ++b) {
            done.push_back(m_io_pool.submit([this, &reads, &batches, &batch_degraded, b] {
                batch_degraded[b] = read_batch(reads, batches[b].first, batches[b].second) ? 1 : 0;
            }));
        }
        for (std::future<void>& batch : done) {
            batch.get();
        }
        for (std::uint8_t batch : batch_degraded) {
            degraded += batch;
        }
        for (BlockRead& item : reads) {
            item.failure = verify(state, item);
        }
        return degraded;
    }

    // Every request and result array stays in place, unmodified, until its token is terminal
    struct Submitted {
        std::shared_ptr<StoreInstance> store;
        std::vector<OvKVCacheStoreReadRequest> requests;
        std::vector<OvKVCacheStoreReadResult> results;
        OvKVCacheAsyncToken token = 0;
    };
    std::vector<Submitted> submitted(batches.size());
    for (std::size_t b = 0; b < batches.size(); ++b) {
        const auto [start, end] = batches[b];
        Submitted& batch = submitted[b];
        batch.store = reads[start].ns->store;
        for (std::size_t i = start; i < end; ++i) {
            batch.requests.push_back(reads[i].request);
        }
        batch.results.resize(end - start);
        // a failed submit leaves the token 0: that batch is read synchronously below
        batch.store->read_blocks_async(batch.requests.data(), batch.results.data(), end - start, batch.token);
    }
    bool cancelled = false;
    for (std::size_t b = 0; b < batches.size(); ++b) {
        const auto [start, end] = batches[b];
        Submitted& batch = submitted[b];
        if (batch.token == 0) {
            if (cancelled) {
                for (std::size_t i = start; i < end; ++i) {
                    reads[i].result.status = OV_KVCACHE_ERR_CANCELLED;
                }
            } else {
                degraded += read_batch(reads, start, end) ? 1 : 0;
            }
        } else {
            OvKVCacheStatus status = OV_KVCACHE_PENDING;
            while (status == OV_KVCACHE_PENDING) {
                status = batch.store->wait(batch.token, OV_KVCACHE_WAIT_INFINITE);
            }
            for (std::size_t i = start; i < end; ++i) {
                reads[i].result = batch.results[i - start];
            }
            if (status < 0 && status != OV_KVCACHE_ERR_CANCELLED) {
                if (end - start > 1) {
                    // the batch failed as a whole: the chunk degrades to one call per block
                    degraded += read_batch(reads, start, end) ? 1 : 0;
                } else if (reads[start].result.status == OV_KVCACHE_OK) {
                    reads[start].result.status = status;
                }
            }
        }
        for (std::size_t i = start; i < end; ++i) {
            BlockRead& item = reads[i];
            item.failure = verify(state, item);
            if (!item.failure.empty() && item.channel == m_kv_channel && !cancelled) {
                // nothing after a failed KV block can be used: stop the batches that have not started
                cancelled = true;
                for (std::size_t later = b + 1; later < batches.size(); ++later) {
                    if (submitted[later].token != 0) {
                        submitted[later].store->cancel(submitted[later].token);
                    }
                }
            }
        }
    }
    return degraded;
}

bool CacheStoreTier::read_batch(std::vector<BlockRead>& reads, std::size_t start, std::size_t end) const {
    const std::size_t count = end - start;
    std::vector<OvKVCacheStoreReadRequest> requests(count);
    std::vector<OvKVCacheStoreReadResult> results(count);
    for (std::size_t i = 0; i < count; ++i) {
        requests[i] = reads[start + i].request;
    }
    const std::shared_ptr<StoreInstance>& store = reads[start].ns->store;
    const OvKVCacheStatus status = store->read_blocks(requests.data(), results.data(), count);
    const bool degraded = status < 0 && count > 1;
    if (degraded) {
        // the batch failed as a whole: the chunk degrades to one call per KV block, with the state ending in it
        for (std::size_t i = 0; i < count;) {
            std::size_t next = i + 1;
            while (next < count && reads[start + next].channel != m_kv_channel) {
                ++next;
            }
            store->read_blocks(&requests[i], &results[i], next - i);
            i = next;
        }
    } else if (status < 0 && results[0].status == OV_KVCACHE_OK) {
        results[0].status = status;
    }
    for (std::size_t i = 0; i < count; ++i) {
        reads[start + i].result = results[i];
    }
    return degraded;
}

std::string CacheStoreTier::verify(RequestState& state, const BlockRead& item) const {
    const OvKVCacheStoreReadResult& result = item.result;
    if (result.status != OV_KVCACHE_OK) {
        return "read returned " + status_to_string(result.status);
    }
    if (result.bytes_read != item.channel->block_bytes) {
        return "short read";
    }
    WitnessRecord record;
    if (result.witness.data != item.witness.data() || !WitnessRecord::decode(item.witness.data(), result.witness.len, record)) {
        return "malformed witness";
    }
    const Digest128 expected = compute_witness_digest(m_witness_key, item.ns->root, item.channel->group,
                                                      (item.block + 1) * m_kv_channel->block_size_tokens,
                                                      state.identity.witness_chain(item.block), record);
    if (expected != record.digest) {
        return "witness mismatch";
    }
    if (record.expires_at != 0 && record.expires_at <= unix_seconds_now()) {
        return "expired";
    }
    if ((record.flags & WitnessRecord::FLAG_PAYLOAD_CRC) &&
        crc32c(item.payload.data(), item.channel->block_bytes) != record.payload_crc) {
        return "payload checksum mismatch";
    }
    return {};
}

void CacheStoreTier::forget_stored_from(RequestState& state, std::size_t token_position) const {
    for (std::size_t block = token_position / m_kv_channel->block_size_tokens; block < state.kv_writes.size(); ++block) {
        if (state.kv_writes[block] == StoreWrite::Never) {
            state.kv_writes[block] = StoreWrite::Always;
        }
    }
}

std::optional<CacheStoreTier::Discovery> CacheStoreTier::discover(RequestState& state, std::size_t device_tokens) {
    SequenceGroup& group = *state.group;
    CacheDiagnostics& diagnostics = group.get_cache_diagnostics();
    const std::size_t block_size = m_kv_channel->block_size_tokens;
    const std::size_t prompt_len = group.get_prompt_len();
    // at least one prompt token is always computed
    if (device_tokens + 1 >= prompt_len) {
        return std::nullopt;
    }
    if (device_tokens % block_size != 0) {
        diagnostics.miss_reason = "the device prefix ends inside a block";
        return std::nullopt;
    }
    const std::size_t max_position = prompt_len - 1;
    const std::size_t first_block = device_tokens / block_size;
    const std::size_t num_complete = prompt_len / block_size;
    if (first_block >= num_complete) {
        return std::nullopt;
    }
    state.identity.extend(group, *group[0], num_complete);

    // One probe per namespace with the prompt's whole key chain, from its first block and in order:
    // a store may rely on that order. A paged channel reaches the end of the run of present blocks, a whole-state
    // channel the largest present checkpoint below it (guide section 14.4); one probe answers both.
    Discovery found;
    diagnostics.store_probe_rounds = 1;
    found.kv_source.assign(num_complete, nullptr);
    std::shared_ptr<const KeyChain> layout;  // where each key sits in a chain; the same in every namespace
    std::vector<std::pair<const OpenNamespace*, std::vector<bool>>> probes;
    for (const OpenNamespace* ns : state.read_namespaces) {
        layout = key_chain(state, *ns, num_complete);
        bool probed = false;
        probes.emplace_back(ns, probe(state, *ns, layout->keys, probed, found.probe_failure));
        // each block from the first namespace that holds it, in the request's order of preference
        for (std::size_t block = 0; block < num_complete; ++block) {
            if (!found.kv_source[block] && probes.back().second[layout->kv_index[block]]) {
                found.kv_source[block] = ns;
            }
        }
    }
    std::size_t reach = first_block;
    while (reach < num_complete && found.kv_source[reach]) {
        ++reach;
    }
    std::size_t position = std::min(max_position, reach * block_size);
    if (m_la_channel) {
        const std::size_t interval = m_la_channel->block_size_tokens;
        std::size_t best = device_tokens;
        for (std::size_t checkpoint = position / interval * interval; checkpoint > device_tokens && best == device_tokens;
             checkpoint -= interval) {
            // A state and the KV block it ends in are read from the same namespace, in one call
            const std::size_t block = checkpoint / block_size - 1;
            for (const auto& [ns, present] : probes) {
                if (present[layout->la_index.at(block)] && present[layout->kv_index[block]]) {
                    best = checkpoint;
                    found.la_source = ns;
                    found.kv_source[block] = ns;
                    break;
                }
            }
        }
        position = best;
    }
    found.position = position;
    return found;
}

void CacheStoreTier::plan_reload(RequestState& state) {
    SequenceGroup& group = *state.group;
    CacheDiagnostics& diagnostics = group.get_cache_diagnostics();
    const std::size_t block_size = m_kv_channel->block_size_tokens;
    const std::size_t device_tokens = group.get_num_processed_tokens();
    const std::optional<Discovery> found = discover(state, device_tokens);
    if (!found) {
        return;
    }
    const std::size_t first_block = device_tokens / block_size;
    const std::size_t num_complete = group.get_prompt_len() / block_size;
    std::size_t position = found->position;
    // A block one of the request's namespaces holds is not written again: a session that extends a shared prefix
    // stores only its own part
    state.kv_writes.assign(num_complete, StoreWrite::Always);
    for (std::size_t block = 0; block < num_complete; ++block) {
        if (found->kv_source[block]) {
            state.kv_writes[block] = StoreWrite::Never;
        }
    }
    if (position <= device_tokens) {
        diagnostics.miss_reason =
            found->probe_failure.empty() ? "not in the store" : "the store is unavailable: " + found->probe_failure;
        return;
    }
    // Fetch into staging, each block from the namespace its probe found it in: the KV blocks in order, then the state
    // at the end, right after the KV block it ends in
    const std::size_t num_blocks = (position + block_size - 1) / block_size;
    std::vector<BlockRead> reads;
    reads.reserve(num_blocks - first_block + 1);
    for (std::size_t block = first_block; block < num_blocks; ++block) {
        BlockRead item;
        item.channel = m_kv_channel;
        item.ns = found->kv_source[block];
        item.block = block;
        item.payload = m_staging.try_acquire(m_kv_channel->block_bytes);
        if (!item.payload) {
            break;
        }
        item.request.key.hash = state.identity.key(item.ns->root, block, m_kv_channel->group);
        reads.push_back(std::move(item));
    }
    if (m_la_channel) {
        BlockRead item;
        item.channel = m_la_channel;
        item.ns = found->la_source;
        item.block = position / block_size - 1;
        item.payload = m_staging.try_acquire(m_la_channel->block_bytes);
        if (reads.size() < num_blocks - first_block || !item.payload) {
            diagnostics.miss_reason = "the staging pool cannot hold the reload";
            return;
        }
        item.request.key.hash = state.identity.key(item.ns->root, item.block, m_la_channel->group);
        reads.push_back(std::move(item));
    } else if (reads.empty()) {
        diagnostics.miss_reason = "the staging pool cannot hold the reload";
        return;
    }
    diagnostics.degraded_chunks += fetch(state, reads);

    // A block that fails verification is a miss for itself and everything after it
    std::size_t verified_blocks = first_block;
    std::string failure;
    for (const BlockRead& item : reads) {
        if (item.result.status == OV_KVCACHE_OK) {
            diagnostics.store_read_blocks += 1;
            diagnostics.store_read_bytes += item.result.bytes_read;
        }
    }
    for (const BlockRead& item : reads) {
        if (item.channel != m_kv_channel) {
            continue;
        }
        if (!item.failure.empty()) {
            // nothing from here on is trusted: it is written again once recomputed, which repairs the store in one pass
            failure = item.failure;
            forget_stored_from(state, item.block * block_size);
            break;
        }
        ++verified_blocks;
    }
    if (m_la_channel) {
        const std::string& la_failure = reads.back().failure;
        if (!la_failure.empty()) {
            // A state is stored with the KV block it ends in, so the pair is written again from that block on
            forget_stored_from(state, position - block_size);
            failure = failure.empty() ? la_failure : failure;
        }
        if (!failure.empty() || verified_blocks < num_blocks) {
            diagnostics.miss_reason = "verification failed: " + failure;
            return;  // the only state fetched is the one at the end, so any failure voids the reload
        }
    } else {
        position = std::min(position, verified_blocks * block_size);
        if (verified_blocks < num_blocks) {
            diagnostics.miss_reason = failure.empty() ? "the staging pool cannot hold the reload"
                                                      : "verification stopped the reload: " + failure;
        }
        if (position <= device_tokens) {
            return;
        }
    }

    auto plan = std::make_unique<ReloadPlan>();
    plan->device_tokens = device_tokens;
    plan->target_tokens = position;
    const std::size_t kv_needed = (position + block_size - 1) / block_size - first_block;
    for (BlockRead& item : reads) {
        if (item.channel == m_kv_channel && plan->kv_payloads.size() < kv_needed) {
            plan->kv_payloads.push_back(std::move(item.payload));
        } else if (item.channel == m_la_channel) {
            plan->la_payload = std::move(item.payload);
        }
    }
    state.reload = std::move(plan);
}

void CacheStoreTier::allocate_reloads(const std::vector<SequenceGroup::Ptr>& requests) {
    for (const SequenceGroup::Ptr& group : requests) {
        if (group->num_total_seqs() != 1) {
            continue;
        }
        const std::uint64_t seq_id = (*group)[0]->get_id();
        RequestState* state = nullptr;
        {
            std::lock_guard<std::mutex> lock(m_requests_mutex);
            const auto it = m_requests.find(seq_id);
            state = it == m_requests.end() ? nullptr : it->second.get();
        }
        if (!state || !state->reload || state->reload->allocated) {
            continue;
        }
        ReloadPlan& plan = *state->reload;
        CacheDiagnostics& diagnostics = group->get_cache_diagnostics();
        const std::size_t additional_tokens = plan.target_tokens - plan.device_tokens;
        if (group->get_num_processed_tokens() != plan.device_tokens ||
            !m_orchestrator->can_allocate_tokens(group, additional_tokens)) {
            diagnostics.miss_reason = "the device cache has no room for the reloaded blocks";
            state->reload.reset();
            continue;
        }
        // The pipeline's own decision, never a count from the provider (guide section 11, step 7)
        m_orchestrator->allocate_tokens((*group)[0], group, additional_tokens, group->get_prompt_len());
        group->update_processed_tokens_num(plan.target_tokens);
        diagnostics.store_hit_tokens = additional_tokens;
        diagnostics.served_by = CacheTier::Store;
        const std::size_t block_size = m_kv_channel->block_size_tokens;
        state->write_cursor = std::max(state->write_cursor, (plan.target_tokens + block_size - 1) / block_size);
        plan.allocated = true;
        m_pending_uploads.push_back(seq_id);
    }
}

void CacheStoreTier::upload_reloads() {
    for (std::uint64_t seq_id : m_pending_uploads) {
        RequestState* state = nullptr;
        {
            std::lock_guard<std::mutex> lock(m_requests_mutex);
            const auto it = m_requests.find(seq_id);
            state = it == m_requests.end() ? nullptr : it->second.get();
        }
        if (!state || !state->reload || !m_orchestrator->has_block_table(seq_id)) {
            continue;
        }
        ReloadPlan& plan = *state->reload;
        const std::size_t block_size = m_kv_channel->block_size_tokens;
        const BlocksPerLayer& kv_table = m_orchestrator->get_kv_block_tables(seq_id)[0];
        const std::size_t first_block = plan.device_tokens / block_size;
        std::vector<BlockTransfer> kv_transfers;
        for (std::size_t i = 0; i < plan.kv_payloads.size(); ++i) {
            kv_transfers.push_back({static_cast<std::size_t>(kv_table.at(first_block + i)->get_index()),
                                    plan.kv_payloads[i].data()});
        }
        m_codec.upload(*m_kv_channel, cache_tensors(*m_kv_channel), kv_transfers);
        if (m_la_channel && plan.la_payload) {
            // the model continues from the state row of its last processed token
            const std::size_t la_position = (plan.target_tokens - 1) / m_la_channel->block_size_tokens;
            const std::size_t la_start = m_orchestrator->get_linear_attention_block_table_logical_start(seq_id);
            const BlocksPerLayer& la_table = m_orchestrator->get_linear_attention_block_table(seq_id);
            OPENVINO_ASSERT(la_position >= la_start && la_position - la_start < la_table.size(),
                            "Cache store: the linear attention row of position ", plan.target_tokens,
                            " was not allocated for sequence ", seq_id);
            const std::vector<BlockTransfer> la_transfer = {
                {static_cast<std::size_t>(la_table[la_position - la_start]->get_index()), plan.la_payload.data()}};
            m_codec.upload(*m_la_channel, cache_tensors(*m_la_channel), la_transfer);
        }
        state->reload.reset();
    }
    m_pending_uploads.clear();
}

void CacheStoreTier::after_step(const std::vector<SequenceGroup::Ptr>& requests,
                                const std::vector<std::uint64_t>& scheduled_group_ids) {
    m_step_copied_bytes = 0;
    for (std::uint64_t group_id : scheduled_group_ids) {
        const SequenceGroup::Ptr& group = requests.at(group_id);
        if (group->num_total_seqs() != 1) {
            continue;
        }
        RequestState* state = nullptr;
        {
            std::lock_guard<std::mutex> lock(m_requests_mutex);
            const auto it = m_requests.find((*group)[0]->get_id());
            state = it == m_requests.end() ? nullptr : it->second.get();
        }
        if (!state) {
            continue;
        }
        if (state->write_mode == WriteMode::Immediate || state->write_mode == WriteMode::Push) {
            write_complete_blocks(*state, true);
        } else if (state->write_mode == WriteMode::OnAccess) {
            write_reused_blocks(*state, true);
        }
    }
}

void CacheStoreTier::on_free_sequence(std::uint64_t seq_id) {
    std::unique_ptr<RequestState> state;
    {
        std::lock_guard<std::mutex> lock(m_requests_mutex);
        const auto it = m_requests.find(seq_id);
        if (it == m_requests.end()) {
            return;
        }
        state = std::move(it->second);
        m_requests.erase(it);
    }
    // the blocks are still the sequence's, so what the step budget held back is written now
    switch (state->write_mode) {
    case WriteMode::Immediate:
    case WriteMode::Push:
        write_complete_blocks(*state, false);
        break;
    case WriteMode::OnAccess:
        write_reused_blocks(*state, false);
        break;
    case WriteMode::OnEvict:
        remember_for_write_back(*state);
        break;
    case WriteMode::None:
        break;
    }
    finish_expected(*state);
}

void CacheStoreTier::on_overwrite(CacheType type, uint64_t hash, size_t block_index) {
    std::lock_guard<std::mutex> lock(m_write_back_mutex);
    const auto it = m_write_back.find({type, hash});
    if (it == m_write_back.end()) {
        return;
    }
    // the block gets other content in this step: its bytes are copied out before any device write of the step
    m_evicted.emplace_back(it->second, block_index);
    m_write_back.erase(it);
}

void CacheStoreTier::before_block_writes() {
    std::vector<std::pair<WriteBack, std::size_t>> evicted;
    std::vector<std::pair<WriteBackId, WriteBack>> partners;
    {
        std::lock_guard<std::mutex> lock(m_write_back_mutex);
        evicted.swap(m_evicted);
        // A KV block and the state ending in it are stored together, so the eviction of one writes both back
        for (const auto& [block, block_index] : evicted) {
            if (!block.partner) {
                continue;
            }
            const auto partner = m_write_back.find(*block.partner);
            if (partner != m_write_back.end()) {
                partners.emplace_back(partner->first, std::move(partner->second));
                m_write_back.erase(partner);
            }
        }
    }
    if (evicted.empty()) {
        return;
    }
    for (auto& [id, block] : partners) {
        const std::optional<std::size_t> block_index = m_orchestrator->find_cached_block_index(id.first, id.second);
        if (block_index) {
            evicted.emplace_back(std::move(block), *block_index);
        }
    }
    const std::vector<bool> queued = write_back(evicted);
    const std::size_t lost = static_cast<std::size_t>(std::count(queued.begin(), queued.end(), false));
    if (lost > 0 && !m_reported_write_back_loss.exchange(true)) {
        GENAI_WARN("Cache store: the staging pool had no space for " + std::to_string(lost) +
                   " evicted blocks, so they were not written back. Further losses are not reported.");
    }
}

std::vector<ov::Tensor> CacheStoreTier::cache_tensors(const StoreChannel& channel) const {
    if (channel.cache_type == CacheType::KV_CACHE) {
        const auto& kv_manager =
            static_cast<const KVCacheManager&>(m_orchestrator->get_cache_manager(CacheType::KV_CACHE));
        std::vector<ov::Tensor> tensors;
        tensors.reserve(2 * kv_manager.get_num_layers());
        for (std::size_t layer = 0; layer < kv_manager.get_num_layers(); ++layer) {
            tensors.push_back(kv_manager.get_key_cache(layer));
            tensors.push_back(kv_manager.get_value_cache(layer));
        }
        return tensors;
    }
    return static_cast<const LinearAttentionCacheManager&>(
               m_orchestrator->get_cache_manager(CacheType::LINEAR_ATTENTION_CACHE))
        .get_state_table_tensors();
}

void CacheStoreTier::write_complete_blocks(RequestState& state, bool within_budget) {
    const std::uint64_t seq_id = (*state.group)[0]->get_id();
    if (!m_orchestrator->has_block_table(seq_id)) {
        return;
    }
    const std::size_t num_complete = std::min(state.group->get_num_processed_tokens() / m_kv_channel->block_size_tokens,
                                              m_orchestrator->get_kv_block_tables(seq_id)[0].size());
    // speculative decoding can take processed tokens back
    state.write_cursor = std::min(state.write_cursor, num_complete);
    state.write_cursor = write_blocks(state, state.write_cursor, num_complete, within_budget);
}

void CacheStoreTier::write_reused_blocks(RequestState& state, bool within_budget) {
    const std::uint64_t seq_id = (*state.group)[0]->get_id();
    if (state.access_cursor >= state.reused_blocks || !m_orchestrator->has_block_table(seq_id)) {
        return;
    }
    const std::size_t end_block = std::min(state.reused_blocks, m_orchestrator->get_kv_block_tables(seq_id)[0].size());
    state.access_cursor = write_blocks(state, state.access_cursor, end_block, within_budget);
}

std::size_t CacheStoreTier::write_blocks(RequestState& state,
                                         std::size_t first_block,
                                         std::size_t end_block,
                                         bool within_budget) {
    if (first_block >= end_block || !state.write_namespace) {
        return first_block;
    }
    const OpenNamespace& ns = *state.write_namespace;
    const SequenceGroup& group = *state.group;
    const Sequence::CPtr sequence = group[0];
    const std::uint64_t seq_id = sequence->get_id();
    const std::size_t block_size = m_kv_channel->block_size_tokens;
    const std::vector<BlocksPerLayer>& kv_block_tables = m_orchestrator->get_kv_block_tables(seq_id);
    state.identity.extend(group, *sequence, end_block);

    const PromptCachePolicy& policy = group.get_sampling_parameters().prompt_cache;
    // One batch for the request: its blocks in order, each state right after the KV block it ends in
    auto batch = std::make_shared<WriteBatch>();
    batch->store = ns.store;
    batch->root = ns.root;
    batch->expires_at = policy.ttl.is_unlimited() ? 0 : unix_seconds_now() + policy.ttl.get_seconds();
    batch->push = state.expected.push;
    std::vector<BlockTransfer> kv_transfers, la_transfers;
    std::vector<std::size_t> staged_blocks;

    std::size_t next = first_block;
    for (std::size_t block = first_block; block < end_block; ++block) {
        const std::size_t end_position = (block + 1) * block_size;
        // what the store already has stays as it is: the first write of a key wins
        const StoreWrite kv_write = block < state.kv_writes.size() ? state.kv_writes[block] : StoreWrite::IfAbsent;
        if (kv_write == StoreWrite::Never) {
            next = block + 1;
            continue;
        }
        // The linear attention state after this block, when a checkpoint ends here. It is written with its KV block
        // or not at all: a store may keep the two as one record.
        std::optional<std::size_t> la_row;
        if (m_la_channel && end_position % m_la_channel->block_size_tokens == 0) {
            const std::size_t la_position = end_position / m_la_channel->block_size_tokens - 1;
            const std::size_t la_start = m_orchestrator->get_linear_attention_block_table_logical_start(seq_id);
            const BlocksPerLayer& la_table = m_orchestrator->get_linear_attention_block_table(seq_id);
            if (la_position >= la_start && la_position - la_start < la_table.size()) {
                la_row = static_cast<std::size_t>(la_table[la_position - la_start]->get_index());
            }
        }
        const std::size_t bytes = m_kv_channel->block_bytes + (la_row ? m_la_channel->block_bytes : 0);
        if (within_budget && m_step_copied_bytes > 0 &&
            m_step_copied_bytes + bytes > m_config.step_transfer_budget_bytes) {
            break;  // the rest waits for the next step; the sequence keeps its blocks meanwhile
        }
        StagingLease kv_payload = m_staging.try_acquire(m_kv_channel->block_bytes);
        StagingLease la_payload = la_row ? m_staging.try_acquire(m_la_channel->block_bytes) : StagingLease();
        if (!kv_payload || (la_row && !la_payload)) {
            break;  // staging is full of writes in flight
        }

        const bool if_absent = kv_write == StoreWrite::IfAbsent;
        kv_transfers.push_back({static_cast<std::size_t>(kv_block_tables[0][block]->get_index()), kv_payload.data()});
        batch->blocks.push_back({std::move(kv_payload), m_kv_channel, OvKVCacheBlockKey{}, 0, end_position,
                                 state.identity.witness_chain(block), if_absent});
        staged_blocks.push_back(block);
        if (la_row) {
            la_transfers.push_back({*la_row, la_payload.data()});
            batch->blocks.push_back({std::move(la_payload), m_la_channel, OvKVCacheBlockKey{}, 0, end_position,
                                     state.identity.witness_chain(block), if_absent});
            staged_blocks.push_back(block);
        }
        m_step_copied_bytes += bytes;
        next = block + 1;
    }
    if (batch->blocks.empty()) {
        return next;
    }

    // The sequence's keys from its first block: the store is probed with them before the write (run_write_batch)
    batch->chain = key_chain(state, ns, staged_blocks.back() + 1);
    for (std::size_t i = 0; i < batch->blocks.size(); ++i) {
        BlockWrite& item = batch->blocks[i];
        item.chain_index = item.channel == m_kv_channel ? batch->chain->kv_index[staged_blocks[i]]
                                                        : batch->chain->la_index.at(staged_blocks[i]);
        item.key = batch->chain->keys[item.chain_index];
    }
    CacheDiagnostics& diagnostics = state.group->get_cache_diagnostics();
    m_codec.download(*m_kv_channel, cache_tensors(*m_kv_channel), kv_transfers);
    diagnostics.pushed_blocks += kv_transfers.size();
    diagnostics.pushed_bytes += kv_transfers.size() * m_kv_channel->block_bytes;
    if (!la_transfers.empty()) {
        m_codec.download(*m_la_channel, cache_tensors(*m_la_channel), la_transfers);
        diagnostics.pushed_blocks += la_transfers.size();
        diagnostics.pushed_bytes += la_transfers.size() * m_la_channel->block_bytes;
    }
    if (batch->push) {
        std::lock_guard<std::mutex> lock(batch->push->mutex);
        batch->push->pending_batches += 1;
    }
    m_io_pool.submit([this, batch] {
        run_write_batch(batch);
    });
    return next;
}

void CacheStoreTier::remember_for_write_back(RequestState& state) {
    const SequenceGroup& group = *state.group;
    const Sequence::CPtr sequence = group[0];
    const std::uint64_t seq_id = sequence->get_id();
    if (!m_orchestrator->has_block_table(seq_id)) {
        return;
    }
    const std::size_t block_size = m_kv_channel->block_size_tokens;
    const BlocksPerLayer& kv_table = m_orchestrator->get_kv_block_tables(seq_id)[0];
    const std::size_t num_complete = std::min(group.get_num_processed_tokens() / block_size, kv_table.size());
    if (num_complete == 0 || !state.write_namespace) {
        return;
    }
    const OpenNamespace* ns = state.write_namespace;
    state.identity.extend(group, *sequence, num_complete);
    const std::shared_ptr<const KeyChain> chain = key_chain(state, *ns, num_complete);
    const PromptCachePolicy& policy = group.get_sampling_parameters().prompt_cache;
    const std::uint64_t ttl_seconds = policy.ttl.is_unlimited() ? 0 : policy.ttl.get_seconds();

    // collected first: the orchestrator is never called while the write-back lock is held
    std::vector<std::pair<WriteBackId, WriteBack>> blocks;
    for (std::size_t block = 0; block < num_complete; ++block) {
        const std::size_t end_position = (block + 1) * block_size;
        const StoreWrite kv_write = block < state.kv_writes.size() ? state.kv_writes[block] : StoreWrite::IfAbsent;
        if (kv_write == StoreWrite::Never) {
            continue;  // the state ending here is stored with this block or not at all
        }
        WriteBack kv{m_kv_channel, ns, chain, chain->kv_index[block], block, end_position,
                     state.identity.witness_chain(block), ttl_seconds, kv_write == StoreWrite::IfAbsent, std::nullopt};
        const WriteBackId kv_id{CacheType::KV_CACHE, kv_table[block]->get_hash()};
        if (m_la_channel && end_position % m_la_channel->block_size_tokens == 0) {
            const std::size_t la_position = end_position / m_la_channel->block_size_tokens - 1;
            const std::size_t la_start = m_orchestrator->get_linear_attention_block_table_logical_start(seq_id);
            const BlocksPerLayer& la_table = m_orchestrator->get_linear_attention_block_table(seq_id);
            if (la_position >= la_start && la_position - la_start < la_table.size()) {
                const WriteBackId la_id{CacheType::LINEAR_ATTENTION_CACHE, la_table[la_position - la_start]->get_hash()};
                WriteBack la = kv;
                la.channel = m_la_channel;
                la.chain_index = chain->la_index.at(block);
                la.partner = kv_id;
                kv.partner = la_id;
                blocks.emplace_back(la_id, std::move(la));
            }
        }
        blocks.emplace_back(kv_id, std::move(kv));
    }
    std::lock_guard<std::mutex> lock(m_write_back_mutex);
    for (auto& [id, block] : blocks) {
        m_write_back[id] = std::move(block);
    }
}

std::vector<bool> CacheStoreTier::write_back(const std::vector<std::pair<WriteBack, std::size_t>>& blocks) {
    // In the order of their sequences: a KV block, then the state that ends in it
    std::vector<std::size_t> order(blocks.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    const auto sort_key = [&](std::size_t i) {
        const WriteBack& block = blocks[i].first;
        return std::make_tuple(block.ns, block.chain.get(), block.ttl_seconds, block.block, block.channel != m_kv_channel);
    };
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return sort_key(a) < sort_key(b);
    });

    // one batch per sequence and expiry
    struct Pending {
        std::shared_ptr<WriteBatch> batch;
        std::vector<BlockTransfer> kv_transfers, la_transfers;
    };
    std::map<std::tuple<const OpenNamespace*, const KeyChain*, std::uint64_t>, Pending> batches;
    std::vector<bool> queued(blocks.size(), false);
    const std::uint64_t now = unix_seconds_now();
    for (std::size_t o = 0; o < order.size(); ++o) {
        const std::size_t i = order[o];
        const WriteBack& block = blocks[i].first;
        if (block.channel != m_kv_channel) {
            queued[i] = true;  // a state whose KV block is not written with it cannot be stored: nothing to retry
            continue;
        }
        // the state ending in this block, when it is written back too
        std::optional<std::size_t> la;
        if (o + 1 < order.size()) {
            const WriteBack& following = blocks[order[o + 1]].first;
            if (following.channel != m_kv_channel && following.chain == block.chain && following.block == block.block &&
                following.ns == block.ns) {
                la = order[o + 1];
            }
        }
        StagingLease kv_payload = m_staging.try_acquire(m_kv_channel->block_bytes);
        StagingLease la_payload = la ? m_staging.try_acquire(m_la_channel->block_bytes) : StagingLease();
        if (!kv_payload || (la && !la_payload)) {
            if (la) {
                ++o;  // the pair is tried again together
            }
            continue;
        }
        const std::uint64_t expires_at = block.ttl_seconds == 0 ? 0 : now + block.ttl_seconds;
        Pending& pending = batches[{block.ns, block.chain.get(), expires_at}];
        if (!pending.batch) {
            pending.batch = std::make_shared<WriteBatch>();
            pending.batch->store = block.ns->store;
            pending.batch->root = block.ns->root;
            pending.batch->expires_at = expires_at;
            pending.batch->chain = block.chain;
        }
        pending.kv_transfers.push_back({blocks[i].second, kv_payload.data()});
        pending.batch->blocks.push_back({std::move(kv_payload), m_kv_channel, block.chain->keys[block.chain_index],
                                         block.chain_index, block.end_position, block.witness_chain, block.if_absent});
        queued[i] = true;
        if (la) {
            const WriteBack& state_block = blocks[*la].first;
            pending.la_transfers.push_back({blocks[*la].second, la_payload.data()});
            pending.batch->blocks.push_back({std::move(la_payload), m_la_channel,
                                             state_block.chain->keys[state_block.chain_index], state_block.chain_index,
                                             state_block.end_position, state_block.witness_chain, state_block.if_absent});
            queued[*la] = true;
            ++o;
        }
    }
    for (auto& [id, pending] : batches) {
        m_codec.download(*m_kv_channel, cache_tensors(*m_kv_channel), pending.kv_transfers);
        if (!pending.la_transfers.empty()) {
            m_codec.download(*m_la_channel, cache_tensors(*m_la_channel), pending.la_transfers);
        }
        m_io_pool.submit([this, batch = pending.batch] {
            run_write_batch(batch);
        });
    }
    return queued;
}

void CacheStoreTier::write_back_cached_blocks() {
    std::map<std::pair<CacheType, std::uint64_t>, WriteBack> remembered;
    {
        std::lock_guard<std::mutex> lock(m_write_back_mutex);
        remembered.swap(m_write_back);
    }
    std::vector<std::pair<WriteBack, std::size_t>> pending;
    for (const auto& [id, block] : remembered) {
        // a hash that no longer names cached content (it was dropped without an eviction) has nothing to write
        const std::optional<std::size_t> block_index = m_orchestrator->find_cached_block_index(id.first, id.second);
        if (block_index) {
            pending.emplace_back(block, *block_index);
        }
    }
    // In rounds: each queues what fits in staging, the next starts once those writes are done
    bool staging_idle = false;
    while (!pending.empty()) {
        const std::vector<bool> queued = write_back(pending);
        std::vector<std::pair<WriteBack, std::size_t>> rest;
        for (std::size_t i = 0; i < pending.size(); ++i) {
            if (!queued[i]) {
                rest.push_back(pending[i]);
            }
        }
        if (rest.size() == pending.size() && staging_idle) {
            GENAI_WARN("Cache store: " + std::to_string(rest.size()) +
                       " blocks are larger than the staging pool and were not written back");
            break;
        }
        pending.swap(rest);
        if (!pending.empty()) {
            wait_for_writes();
            staging_idle = true;
        }
    }
}

void CacheStoreTier::wait_for_writes() {
    m_io_pool.drain();
    std::unique_lock<std::mutex> lock(m_inflight_mutex);
    m_inflight_done.wait(lock, [this] {
        return m_inflight_count == 0;
    });
}

void CacheStoreTier::run_write_batch(const std::shared_ptr<WriteBatch>& batch) {
    // The store is probed with the sequence's keys from its first block, in order, before every write: a store that
    // keeps sequences learns where the blocks belong, and a block the reload never probed (e.g. of
    // generated tokens) is written only if the store does not hold its key.
    std::vector<bool> skipped(batch->blocks.size(), false);
    if (batch->store->can_probe()) {
        const std::vector<OvKVCacheBlockKey>& keys = batch->chain->keys;
        std::vector<std::uint8_t> present(keys.size(), 0);
        if (batch->store->has_blocks(keys.data(), present.data(), keys.size()) == OV_KVCACHE_OK) {
            for (std::size_t i = 0; i < batch->blocks.size(); ++i) {
                skipped[i] = batch->blocks[i].if_absent && present[batch->blocks[i].chain_index] == 1;
            }
        }
    }
    // a state is written with the KV block it ends in, or not at all
    for (std::size_t i = 1; i < batch->blocks.size(); ++i) {
        if (batch->blocks[i].channel != m_kv_channel && batch->blocks[i - 1].channel == m_kv_channel &&
            batch->blocks[i - 1].end_position == batch->blocks[i].end_position) {
            skipped[i] = skipped[i - 1];
        }
    }

    auto write = std::make_unique<InflightWrite>();
    write->witnesses.reserve(batch->blocks.size());  // requests point into it
    write->requests.reserve(batch->blocks.size());
    for (std::size_t i = 0; i < batch->blocks.size(); ++i) {
        if (skipped[i]) {
            continue;
        }
        const BlockWrite& block = batch->blocks[i];
        const StoreChannel& channel = *block.channel;
        WitnessRecord record;
        record.expires_at = batch->expires_at;
        if (m_config.verify_payload_crc) {
            record.flags |= WitnessRecord::FLAG_PAYLOAD_CRC;
            record.payload_crc = crc32c(block.payload.data(), channel.block_bytes);
        }
        record.digest = compute_witness_digest(m_witness_key, batch->root, channel.group, block.end_position,
                                               block.witness_chain, record);
        write->witnesses.push_back(record.encode());
        OvKVCacheStoreWriteRequest request{};
        request.key = block.key;
        request.source = block.payload.data();
        request.bytes = channel.block_bytes;
        request.witness = {write->witnesses.back().data(), write->witnesses.back().size()};
        write->requests.push_back(request);
        write->channels.push_back(&channel);
    }
    const std::size_t count = write->requests.size();
    if (count == 0) {
        settle_push(batch->push, {});
        return;
    }
    write->results.resize(count);
    write->batch = batch;

    if (batch->store->is_async()) {
        batch->store->write_blocks_async(write->requests.data(), write->results.data(), count, write->token);
        if (write->token != 0) {
            std::lock_guard<std::mutex> lock(m_inflight_mutex);
            if (!m_completion_thread.joinable()) {
                m_completion_thread = std::thread(&CacheStoreTier::complete_writes, this);
            }
            ++m_inflight_count;
            m_inflight_writes.push_back(std::move(write));
            m_inflight_added.notify_one();
            return;
        }
        // the submit failed: the batch is written synchronously instead
    }
    const OvKVCacheStatus status = batch->store->write_blocks(write->requests.data(), write->results.data(), count);
    finish_write(*write, status);
}

void CacheStoreTier::finish_write(InflightWrite& write, OvKVCacheStatus status) {
    const WriteBatch& batch = *write.batch;
    const std::vector<OvKVCacheStoreWriteRequest>& requests = write.requests;
    std::vector<OvKVCacheStoreWriteResult>& results = write.results;
    const std::size_t count = requests.size();
    if (status < 0 && count > 1) {
        // the batch failed as a whole: the chunk degrades to one call per KV block, with the state ending in it
        for (std::size_t i = 0; i < count;) {
            std::size_t next = i + 1;
            while (next < count && write.channels[next] != m_kv_channel) {
                ++next;
            }
            if (batch.store->write_blocks(&requests[i], &results[i], next - i) < 0) {
                for (std::size_t j = i; j < next; ++j) {
                    results[j].status = results[j].status == OV_KVCACHE_OK ? OV_KVCACHE_ERR_INTERNAL : results[j].status;
                }
            }
            i = next;
        }
    } else if (status < 0) {
        results[0].status = results[0].status == OV_KVCACHE_OK ? status : results[0].status;
    }
    std::size_t failed = 0;
    OvKVCacheStatus first_failure = OV_KVCACHE_OK;
    for (const OvKVCacheStoreWriteResult& result : results) {
        if (result.status != OV_KVCACHE_OK) {
            first_failure = failed++ == 0 ? result.status : first_failure;
        }
    }
    if (failed > 0) {
        m_failed_writes += failed;
        if (!m_reported_write_failure.exchange(true)) {
            GENAI_WARN("Cache store: writing " + std::to_string(failed) + " of " + std::to_string(count) +
                       " blocks failed (" + status_to_string(first_failure) +
                       "); they stay recomputable. Further failures are only counted.");
        }
    }
    if (m_config.durability == CacheStoreConfig::Durability::Strict) {
        batch.store->flush();
    }
    std::map<std::string, std::pair<std::size_t, std::uint64_t>> written;  // blocks and bytes by group
    for (std::size_t i = 0; i < count; ++i) {
        if (results[i].status == OV_KVCACHE_OK) {
            auto& [blocks, bytes] = written[write.channels[i]->group];
            blocks += 1;
            bytes += requests[i].bytes;
        }
    }
    settle_push(batch.push, written);
}

void CacheStoreTier::complete_writes() {
    while (true) {
        std::unique_ptr<InflightWrite> write;
        {
            std::unique_lock<std::mutex> lock(m_inflight_mutex);
            m_inflight_added.wait(lock, [this] {
                return m_stop_completion || !m_inflight_writes.empty();
            });
            if (m_inflight_writes.empty()) {
                return;  // stopping, and every submitted batch is terminal
            }
            write = std::move(m_inflight_writes.front());
            m_inflight_writes.pop_front();
        }
        // no host lock is held while waiting (guide section 9)
        OvKVCacheStatus status = OV_KVCACHE_PENDING;
        while (status == OV_KVCACHE_PENDING) {
            status = write->batch->store->wait(write->token, OV_KVCACHE_WAIT_INFINITE);
        }
        finish_write(*write, status);
        // returns the batch's staging leases, now that the plugin no longer touches them
        write.reset();
        {
            std::lock_guard<std::mutex> lock(m_inflight_mutex);
            --m_inflight_count;
        }
        m_inflight_done.notify_all();
    }
}

}  // namespace ov::genai::cache_store
