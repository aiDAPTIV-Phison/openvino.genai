// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/cache_store/cache_store_tier.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <future>
#include <optional>

#include "continuous_batching/cache/cache_orchestrator.hpp"
#include "continuous_batching/cache/kv_cache_manager.hpp"
#include "continuous_batching/cache/linear_attention_cache_manager.hpp"
#include "logger.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov::genai::cache_store {

namespace {

// Rounds of the multi-channel prefix reconciliation (guide section 14.4)
constexpr std::size_t MAX_RECONCILE_ROUNDS = 3;

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
      m_provider(std::move(provider)),
      m_witness_key(load_witness_key()),
      m_staging(config.staging_bytes),
      m_io_pool(config.io_threads) {
    for (const StoreChannel& channel : m_channels) {
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
    m_io_pool.drain();
    if (m_shared.store) {
        const OvKVCacheStatus status = m_shared.store->flush();
        if (status != OV_KVCACHE_OK) {
            GENAI_WARN("Cache store flush failed: " + status_to_string(status));
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

void CacheStoreTier::on_add_request(const SequenceGroup::Ptr& group) {
    CacheDiagnostics& diagnostics = group->get_cache_diagnostics();
    const GenerationConfig& sampling_params = group->get_sampling_parameters();
    if (group->num_total_seqs() != 1 || sampling_params.is_beam_search() || sampling_params.num_return_sequences != 1) {
        diagnostics.miss_reason = "unsupported request: the cache store serves single-sequence requests";
        return;
    }

    const std::size_t block_size = m_kv_channel->block_size_tokens;
    auto state = std::make_unique<RequestState>(group, block_size, m_witness_key);
    // A block the device cache supplied is never written again, even when its last token is recomputed: that rewrites
    // the block's quantization data, so its bytes would differ from the ones already stored under the same key
    state->write_cursor = (group->get_num_processed_tokens() + block_size - 1) / block_size;
    const PromptCachePolicy& policy = sampling_params.prompt_cache;
    state->writes_immediately = m_config.write.device_to_store.trigger == WriteTrigger::Immediate &&
                                !policy.is_explicit_only() && !policy.opt_out_write_through;
    plan_reload(*state);

    const std::uint64_t seq_id = (*group)[0]->get_id();
    std::lock_guard<std::mutex> lock(m_requests_mutex);
    m_requests[seq_id] = std::move(state);
}

std::vector<bool> CacheStoreTier::probe(RequestState& state,
                                        const StoreChannel& channel,
                                        const std::vector<std::size_t>& blocks) {
    std::vector<bool> present(blocks.size(), true);
    if (blocks.empty() || !m_shared.store->can_probe()) {
        return present;  // without a probe every candidate is a candidate for the fetch, which verifies it
    }
    std::vector<OvKVCacheBlockKey> keys;
    keys.reserve(blocks.size());
    for (std::size_t block : blocks) {
        keys.push_back(make_key(channel, state.identity.key(m_shared.root, block, channel.group)));
    }
    std::vector<std::uint8_t> out_present(blocks.size(), 0);
    OvKVCacheStatus status = OV_KVCACHE_OK;
    m_io_pool
        .submit([&] {
            status = m_shared.store->has_blocks(keys.data(), out_present.data(), keys.size());
        })
        .get();
    state.group->get_cache_diagnostics().store_probed_keys += keys.size();
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        present[i] = status == OV_KVCACHE_OK && out_present[i] == 1;
    }
    return present;
}

std::size_t CacheStoreTier::read(std::vector<BlockRead>& reads) {
    for (BlockRead& item : reads) {
        item.request.key = make_key(*item.channel, item.request.key.hash);
        item.request.target.kind = OV_KVCACHE_TARGET_HOST;
        item.request.target.base = item.payload.data();
        item.request.target.bytes = item.channel->block_bytes;
        item.request.witness_out = {item.witness.data(), item.witness.size()};
    }
    // batches of consecutive items of one channel, spread over the I/O threads
    const std::size_t batch_size = std::max<std::size_t>(1, (reads.size() + m_config.io_threads - 1) / m_config.io_threads);
    std::atomic<std::size_t> degraded{0};
    std::vector<std::future<void>> done;
    for (std::size_t start = 0; start < reads.size();) {
        std::size_t end = start + 1;
        while (end < reads.size() && end - start < batch_size && reads[end].channel == reads[start].channel) {
            ++end;
        }
        done.push_back(m_io_pool.submit([this, &reads, &degraded, start, end] {
            const std::size_t count = end - start;
            std::vector<OvKVCacheStoreReadRequest> requests(count);
            std::vector<OvKVCacheStoreReadResult> results(count);
            for (std::size_t i = 0; i < count; ++i) {
                requests[i] = reads[start + i].request;
            }
            const OvKVCacheStatus status = m_shared.store->read_blocks(requests.data(), results.data(), count);
            if (status < 0 && count > 1) {
                // the batch failed as a whole: the chunk degrades to one call per block
                ++degraded;
                for (std::size_t i = 0; i < count; ++i) {
                    m_shared.store->read_blocks(&requests[i], &results[i], 1);
                }
            } else if (status < 0 && results[0].status == OV_KVCACHE_OK) {
                results[0].status = status;
            }
            for (std::size_t i = 0; i < count; ++i) {
                reads[start + i].result = results[i];
            }
        }));
        start = end;
    }
    for (std::future<void>& batch : done) {
        batch.get();
    }
    return degraded.load();
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
    const Digest128 expected = compute_witness_digest(m_witness_key, m_shared.root, item.channel->group,
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
    for (std::size_t block = token_position / m_kv_channel->block_size_tokens; block < state.stored_blocks.size(); ++block) {
        state.stored_blocks[block] = false;
    }
    state.stored_checkpoints.erase(state.stored_checkpoints.upper_bound(token_position), state.stored_checkpoints.end());
}

void CacheStoreTier::plan_reload(RequestState& state) {
    SequenceGroup& group = *state.group;
    CacheDiagnostics& diagnostics = group.get_cache_diagnostics();
    const std::size_t block_size = m_kv_channel->block_size_tokens;
    const std::size_t prompt_len = group.get_prompt_len();
    const std::size_t device_tokens = group.get_num_processed_tokens();
    // at least one prompt token is always computed
    if (device_tokens + 1 >= prompt_len) {
        return;
    }
    if (device_tokens % block_size != 0) {
        diagnostics.miss_reason = "the device prefix ends inside a block";
        return;
    }
    const std::size_t max_position = prompt_len - 1;
    const std::size_t first_block = device_tokens / block_size;
    const std::size_t num_complete = prompt_len / block_size;
    if (first_block >= num_complete) {
        return;
    }
    state.identity.extend(group, *group[0], num_complete);

    // Bounded fixpoint over the channels (guide section 14.4): a paged channel reaches the end of the run of present
    // blocks, a whole-state channel the largest present checkpoint; the common position is their minimum
    std::vector<int> kv_present(num_complete, -1);
    std::map<std::size_t, bool> la_present;
    std::size_t position = max_position;
    for (std::size_t round = 1; round <= MAX_RECONCILE_ROUNDS; ++round) {
        diagnostics.store_probe_rounds = round;
        const std::size_t needed_blocks = std::min((position + block_size - 1) / block_size, num_complete);
        std::vector<std::size_t> unknown;
        for (std::size_t block = first_block; block < needed_blocks; ++block) {
            if (kv_present[block] < 0) {
                unknown.push_back(block);
            }
        }
        const std::vector<bool> kv_found = probe(state, *m_kv_channel, unknown);
        for (std::size_t i = 0; i < unknown.size(); ++i) {
            kv_present[unknown[i]] = kv_found[i] ? 1 : 0;
        }
        std::size_t reach = first_block;
        while (reach < needed_blocks && kv_present[reach] == 1) {
            ++reach;
        }
        std::size_t candidate = std::min(position, reach * block_size);

        if (m_la_channel) {
            const std::size_t interval = m_la_channel->block_size_tokens;
            std::vector<std::size_t> unknown_checkpoints;
            for (std::size_t checkpoint = candidate / interval * interval; checkpoint > device_tokens; checkpoint -= interval) {
                if (!la_present.count(checkpoint)) {
                    unknown_checkpoints.push_back(checkpoint);
                }
            }
            std::vector<std::size_t> checkpoint_blocks;
            for (std::size_t checkpoint : unknown_checkpoints) {
                checkpoint_blocks.push_back(checkpoint / block_size - 1);
            }
            const std::vector<bool> la_found = probe(state, *m_la_channel, checkpoint_blocks);
            for (std::size_t i = 0; i < unknown_checkpoints.size(); ++i) {
                la_present[unknown_checkpoints[i]] = la_found[i];
            }
            std::size_t best = device_tokens;
            for (std::size_t checkpoint = candidate / interval * interval; checkpoint > device_tokens; checkpoint -= interval) {
                if (la_present[checkpoint]) {
                    best = checkpoint;
                    break;
                }
            }
            candidate = std::min(candidate, best);
        }
        // every round's candidate is servable by all channels, so stopping at the bound keeps a valid position
        const bool converged = candidate == position;
        position = candidate;
        if (converged) {
            break;
        }
    }
    state.stored_blocks.assign(num_complete, false);
    for (std::size_t block = first_block; block < num_complete; ++block) {
        state.stored_blocks[block] = kv_present[block] == 1;
    }
    for (const auto& [checkpoint, present] : la_present) {
        if (present) {
            state.stored_checkpoints.insert(checkpoint);
        }
    }
    if (position <= device_tokens) {
        diagnostics.miss_reason = "not in the store";
        return;
    }

    // Fetch into staging
    const std::size_t num_blocks = (position + block_size - 1) / block_size;
    std::vector<BlockRead> reads;
    reads.reserve(num_blocks - first_block + 1);
    for (std::size_t block = first_block; block < num_blocks; ++block) {
        BlockRead item;
        item.channel = m_kv_channel;
        item.block = block;
        item.payload = m_staging.try_acquire(m_kv_channel->block_bytes);
        if (!item.payload) {
            break;
        }
        item.request.key.hash = state.identity.key(m_shared.root, block, m_kv_channel->group);
        reads.push_back(std::move(item));
    }
    if (m_la_channel) {
        BlockRead item;
        item.channel = m_la_channel;
        item.block = position / block_size - 1;
        item.payload = m_staging.try_acquire(m_la_channel->block_bytes);
        if (reads.size() < num_blocks - first_block || !item.payload) {
            diagnostics.miss_reason = "the staging pool cannot hold the reload";
            return;
        }
        item.request.key.hash = state.identity.key(m_shared.root, item.block, m_la_channel->group);
        reads.push_back(std::move(item));
    } else if (reads.empty()) {
        diagnostics.miss_reason = "the staging pool cannot hold the reload";
        return;
    }
    diagnostics.degraded_chunks += read(reads);

    // Verify: a block that fails is a miss for itself and everything after it
    std::size_t verified_blocks = first_block;
    std::string failure;
    for (const BlockRead& item : reads) {
        diagnostics.store_read_blocks += 1;
        diagnostics.store_read_bytes += item.result.bytes_read;
    }
    for (const BlockRead& item : reads) {
        if (item.channel != m_kv_channel) {
            continue;
        }
        failure = verify(state, item);
        if (!failure.empty()) {
            // nothing from here on is trusted: it is written again once recomputed, which repairs the store in one pass
            forget_stored_from(state, item.block * block_size);
            break;
        }
        ++verified_blocks;
    }
    if (m_la_channel) {
        const std::string la_failure = verify(state, reads.back());
        if (!la_failure.empty()) {
            // the blocks before it verified; checkpoints from this one on are written again
            state.stored_checkpoints.erase(state.stored_checkpoints.lower_bound(position), state.stored_checkpoints.end());
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
        if (state && state->writes_immediately) {
            write_complete_blocks(*state, true);
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
    if (state->writes_immediately) {
        write_complete_blocks(*state, false);
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
    const SequenceGroup& group = *state.group;
    const Sequence::CPtr sequence = group[0];
    const std::uint64_t seq_id = sequence->get_id();
    if (!m_orchestrator->has_block_table(seq_id)) {
        return;
    }
    const std::size_t block_size = m_kv_channel->block_size_tokens;
    const std::vector<BlocksPerLayer>& kv_block_tables = m_orchestrator->get_kv_block_tables(seq_id);
    const std::size_t num_complete =
        std::min(group.get_num_processed_tokens() / block_size, kv_block_tables[0].size());
    // speculative decoding can take processed tokens back
    state.write_cursor = std::min(state.write_cursor, num_complete);
    if (state.write_cursor == num_complete) {
        return;
    }
    state.identity.extend(group, *sequence, num_complete);

    const PromptCachePolicy& policy = group.get_sampling_parameters().prompt_cache;
    const std::uint64_t expires_at = policy.ttl.is_unlimited() ? 0 : unix_seconds_now() + policy.ttl.get_seconds();
    const auto make_batch = [&](const StoreChannel* channel) {
        auto batch = std::make_shared<WriteBatch>();
        batch->store = m_shared.store;
        batch->channel = channel;
        batch->root = m_shared.root;
        batch->expires_at = expires_at;
        return batch;
    };
    const std::shared_ptr<WriteBatch> kv_batch = make_batch(m_kv_channel);
    const std::shared_ptr<WriteBatch> la_batch = make_batch(m_la_channel);
    std::vector<BlockTransfer> kv_transfers, la_transfers;

    for (std::size_t block = state.write_cursor; block < num_complete; ++block) {
        const std::size_t end_position = (block + 1) * block_size;
        // the linear attention state after this block, when a checkpoint ends here
        std::optional<std::size_t> la_row;
        if (m_la_channel && end_position % m_la_channel->block_size_tokens == 0) {
            const std::size_t la_position = end_position / m_la_channel->block_size_tokens - 1;
            const std::size_t la_start = m_orchestrator->get_linear_attention_block_table_logical_start(seq_id);
            const BlocksPerLayer& la_table = m_orchestrator->get_linear_attention_block_table(seq_id);
            if (la_position >= la_start && la_position - la_start < la_table.size()) {
                la_row = static_cast<std::size_t>(la_table[la_position - la_start]->get_index());
            }
        }
        // what the store already has stays as it is: the first write of a key wins
        const bool write_kv = block >= state.stored_blocks.size() || !state.stored_blocks[block];
        if (la_row && state.stored_checkpoints.count(end_position)) {
            la_row.reset();
        }
        const std::size_t bytes = (write_kv ? m_kv_channel->block_bytes : 0) + (la_row ? m_la_channel->block_bytes : 0);
        if (within_budget && m_step_copied_bytes > 0 &&
            m_step_copied_bytes + bytes > m_config.step_transfer_budget_bytes) {
            break;  // the rest waits for the next step; the sequence keeps its blocks meanwhile
        }
        StagingLease kv_payload = write_kv ? m_staging.try_acquire(m_kv_channel->block_bytes) : StagingLease();
        StagingLease la_payload = la_row ? m_staging.try_acquire(m_la_channel->block_bytes) : StagingLease();
        if ((write_kv && !kv_payload) || (la_row && !la_payload)) {
            break;  // staging is full of writes in flight
        }

        if (write_kv) {
            kv_transfers.push_back({static_cast<std::size_t>(kv_block_tables[0][block]->get_index()), kv_payload.data()});
            kv_batch->blocks.push_back({std::move(kv_payload),
                                        make_key(*m_kv_channel, state.identity.key(m_shared.root, block, m_kv_channel->group)),
                                        end_position,
                                        state.identity.witness_chain(block)});
        }
        if (la_row) {
            la_transfers.push_back({*la_row, la_payload.data()});
            la_batch->blocks.push_back({std::move(la_payload),
                                        make_key(*m_la_channel, state.identity.key(m_shared.root, block, m_la_channel->group)),
                                        end_position,
                                        state.identity.witness_chain(block)});
        }
        m_step_copied_bytes += bytes;
        state.write_cursor = block + 1;
    }

    CacheDiagnostics& diagnostics = state.group->get_cache_diagnostics();
    for (const auto& [transfers, batch] : {std::make_pair(&kv_transfers, kv_batch), std::make_pair(&la_transfers, la_batch)}) {
        if (transfers->empty()) {
            continue;
        }
        m_codec.download(*batch->channel, cache_tensors(*batch->channel), *transfers);
        diagnostics.pushed_blocks += transfers->size();
        diagnostics.pushed_bytes += transfers->size() * batch->channel->block_bytes;
        m_io_pool.submit([this, batch] {
            run_write_batch(*batch);
        });
    }
}

void CacheStoreTier::run_write_batch(WriteBatch& batch) {
    const StoreChannel& channel = *batch.channel;
    const std::size_t count = batch.blocks.size();
    std::vector<std::array<std::uint8_t, WitnessRecord::SIZE>> witnesses(count);
    std::vector<OvKVCacheStoreWriteRequest> requests(count);
    for (std::size_t i = 0; i < count; ++i) {
        const BlockWrite& block = batch.blocks[i];
        WitnessRecord record;
        record.expires_at = batch.expires_at;
        if (m_config.verify_payload_crc) {
            record.flags |= WitnessRecord::FLAG_PAYLOAD_CRC;
            record.payload_crc = crc32c(block.payload.data(), channel.block_bytes);
        }
        record.digest = compute_witness_digest(m_witness_key, batch.root, channel.group, block.end_position,
                                               block.witness_chain, record);
        witnesses[i] = record.encode();
        requests[i].key = block.key;
        requests[i].source = block.payload.data();
        requests[i].bytes = channel.block_bytes;
        requests[i].witness = {witnesses[i].data(), witnesses[i].size()};
    }

    std::vector<OvKVCacheStoreWriteResult> results(count);
    OvKVCacheStatus status = batch.store->write_blocks(requests.data(), results.data(), count);
    if (status < 0 && count > 1) {
        // the batch failed as a whole: the chunk degrades to one call per block
        for (std::size_t i = 0; i < count; ++i) {
            if (batch.store->write_blocks(&requests[i], &results[i], 1) < 0 && results[i].status == OV_KVCACHE_OK) {
                results[i].status = OV_KVCACHE_ERR_INTERNAL;
            }
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
            GENAI_WARN("Cache store: writing " + std::to_string(failed) + " of " + std::to_string(count) + " '" +
                       channel.group + "' blocks failed (" + status_to_string(first_failure) +
                       "); they stay recomputable. Further failures are only counted.");
        }
    }
    if (m_config.durability == CacheStoreConfig::Durability::Strict) {
        batch.store->flush();
    }
}

}  // namespace ov::genai::cache_store
