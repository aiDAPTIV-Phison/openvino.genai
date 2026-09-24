// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/cache_store/cache_store_tier.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <optional>

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

    auto state = std::make_unique<RequestState>(group, m_kv_channel->block_size_tokens, m_witness_key);
    state->write_cursor = group->get_num_processed_tokens() / m_kv_channel->block_size_tokens;
    const PromptCachePolicy& policy = sampling_params.prompt_cache;
    state->writes_immediately = m_config.write.device_to_store.trigger == WriteTrigger::Immediate &&
                                !policy.is_explicit_only() && !policy.opt_out_write_through;

    const std::uint64_t seq_id = (*group)[0]->get_id();
    std::lock_guard<std::mutex> lock(m_requests_mutex);
    m_requests[seq_id] = std::move(state);
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

        kv_transfers.push_back({static_cast<std::size_t>(kv_block_tables[0][block]->get_index()), kv_payload.data()});
        kv_batch->blocks.push_back({std::move(kv_payload),
                                    make_key(*m_kv_channel, state.identity.key(m_shared.root, block, m_kv_channel->group)),
                                    end_position,
                                    state.identity.witness_chain(block)});
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
