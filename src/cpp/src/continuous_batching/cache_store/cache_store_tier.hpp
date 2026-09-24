// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "continuous_batching/cache_store/block_codec.hpp"
#include "continuous_batching/cache_store/content_identity.hpp"
#include "continuous_batching/cache_store/hashing.hpp"
#include "continuous_batching/cache_store/io_pool.hpp"
#include "continuous_batching/cache_store/staging_pool.hpp"
#include "continuous_batching/cache_store/store_channels.hpp"
#include "continuous_batching/cache_store/store_namespace.hpp"
#include "continuous_batching/cache_store/store_provider.hpp"
#include "openvino/core/any.hpp"
#include "openvino/genai/scheduler_config.hpp"
#include "sequence_group.hpp"

namespace ov::genai {
class CacheOrchestrator;
}

namespace ov::genai::cache_store {

/**
 * @brief The cache store tier of one pipeline: the host side of the KV cache store proposal ("the connector" of
 * provider_abi.hpp). It owns content identity, namespaces, staging and I/O, and talks to one store provider.
 *
 * Threads: `on_add_request` runs on the caller of `add_request`; the other hooks run on the pipeline's step thread;
 * every provider call runs on the I/O pool.
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

    /// Waits for every write in flight, then flushes the stores.
    ~CacheStoreTier();

    CacheStoreTier(const CacheStoreTier&) = delete;
    CacheStoreTier& operator=(const CacheStoreTier&) = delete;

    /// `add_request`, after the device prefix cache restored what it holds: starts tracking the request and reloads
    /// what the store has beyond the device prefix (discover, verify) into staging. Changes nothing in the device cache.
    void on_add_request(const SequenceGroup::Ptr& group);

    /// Start of a step, after new requests were pulled: allocates the device blocks of the reloads verified since the
    /// previous step and sets their requests' processed-token counts.
    void allocate_reloads(const std::vector<SequenceGroup::Ptr>& requests);

    /// Before `forward`, after scheduling: copies every allocated reload into the device cache (admit).
    void upload_reloads();

    /// After sampling and the linear attention checkpoint commit of a step: writes the blocks the scheduled requests
    /// completed, as far as the step's transfer budget allows.
    void after_step(const std::vector<SequenceGroup::Ptr>& requests, const std::vector<std::uint64_t>& scheduled_group_ids);

    /// Before the scheduler frees a sequence: writes every complete block not written yet, then forgets the sequence.
    void on_free_sequence(std::uint64_t seq_id);

private:
    /// A namespace opened in the provider: its bytes, its store instance and the root of its key chains.
    struct OpenNamespace {
        std::string scope;
        std::vector<std::uint8_t> bytes;
        std::shared_ptr<StoreInstance> store;
        Digest128 root{0, 0};
    };

    /// Verified store blocks waiting in staging to be admitted into the device cache.
    struct ReloadPlan {
        std::size_t device_tokens = 0;  // prompt tokens the device prefix cache supplied
        std::size_t target_tokens = 0;  // processed-token count after the reload
        std::vector<StagingLease> kv_payloads;  // blocks from device_tokens / block size on, in order
        StagingLease la_payload;                 // linear attention state at target_tokens
        bool allocated = false;
    };

    /// How a block is written, from what the reload's probes found in the store. Stored bytes depend on the history of
    /// the computation (FINDINGS F8), so the first write of a key stays and a present key is never written again.
    enum class StoreWrite : std::uint8_t {
        IfAbsent,  // never probed: written only when the store does not hold the key
        Always,    // absent, or present but failed verification: written, over any old one
        Never,     // present and trusted
    };

    /// What the tier keeps per request (one sequence).
    struct RequestState {
        RequestState(SequenceGroup::Ptr request_group, std::size_t block_size, const SipHash128::Key& witness_key)
            : group(std::move(request_group)),
              identity(block_size, witness_key) {}

        SequenceGroup::Ptr group;
        SequenceIdentity identity;
        // next block to write; blocks before it are in the store or came from it
        std::size_t write_cursor = 0;
        bool writes_immediately = false;
        std::unique_ptr<ReloadPlan> reload;
        // what the reload's probes found in the store; a block or checkpoint missing here was not probed (IfAbsent)
        std::vector<StoreWrite> kv_writes;                    // by KV block index
        std::map<std::size_t, StoreWrite> checkpoint_writes;  // by end position
    };

    /// One block to read and where it goes.
    struct BlockRead {
        const StoreChannel* channel = nullptr;
        std::size_t block = 0;  // KV block index the item ends with
        StagingLease payload;
        std::array<std::uint8_t, 2 * WitnessRecord::SIZE> witness{};
        OvKVCacheStoreReadRequest request{};
        OvKVCacheStoreReadResult result{};
    };

    /// One block to write: where its payload is and what identifies it.
    struct BlockWrite {
        StagingLease payload;
        OvKVCacheBlockKey key{};
        std::size_t end_position = 0;
        Digest128 witness_chain{0, 0};
        bool if_absent = false;  // probed on the I/O pool first, and skipped when the store holds the key
    };

    /// A batch of blocks of one channel, written by one provider call on the I/O pool.
    struct WriteBatch {
        std::shared_ptr<StoreInstance> store;
        const StoreChannel* channel = nullptr;
        Digest128 root{0, 0};
        std::uint64_t expires_at = 0;
        std::vector<BlockWrite> blocks;
    };

    CacheStoreTier(const CacheStoreConfig& config,
                   std::shared_ptr<CacheOrchestrator> orchestrator,
                   std::vector<StoreChannel> channels,
                   StoreFingerprints fingerprints,
                   std::shared_ptr<StoreProvider> provider);

    /// Opens the namespace of a scope; when the provider fails, falls back to the built-in "file" provider for good.
    bool open_namespace(const std::string& scope, OpenNamespace& opened);

    /// Copies the request's complete blocks from the write cursor on into staging and queues their writes.
    void write_complete_blocks(RequestState& state, bool within_budget);

    /// Discover and verify (guide section 11): probes the store beyond the device prefix, converges on the prefix every
    /// channel can serve, reads it into staging and verifies it.
    void plan_reload(RequestState& state);

    /// Probes the keys of @p blocks of a channel (each KV block index the item ends with); true = present.
    std::vector<bool> probe(RequestState& state, const StoreChannel& channel, const std::vector<std::size_t>& blocks);

    /// Reads the items in batches of consecutive blocks on the I/O pool; a batch that fails as a whole is retried per
    /// block. @return The number of batches that degraded to per-block reads.
    std::size_t read(std::vector<BlockRead>& reads);

    /// @return Why a block read back cannot be used, or an empty string when it is verified.
    std::string verify(RequestState& state, const BlockRead& read) const;

    /// Stops trusting what the probes found after @p token_position: blocks and checkpoints there are written again
    /// once the request recomputes them.
    void forget_stored_from(RequestState& state, std::size_t token_position) const;

    /// Runs on the I/O pool: drops the IfAbsent blocks the store already holds, completes the witnesses and calls the
    /// provider.
    void run_write_batch(WriteBatch& batch);

    std::vector<ov::Tensor> cache_tensors(const StoreChannel& channel) const;

    CacheStoreConfig m_config;
    std::shared_ptr<CacheOrchestrator> m_orchestrator;
    std::vector<StoreChannel> m_channels;
    const StoreChannel* m_kv_channel = nullptr;
    const StoreChannel* m_la_channel = nullptr;
    StoreFingerprints m_fingerprints;
    std::shared_ptr<StoreProvider> m_provider;
    OpenNamespace m_shared;
    SipHash128::Key m_witness_key{};
    StagingPool m_staging;
    BlockCodec m_codec;
    // bytes copied from the device cache for writes in the current step
    std::size_t m_step_copied_bytes = 0;

    std::mutex m_requests_mutex;
    std::map<std::uint64_t, std::unique_ptr<RequestState>> m_requests;  // by sequence id
    // sequences whose reload is allocated and waits for its upload before the next forward (step thread only)
    std::vector<std::uint64_t> m_pending_uploads;

    std::atomic<std::size_t> m_failed_writes{0};
    std::atomic<bool> m_reported_write_failure{false};

    // Declared last so it is destroyed first: its threads finish every queued write while the rest still exists
    IoPool m_io_pool;
};

}  // namespace ov::genai::cache_store
