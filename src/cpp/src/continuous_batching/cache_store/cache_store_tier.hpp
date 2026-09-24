// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "continuous_batching/cache/cache_overwrite_observer.hpp"
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
 * provider calls run on the I/O pool, the completion thread, or (asynchronous reads) the caller of `add_request`.
 */
class CacheStoreTier : public ICacheOverwriteObserver {
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

    /// Writes back the blocks `on_evict` still holds in the device cache, waits for every write in flight, then flushes
    /// the stores.
    ~CacheStoreTier() override;

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
    /// completed (`immediate`) or reused from the device prefix cache (`on_access_selective`), as far as the step's
    /// transfer budget allows.
    void after_step(const std::vector<SequenceGroup::Ptr>& requests, const std::vector<std::uint64_t>& scheduled_group_ids);

    /// Before the scheduler frees a sequence: writes what its trigger still owes, or (`on_evict`) remembers its blocks
    /// for write-back when the device cache evicts them; then forgets the sequence.
    void on_free_sequence(std::uint64_t seq_id);

    /// Allocation evicts a cached block: an `on_evict` block is queued for write-back.
    void on_overwrite(CacheType type, uint64_t hash, size_t block_index) override;

    /// Inside `schedule()`, before any device write of the step: copies the evicted blocks out and queues their writes.
    void before_block_writes() override;

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

    /// When a request's blocks are written: the engine trigger (guide section 14.1) adjusted by the request's policy
    /// (section 14.2; interpretations I7-I10).
    enum class WriteMode : std::uint8_t {
        None,       // `explicit_only`, or an explicit-only request: only push_cache writes
        Immediate,  // every block as soon as it is complete
        OnAccess,   // the blocks the request reused from the device prefix cache (`on_access_selective`)
        OnEvict,    // when the device cache evicts the blocks, or when the pipeline is destroyed
    };

    /// What the tier keeps per request (one sequence).
    struct RequestState {
        RequestState(SequenceGroup::Ptr request_group, std::size_t block_size, const SipHash128::Key& witness_key)
            : group(std::move(request_group)),
              identity(block_size, witness_key) {}

        SequenceGroup::Ptr group;
        SequenceIdentity identity;
        WriteMode write_mode = WriteMode::None;
        // Immediate: next block to write; blocks before it are in the store or came from it
        std::size_t write_cursor = 0;
        // OnAccess: the complete blocks the device prefix cache supplied, and the next of them to write
        std::size_t reused_blocks = 0;
        std::size_t access_cursor = 0;
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
        std::string failure;  // why the block cannot be used; empty when it is verified
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

    /// A block a finished `on_evict` request left in the device prefix cache: what its write needs once it is evicted.
    struct WriteBack {
        const StoreChannel* channel = nullptr;
        OvKVCacheContentHash key{0, 0};
        std::size_t end_position = 0;
        Digest128 witness_chain{0, 0};
        std::uint64_t ttl_seconds = 0;  // 0 = unlimited
        bool if_absent = true;
    };

    /// A write batch submitted to an asynchronous store: everything the plugin may touch stays here, unmodified, until
    /// `wait` reports a terminal status (guide section 9).
    struct InflightWrite {
        std::shared_ptr<WriteBatch> batch;  // holds the payload leases
        std::vector<std::array<std::uint8_t, WitnessRecord::SIZE>> witnesses;
        std::vector<OvKVCacheStoreWriteRequest> requests;
        std::vector<OvKVCacheStoreWriteResult> results;
        OvKVCacheAsyncToken token = 0;
    };

    CacheStoreTier(const CacheStoreConfig& config,
                   std::shared_ptr<CacheOrchestrator> orchestrator,
                   std::vector<StoreChannel> channels,
                   StoreFingerprints fingerprints,
                   std::shared_ptr<StoreProvider> provider);

    /// Opens the namespace of a scope; when the provider fails, falls back to the built-in "file" provider for good.
    bool open_namespace(const std::string& scope, OpenNamespace& opened);

    /// @return How the request's blocks are written, from the engine trigger and the request's policy.
    WriteMode write_mode_of(const PromptCachePolicy& policy) const;

    /// Immediate: writes the request's complete blocks from the write cursor on.
    void write_complete_blocks(RequestState& state, bool within_budget);

    /// OnAccess: writes the blocks the request reused from the device prefix cache.
    void write_reused_blocks(RequestState& state, bool within_budget);

    /// Copies KV blocks [first_block, end_block) of the request, and the linear attention checkpoints ending in them,
    /// into staging and queues their writes; what the store holds is skipped (StoreWrite).
    /// @return The first block not handled: end_block, or where the step budget or the staging pool stopped it.
    std::size_t write_blocks(RequestState& state, std::size_t first_block, std::size_t end_block, bool within_budget);

    /// OnEvict: remembers the request's complete blocks and checkpoints, by prefix hash, for write-back on eviction.
    void remember_for_write_back(RequestState& state);

    /// Copies blocks out of the device cache (physical block index each) and queues their writes. Not bound by the step
    /// budget: an evicted block is overwritten in this step. @return Which blocks were queued; the others found no
    /// staging space.
    std::vector<bool> write_back(const std::vector<std::pair<WriteBack, std::size_t>>& blocks);

    /// Destruction: writes back every remembered block still in the device cache, as far as staging allows per round.
    void write_back_cached_blocks();

    /// Waits until every queued write is terminal and its staging is free again.
    void wait_for_writes();

    /// Discover and verify (guide section 11): probes the store beyond the device prefix, converges on the prefix every
    /// channel can serve, reads it into staging and verifies it.
    void plan_reload(RequestState& state);

    /// Probes the keys of @p blocks of a channel (each KV block index the item ends with); true = present.
    std::vector<bool> probe(RequestState& state, const StoreChannel& channel, const std::vector<std::size_t>& blocks);

    /// Fetches the items in batches of consecutive blocks of one channel and verifies each (sets `failure`). A
    /// synchronous store reads the batches in parallel on the I/O pool. An asynchronous store gets every batch submitted
    /// at once from this thread; they are waited for in order, and the first KV block that fails verification cancels
    /// the batches after it, which can no longer be used. @return The number of batches that degraded to per-block
    /// reads.
    std::size_t fetch(RequestState& state, std::vector<BlockRead>& reads);

    /// Reads items [start, end) synchronously; a batch that fails as a whole is retried per block.
    /// @return Whether the batch degraded to per-block reads.
    bool read_batch(std::vector<BlockRead>& reads, std::size_t start, std::size_t end) const;

    /// @return Why a block read back cannot be used, or an empty string when it is verified.
    std::string verify(RequestState& state, const BlockRead& read) const;

    /// Stops trusting what the probes found after @p token_position: blocks and checkpoints there are written again
    /// once the request recomputes them.
    void forget_stored_from(RequestState& state, std::size_t token_position) const;

    /// Runs on the I/O pool: drops the IfAbsent blocks the store already holds, completes the witnesses and calls the
    /// provider. An asynchronous store gets the batch submitted and handed to the completion thread.
    void run_write_batch(const std::shared_ptr<WriteBatch>& batch);

    /// Accounts a write batch whose status is terminal; a batch that failed as a whole is retried per block.
    void finish_write(const WriteBatch& batch,
                      const std::vector<OvKVCacheStoreWriteRequest>& requests,
                      std::vector<OvKVCacheStoreWriteResult>& results,
                      OvKVCacheStatus status);

    /// Completion thread: waits for the submitted write batches in order, the only waiter of their tokens.
    void complete_writes();

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

    // OnEvict: blocks of finished requests still in the device cache, by cache type and prefix hash, and the ones
    // evicted in the current step with their physical block index. Never held while calling the orchestrator.
    std::mutex m_write_back_mutex;
    std::map<std::pair<CacheType, std::uint64_t>, WriteBack> m_write_back;
    std::vector<std::pair<WriteBack, std::size_t>> m_evicted;
    std::atomic<bool> m_reported_write_back_loss{false};

    // Write batches an asynchronous store is working on, and the thread that waits for them (started with the first)
    std::mutex m_inflight_mutex;
    std::condition_variable m_inflight_added;
    std::condition_variable m_inflight_done;
    std::deque<std::unique_ptr<InflightWrite>> m_inflight_writes;
    std::size_t m_inflight_count = 0;  // submitted and not yet accounted
    bool m_stop_completion = false;
    std::thread m_completion_thread;

    // Declared last so it is destroyed first: its threads finish every queued write while the rest still exists
    IoPool m_io_pool;
};

}  // namespace ov::genai::cache_store
