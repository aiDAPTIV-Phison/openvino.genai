// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
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
     * used for this model: it has no folder to fingerprint, or the provider fails to load or to open.
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

    /// Start of a step, after new requests were pulled: restores what the store holds for each new request, in rounds
    /// through the staging pool (read, verify, allocate the device blocks, copy them in), and sets the request's
    /// processed-token count. Blocks the calling (inference) thread until the restore is done.
    void allocate_reloads(const std::vector<SequenceGroup::Ptr>& requests);

    /// Before `forward`, after scheduling: copies the linear attention state of each restore into the device cache
    /// (scheduling may zero the row first).
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

    /// push_cache: the request added next with @p request_id publishes every complete block, whatever the trigger.
    /// @return Its outcome, ready once the request is freed and its writes are terminal.
    std::future<PushCacheOutcome> expect_push(std::uint64_t request_id);

    /// prefetch_cache: the request added next with @p request_id only brings stored blocks into the device cache.
    /// @return Its outcome, ready once the request is freed.
    std::future<PrefetchCacheOutcome> expect_prefetch(std::uint64_t request_id);

    /// Forgets an expected push or prefetch whose request could not be added; its future gets @p error.
    void cancel_expected(std::uint64_t request_id, std::exception_ptr error);

    /// `add_request`, before the device prefix restore: for a prefetch request, the number of prompt tokens to keep, i.e.
    /// the prefix the store can serve plus the one token that is always computed (probes only); for any other request,
    /// its whole prompt.
    std::size_t prompt_length_to_keep(const SequenceGroup::Ptr& group);

private:
    /// A namespace opened in the provider: its bytes, its store instance and the root of its key chains.
    struct OpenNamespace {
        std::string scope;
        std::vector<std::uint8_t> bytes;
        std::shared_ptr<StoreInstance> store;
        Digest128 root{0, 0};
    };

    /// What the probes found for a request, restored when the request is scheduled.
    struct ReloadPlan {
        std::size_t device_tokens = 0;  // prompt tokens the device prefix cache supplied
        std::size_t target_tokens = 0;  // processed-token count after the restore
        std::vector<const OpenNamespace*> kv_source;  // by KV block: the namespace it is read from
        const OpenNamespace* la_source = nullptr;     // where the state at target_tokens is read from
        StagingLease la_payload;  // the verified state at target_tokens, copied in after scheduling
        bool allocated = false;
    };

    /// How a block is written, from what the reload's probes found in the store. Stored bytes depend on the history of
    /// the computation, so the first write of a key stays and a present key is never written again.
    enum class StoreWrite : std::uint8_t {
        IfAbsent,  // never probed: written only when the store does not hold the key
        Always,    // absent, or present but failed verification: written, over any old one
        Never,     // present and trusted
    };

    /// When a request's blocks are written: the engine trigger (guide section 14.1) adjusted by the request's policy
    /// (section 14.2).
    enum class WriteMode : std::uint8_t {
        None,       // `explicit_only`, or an explicit-only request: only push_cache writes
        Immediate,  // every block as soon as it is complete
        OnAccess,   // the blocks the request reused from the device prefix cache (`on_access_selective`)
        OnEvict,    // when the device cache evicts the blocks, or when the pipeline is destroyed
        Push,       // push_cache: every complete block, including the ones the device prefix cache supplied
    };

    /// The outcome of one push_cache, completed by whichever thread accounts its last write.
    struct PushTracker {
        std::mutex mutex;
        std::promise<PushCacheOutcome> promise;
        PushCacheOutcome outcome;
        std::set<std::string> groups;
        std::size_t pending_batches = 0;  // queued write batches not accounted yet
        bool request_freed = false;
    };

    /// A push or prefetch announced before its request is added.
    struct ExpectedRequest {
        std::shared_ptr<PushTracker> push;                                // set for a push
        std::shared_ptr<std::promise<PrefetchCacheOutcome>> prefetch;     // set for a prefetch
        std::string miss_reason;  // a prefetch the store could not serve: why
    };

    /// What the tier keeps per request (one sequence).
    struct RequestState {
        RequestState(SequenceGroup::Ptr request_group, std::size_t block_size, const SipHash128::Key& witness_key)
            : group(std::move(request_group)),
              identity(block_size, witness_key) {}

        SequenceGroup::Ptr group;
        SequenceIdentity identity;
        // A session request reads its session namespace, then the shared one, and writes to its session namespace
        std::vector<const OpenNamespace*> read_namespaces;
        const OpenNamespace* write_namespace = nullptr;
        WriteMode write_mode = WriteMode::None;
        // Immediate: next block to write; blocks before it are in the store or came from it
        std::size_t write_cursor = 0;
        // OnAccess: the complete blocks the device prefix cache supplied, and the next of them to write
        std::size_t reused_blocks = 0;
        std::size_t access_cursor = 0;
        std::unique_ptr<ReloadPlan> reload;
        // What the reload's probe found in the request's namespaces, by KV block index; a block missing here was not
        // probed (IfAbsent). A linear attention state is written with the KV block it ends in, so it has no entry of its own.
        std::vector<StoreWrite> kv_writes;
        // push_cache or prefetch_cache that added the request
        ExpectedRequest expected;
    };

    /// The keys of a sequence's first KV blocks in one namespace, from its first block and in order; the key of a
    /// linear attention state follows the key of the KV block it ends in. Every provider call lists a sequence's keys
    /// in this order.
    struct KeyChain {
        std::vector<OvKVCacheBlockKey> keys;
        std::vector<std::size_t> kv_index;             // by KV block: where its key is in `keys`
        std::map<std::size_t, std::size_t> la_index;  // by KV block a checkpoint ends in: where the state's key is
    };

    /// What the probes found: the prefix every channel can serve (guide section 14.4), without any read.
    struct Discovery {
        std::size_t position = 0;  // common token position every channel can serve (not above the device prefix: miss)
        std::vector<const OpenNamespace*> kv_source;  // by KV block: the namespace it is read from, nullptr = absent
        const OpenNamespace* la_source = nullptr;     // where the state at `position` is read from
        std::string probe_failure;
    };

    /// One block to read and where it goes.
    struct BlockRead {
        const StoreChannel* channel = nullptr;
        const OpenNamespace* ns = nullptr;  // where the probe found it
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
        const StoreChannel* channel = nullptr;
        OvKVCacheBlockKey key{};
        std::size_t chain_index = 0;  // of the key in the batch's key chain
        std::size_t end_position = 0;
        Digest128 witness_chain{0, 0};
        bool if_absent = false;  // probed on the I/O pool first, and skipped when the store holds the key
    };

    /// Blocks of one sequence, in its order, written by one provider call on the I/O pool.
    struct WriteBatch {
        std::shared_ptr<StoreInstance> store;
        std::shared_ptr<const KeyChain> chain;  // the sequence's keys up to the batch's last block
        Digest128 root{0, 0};
        std::uint64_t expires_at = 0;
        std::vector<BlockWrite> blocks;
        std::shared_ptr<PushTracker> push;  // the push_cache the batch belongs to, if any
    };

    /// A block a finished `on_evict` request left in the device prefix cache: what its write needs once it is evicted.
    using WriteBackId = std::pair<CacheType, std::uint64_t>;  // cache type and prefix hash of a cached block
    struct WriteBack {
        const StoreChannel* channel = nullptr;
        const OpenNamespace* ns = nullptr;  // where the block is written
        std::shared_ptr<const KeyChain> chain;  // the keys of its sequence
        std::size_t chain_index = 0;
        std::size_t block = 0;  // KV block index the item ends with
        std::size_t end_position = 0;
        Digest128 witness_chain{0, 0};
        std::uint64_t ttl_seconds = 0;  // 0 = unlimited
        bool if_absent = true;
        // the KV block's state, or the state's KV block: the two are written back together
        std::optional<WriteBackId> partner;
    };

    /// A write batch submitted to an asynchronous store: everything the plugin may touch stays here, unmodified, until
    /// `wait` reports a terminal status (guide section 9).
    struct InflightWrite {
        std::shared_ptr<WriteBatch> batch;  // holds the payload leases
        std::vector<std::array<std::uint8_t, WitnessRecord::SIZE>> witnesses;
        std::vector<OvKVCacheStoreWriteRequest> requests;
        std::vector<const StoreChannel*> channels;  // of each request
        std::vector<OvKVCacheStoreWriteResult> results;
        OvKVCacheAsyncToken token = 0;
    };

    CacheStoreTier(const CacheStoreConfig& config,
                   std::shared_ptr<CacheOrchestrator> orchestrator,
                   std::vector<StoreChannel> channels,
                   StoreFingerprints fingerprints,
                   std::shared_ptr<StoreProvider> provider);

    /// Opens the namespace of a scope. @return false (with a warning) when the provider fails to open it.
    bool open_namespace(const std::string& scope, OpenNamespace& opened);

    /// @return The namespace of a session scope, opened on first use (a provider call outside every lock), or nullptr
    /// when the provider cannot open it (@p error says why).
    const OpenNamespace* session_namespace(const std::string& scope, std::string& error);

    /// Logs, at info level, a request's prompt length, the tokens the device cache and the store supplied, and the
    /// processed-token count it starts computing from. Called once that count is final.
    static void log_cache_hits(const SequenceGroup& group);

    /// Sets the request's read and write namespaces from its policy's scope; opens a session namespace on first use.
    void select_namespaces(RequestState& state, const PromptCachePolicy& policy);

    /// @return How the request's blocks are written, from the engine trigger and the request's policy.
    WriteMode write_mode_of(const PromptCachePolicy& policy) const;

    /// Takes the push or prefetch announced for a request id, if any.
    ExpectedRequest take_expected(std::uint64_t request_id);

    /// Accounts what a push's batch wrote (blocks and bytes by group); completes the push when it was the last thing
    /// pending.
    static void settle_push(const std::shared_ptr<PushTracker>& push,
                            const std::map<std::string, std::pair<std::size_t, std::uint64_t>>& written);

    /// The request of a push or prefetch is freed: completes the prefetch, or the push once its writes are accounted.
    static void finish_expected(RequestState& state);

    /// Immediate: writes the request's complete blocks from the write cursor on.
    void write_complete_blocks(RequestState& state, bool within_budget);

    /// OnAccess: writes the blocks the request reused from the device prefix cache.
    void write_reused_blocks(RequestState& state, bool within_budget);

    /// Writes KV blocks [first_block, end_block) of the request, and the linear attention checkpoints ending in them;
    /// what the store holds is skipped (StoreWrite). A synchronous store is written on the calling (inference) thread,
    /// in rounds through staging, until all are written, whatever the step budget. An asynchronous store gets them
    /// through the I/O pool: during a step (@p within_budget) as far as the step budget and the free staging allow;
    /// otherwise all of them, waiting for staging that writes in flight hold.
    /// @return The first block not handled: end_block, or where the step budget or the staging pool stopped it.
    std::size_t write_blocks(RequestState& state, std::size_t first_block, std::size_t end_block, bool within_budget);

    /// One round of write_blocks: copies blocks into staging until @p budgeted's step budget or the staging pool stops
    /// it, then writes them (synchronous store) or queues them (asynchronous store). @return The first block not
    /// handled.
    std::size_t write_round(RequestState& state, std::size_t first_block, std::size_t end_block, bool budgeted);

    /// Leases of all @p sizes, or none. With @p wait, waits for space that writes in flight hold.
    std::vector<StagingLease> acquire_staging(const std::vector<std::size_t>& sizes, bool wait);

    /// OnEvict: remembers the request's complete blocks and checkpoints, by prefix hash, for write-back on eviction.
    void remember_for_write_back(RequestState& state);

    /// Copies blocks out of the device cache (physical block index each) and writes them (synchronous store) or queues
    /// their writes. Not bound by the step budget: an evicted block is overwritten in this step, so when staging is
    /// full this waits for space. @return Which blocks were handled (all).
    std::vector<bool> write_back(const std::vector<std::pair<WriteBack, std::size_t>>& blocks);

    /// Destruction: writes back every remembered block still in the device cache, as far as staging allows per round.
    void write_back_cached_blocks();

    /// Waits until every queued write is terminal and its staging is free again.
    void wait_for_writes();

    /// Discover (guide section 11): probes the store with the prompt's keys and finds the prefix every channel can serve
    /// beyond the device prefix. Nothing is read: the request is restored when it is scheduled (allocate_reloads).
    void plan_reload(RequestState& state);

    /// Verify and admit (guide section 11) a scheduled request's plan, in rounds through staging; see allocate_reloads.
    void restore(RequestState& state);

    /// Discover only: probes each of the request's read namespaces with the prompt's key chain and takes each block
    /// from the first namespace that holds it. @return Nothing when the request cannot use the store
    /// (see plan_reload's guards).
    std::optional<Discovery> discover(RequestState& state, std::size_t device_tokens);

    /// @return The key chain of the sequence's first @p num_blocks KV blocks in a namespace; the request's identity
    /// must cover them.
    std::shared_ptr<const KeyChain> key_chain(RequestState& state, const OpenNamespace& ns, std::size_t num_blocks) const;

    /// Probes keys in one namespace. @return true = present. A failed probe counts as absent, sets @p failure and
    /// leaves @p probed false; so does a store without a probe, for which every key counts as present.
    std::vector<bool> probe(RequestState& state,
                            const OpenNamespace& ns,
                            const std::vector<OvKVCacheBlockKey>& keys,
                            bool& probed,
                            std::string& failure);

    /// Fetches the items in batches of consecutive blocks and verifies each (sets `failure`). A
    /// synchronous store is read on the calling thread. An asynchronous store gets every batch submitted
    /// at once from this thread; they are waited for in order, and the first KV block that fails verification cancels
    /// the batches after it, which can no longer be used. @return The number of batches that degraded to per-block
    /// reads.
    std::size_t fetch(RequestState& state, std::vector<BlockRead>& reads);

    /// Reads items [start, end) synchronously; a batch that fails as a whole is retried per block.
    /// @return Whether the batch degraded to per-block reads.
    bool read_batch(std::vector<BlockRead>& reads, std::size_t start, std::size_t end) const;

    /// @return Why a block read back cannot be used, or an empty string when it is verified.
    std::string verify(RequestState& state, const BlockRead& read) const;

    /// Stops trusting what the probe found from @p token_position on: blocks there are written again once the request
    /// recomputes them.
    void forget_stored_from(RequestState& state, std::size_t token_position) const;

    /// Runs on the I/O pool: probes the store with the sequence's keys, drops the IfAbsent blocks it already holds,
    /// completes the witnesses and calls the provider. An asynchronous store gets the batch submitted and handed to
    /// the completion thread.
    void run_write_batch(const std::shared_ptr<WriteBatch>& batch);

    /// Accounts a write batch whose status is terminal; a batch that failed as a whole is retried per KV block.
    void finish_write(InflightWrite& write, OvKVCacheStatus status);

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
    StoreProvider::BlockLayout m_block_layout;  // what every store is told at open
    OpenNamespace m_shared;
    // session scopes, by scope key; entries are never removed, so their addresses stay valid
    std::mutex m_sessions_mutex;
    std::map<std::string, std::unique_ptr<OpenNamespace>> m_sessions;
    std::atomic<bool> m_reported_session_failure{false};
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

    // push_cache / prefetch_cache announced before their requests are added, by request id
    std::mutex m_expected_mutex;
    std::map<std::uint64_t, ExpectedRequest> m_expected;
    std::atomic<std::uint64_t> m_push_epoch{0};

    // OnEvict: blocks of finished requests still in the device cache, by cache type and prefix hash, and the ones
    // evicted in the current step with their physical block index. Never held while calling the orchestrator.
    std::mutex m_write_back_mutex;
    std::map<WriteBackId, WriteBack> m_write_back;
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
