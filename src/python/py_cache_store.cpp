// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "openvino/genai/cache_store_config.hpp"
#include "openvino/genai/prompt_cache.hpp"

namespace py = pybind11;

using ov::genai::CacheDiagnostics;
using ov::genai::CacheStoreConfig;
using ov::genai::CacheTier;
using ov::genai::CacheTtl;
using ov::genai::ExplicitCache;
using ov::genai::ImplicitCache;
using ov::genai::PrefetchCacheOutcome;
using ov::genai::PromptCachePolicy;
using ov::genai::PushCacheOutcome;
using ov::genai::SessionScope;
using ov::genai::SharedScope;
using ov::genai::WriteTrigger;

namespace {

auto cache_store_config_docstring = R"(
    Cache store tier of the continuous batching pipeline: persists KV cache blocks through a store provider (a plugin
    library implementing the KV cache store C ABI, or the built-in "file" provider) and serves them back to later
    requests, also across process restarts. Set it as SchedulerConfig.cache_store.

    :param enable: master switch of the store tier.
    :param directory: location passed to the provider's open; required by the built-in "file" provider.
    :param capacity_bytes: capacity budget passed to the provider's open; 0 lets the provider decide.
    :param provider: provider name to select from the module, e.g. "file"; empty selects the first one.
    :param library_path: path of the plugin library; empty selects the built-in "file" provider.
    :param staging_bytes: size of the host staging pool every store transfer goes through.
    :param io_threads: number of host I/O threads calling the provider.
    :param durability: CacheStoreConfig.Durability.Relaxed or Strict.
    :param verify_key_tokens: verify blocks read back against their tokens; must stay True.
    :param verify_payload_crc: also store and verify a checksum of every block (default True: the witness proves the
                               block's identity, not the integrity of its bytes).
    :param step_transfer_budget_bytes: bytes copied between device cache and staging for store writes per step.
    :param write: engine-level write policy, write.device_to_store.trigger and write.host_to_store.trigger.
    :param properties: provider settings as a dict of strings; assign a whole dict, items of the returned copy cannot
                       be changed in place.
)";

auto prompt_cache_policy_docstring = R"(
    Request-level prompt cache policy (GenerationConfig.prompt_cache), used by a pipeline with a cache store.

    :param mode: ImplicitCache() (blocks are written by the store's write policy) or ExplicitCache() (written only by
                 push_cache).
    :param scope: SharedScope() or SessionScope(session_id): the namespace the request writes to; a session request
                  also reuses shared blocks.
    :param ttl: CacheTtl.unlimited() (the default), CacheTtl.standard() (one hour) or CacheTtl.seconds(n).
    :param opt_out_write_through: persist this request's blocks only when they are evicted from the device.
)";

auto cache_diagnostics_docstring = R"(
    Cache facts of one request and their interpretation (PerfMetrics.get_cache_diagnostics()).

    Hit counts do not overlap: device + host + store hit tokens is the request's total prompt reuse.
)";

}  // namespace

void init_cache_store(py::module_& m) {
    py::enum_<WriteTrigger>(m, "WriteTrigger", "When a KV cache block is persisted to the cache store.")
        .value("OnEvict", WriteTrigger::OnEvict)
        .value("Immediate", WriteTrigger::Immediate)
        .value("OnAccessSelective", WriteTrigger::OnAccessSelective)
        .value("ExplicitOnly", WriteTrigger::ExplicitOnly);

    py::class_<CacheStoreConfig> cache_store_config(m, "CacheStoreConfig", cache_store_config_docstring);

    py::enum_<CacheStoreConfig::Durability>(cache_store_config, "Durability")
        .value("Relaxed", CacheStoreConfig::Durability::Relaxed)
        .value("Strict", CacheStoreConfig::Durability::Strict);

    py::class_<CacheStoreConfig::MovementPolicy>(cache_store_config, "MovementPolicy")
        .def(py::init<>())
        .def_readwrite("trigger", &CacheStoreConfig::MovementPolicy::trigger);

    py::class_<CacheStoreConfig::WritePolicy>(cache_store_config, "WritePolicy")
        .def(py::init<>())
        .def_readwrite("device_to_store", &CacheStoreConfig::WritePolicy::device_to_store)
        .def_readwrite("host_to_store", &CacheStoreConfig::WritePolicy::host_to_store);

    cache_store_config.def(py::init<>())
        .def_readwrite("enable", &CacheStoreConfig::enable)
        .def_readwrite("directory", &CacheStoreConfig::directory)
        .def_readwrite("capacity_bytes", &CacheStoreConfig::capacity_bytes)
        .def_readwrite("provider", &CacheStoreConfig::provider)
        .def_readwrite("library_path", &CacheStoreConfig::library_path)
        .def_readwrite("staging_bytes", &CacheStoreConfig::staging_bytes)
        .def_readwrite("io_threads", &CacheStoreConfig::io_threads)
        .def_readwrite("durability", &CacheStoreConfig::durability)
        .def_readwrite("verify_key_tokens", &CacheStoreConfig::verify_key_tokens)
        .def_readwrite("verify_payload_crc", &CacheStoreConfig::verify_payload_crc)
        .def_readwrite("step_transfer_budget_bytes", &CacheStoreConfig::step_transfer_budget_bytes)
        .def_readwrite("write", &CacheStoreConfig::write)
        .def_readwrite("properties", &CacheStoreConfig::properties)
        .def("validate", &CacheStoreConfig::validate)
        .def("to_string", &CacheStoreConfig::to_string);

    py::class_<ImplicitCache>(m, "ImplicitCache", "Prompt cache mode: blocks are written by the store's write policy.")
        .def(py::init<>());
    py::class_<ExplicitCache>(m, "ExplicitCache", "Prompt cache mode: blocks are written only by push_cache.")
        .def(py::init<>());
    py::class_<SharedScope>(m, "SharedScope", "Prompt cache scope: the namespace shared by every request.")
        .def(py::init<>());
    py::class_<SessionScope>(m, "SessionScope", "Prompt cache scope: a namespace of the session's own.")
        .def(py::init([](const std::string& session_id) {
                 return SessionScope{session_id};
             }),
             py::arg("session_id"))
        .def_readwrite("session_id", &SessionScope::session_id);

    py::class_<CacheTtl>(m, "CacheTtl", "Time-to-live of the store entries a request writes.")
        .def_static("standard", &CacheTtl::standard, "One hour.")
        .def_static("seconds", &CacheTtl::seconds, py::arg("seconds"))
        .def_static("unlimited", &CacheTtl::unlimited)
        .def("is_unlimited", &CacheTtl::is_unlimited)
        .def("get_seconds", &CacheTtl::get_seconds);

    py::class_<PromptCachePolicy>(m, "PromptCachePolicy", prompt_cache_policy_docstring)
        .def(py::init<>())
        .def_readwrite("mode", &PromptCachePolicy::mode)
        .def_readwrite("scope", &PromptCachePolicy::scope)
        .def_readwrite("ttl", &PromptCachePolicy::ttl)
        .def_readwrite("opt_out_write_through", &PromptCachePolicy::opt_out_write_through)
        .def("is_explicit_only", &PromptCachePolicy::is_explicit_only)
        .def("is_session_scoped", &PromptCachePolicy::is_session_scoped)
        .def("validate", &PromptCachePolicy::validate)
        .def("to_string", &PromptCachePolicy::to_string);

    py::enum_<CacheTier>(m, "CacheTier", "Cache tier a request's reused prompt tokens were served from.")
        .value("NONE", CacheTier::None)
        .value("DEVICE", CacheTier::Device)
        .value("HOST", CacheTier::Host)
        .value("STORE", CacheTier::Store);

    py::class_<CacheDiagnostics>(m, "CacheDiagnostics", cache_diagnostics_docstring)
        .def(py::init<>())
        .def_readonly("prompt_tokens", &CacheDiagnostics::prompt_tokens)
        .def_readonly("device_hit_tokens", &CacheDiagnostics::device_hit_tokens)
        .def_readonly("host_hit_tokens", &CacheDiagnostics::host_hit_tokens)
        .def_readonly("store_hit_tokens", &CacheDiagnostics::store_hit_tokens)
        .def_readonly("store_probe_rounds", &CacheDiagnostics::store_probe_rounds)
        .def_readonly("store_probed_keys", &CacheDiagnostics::store_probed_keys)
        .def_readonly("store_read_blocks", &CacheDiagnostics::store_read_blocks)
        .def_readonly("store_read_bytes", &CacheDiagnostics::store_read_bytes)
        .def_readonly("pushed_blocks", &CacheDiagnostics::pushed_blocks)
        .def_readonly("pushed_bytes", &CacheDiagnostics::pushed_bytes)
        .def_readonly("degraded_chunks", &CacheDiagnostics::degraded_chunks)
        .def_readonly("served_by", &CacheDiagnostics::served_by)
        .def_readonly("miss_reason", &CacheDiagnostics::miss_reason)
        .def("interpretation", &CacheDiagnostics::interpretation);

    py::class_<PushCacheOutcome>(m, "PushCacheOutcome", "What ContinuousBatchingPipeline.push_cache() published.")
        .def(py::init<>())
        .def_readonly("groups", &PushCacheOutcome::groups, "Channel groups that received blocks.")
        .def_readonly("blocks", &PushCacheOutcome::blocks, "Blocks written; blocks the store held are not rewritten.")
        .def_readonly("bytes", &PushCacheOutcome::bytes, "Payload bytes written.")
        .def_readonly("epoch", &PushCacheOutcome::epoch, "Order of this push among the pipeline's pushes, from 1.")
        .def_readonly("diagnostics", &PushCacheOutcome::diagnostics, "Cache facts of the push's prefill.");

    py::class_<PrefetchCacheOutcome>(m,
                                     "PrefetchCacheOutcome",
                                     "What ContinuousBatchingPipeline.prefetch_cache() moved into the device cache.")
        .def(py::init<>())
        .def_readonly("tokens", &PrefetchCacheOutcome::tokens, "Prompt tokens the store supplied to the device cache.")
        .def_readonly("blocks", &PrefetchCacheOutcome::blocks, "Blocks read from the store.")
        .def_readonly("diagnostics", &PrefetchCacheOutcome::diagnostics, "Cache facts of the warm-up.");
}
