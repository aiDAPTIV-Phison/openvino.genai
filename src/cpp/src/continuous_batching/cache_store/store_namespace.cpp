// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/cache_store/store_namespace.hpp"

#include <fstream>
#include <iterator>
#include <variant>

#include "openvino/core/version.hpp"
#include "openvino/genai/cache_store/sdk_fingerprint.h"
#include "openvino/runtime/properties.hpp"
#include "utils.hpp"

namespace ov::genai::cache_store {

namespace {

namespace fs = std::filesystem;

/// Adds a file's name and contents; files that do not exist add nothing.
void add_file_contents(Murmur3Builder& builder, const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return;
    }
    const std::vector<char> contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    builder.add(path.filename().string()).add_u64(contents.size()).add(contents.data(), contents.size());
}

/// Adds a file's name and size only: weights are too large to read at startup, and a changed file almost always changes
/// size (graph and configuration contents are fingerprinted in full).
void add_file_size(Murmur3Builder& builder, const fs::path& path) {
    std::error_code error;
    const auto size = fs::file_size(path, error);
    if (!error) {
        builder.add(path.filename().string()).add_u64(size);
    }
}

}  // namespace

bool compute_fingerprints(const std::filesystem::path& models_path,
                          const std::string& device,
                          const std::vector<StoreChannel>& channels,
                          StoreFingerprints& fingerprints,
                          std::string& reason) {
    std::error_code error;
    if (models_path.empty() || !fs::is_directory(models_path, error)) {
        reason = "the model was not loaded from a folder, so it has no identity to key stored blocks with";
        return false;
    }

    Murmur3Builder model;
    model.add("ov.genai.kvstore.model.v1");
    add_file_contents(model, models_path / "config.json");
    for (const char* graph : {"openvino_model", "openvino_language_model", "openvino_text_embeddings_model"}) {
        add_file_contents(model, models_path / (std::string(graph) + ".xml"));
        add_file_size(model, models_path / (std::string(graph) + ".bin"));
    }
    fingerprints.model = model.finish();

    Murmur3Builder tokenizer;
    tokenizer.add("ov.genai.kvstore.tokenizer.v1");
    add_file_contents(tokenizer, models_path / "openvino_tokenizer.xml");
    add_file_size(tokenizer, models_path / "openvino_tokenizer.bin");
    add_file_contents(tokenizer, models_path / "tokenizer_config.json");
    add_file_contents(tokenizer, models_path / "chat_template.jinja");
    fingerprints.tokenizer = tokenizer.finish();

    // The device's full name and the runtime build pin the byte layout of the cache tensors, which the shapes alone do
    // not: a device plugin may change its cache format between releases.
    std::string device_name = device;
    try {
        device_name += " / " + utils::singleton_core().get_property(device, ov::device::full_name);
    } catch (const ov::Exception&) {
        // a device without a full name is identified by its name alone
    }
    Murmur3Builder spec;
    spec.add("ov.genai.kvstore.spec.v1")
        .add(OV_KVCACHE_STORE_SDK_BUILD_FINGERPRINT)
        .add(ov::get_openvino_version().buildNumber)
        .add(device_name)
        .add(channels_to_string(channels))
        .add("key=murmur3_x64_128 witness=siphash_2_4_128 crc=crc32c");
    fingerprints.spec = spec.finish();
    return true;
}

std::string scope_key(const PromptCacheScope& scope) {
    if (const auto* session = std::get_if<SessionScope>(&scope)) {
        return "session:" + session->session_id;
    }
    return "shared";
}

std::vector<std::uint8_t> make_namespace(const std::string& scope, const StoreFingerprints& fingerprints) {
    std::vector<std::uint8_t> ns = {'O', 'V', 'K', 'V', 'N', 'S', '1', 0};
    const auto append_u64 = [&ns](std::uint64_t value, std::size_t size) {
        for (std::size_t i = 0; i < size; ++i) {
            ns.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
        }
    };
    append_u64(scope.size(), 4);
    ns.insert(ns.end(), scope.begin(), scope.end());
    for (const Digest128& digest : {fingerprints.model, fingerprints.tokenizer, fingerprints.spec}) {
        append_u64(digest.lo, 8);
        append_u64(digest.hi, 8);
    }
    return ns;
}

}  // namespace ov::genai::cache_store
