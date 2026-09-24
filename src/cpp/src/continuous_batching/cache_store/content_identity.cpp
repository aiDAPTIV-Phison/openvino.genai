// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/cache_store/content_identity.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>

#include "logger.hpp"
#include "sequence_group.hpp"

namespace ov::genai::cache_store {

namespace {

namespace fs = std::filesystem;

constexpr std::uint8_t TOKEN_CONTENT = 'T';
constexpr std::uint8_t EMBEDDING_CONTENT = 'E';

fs::path witness_key_path() {
    const fs::path relative = fs::path("openvino_genai") / "kvcache_store" / "witness.key";
#ifdef _WIN32
    if (const char* local_app_data = std::getenv("LOCALAPPDATA"); local_app_data && *local_app_data) {
        return fs::path(local_app_data) / relative;
    }
#else
    if (const char* cache_home = std::getenv("XDG_CACHE_HOME"); cache_home && *cache_home) {
        return fs::path(cache_home) / relative;
    }
    if (const char* home = std::getenv("HOME"); home && *home) {
        return fs::path(home) / ".cache" / relative;
    }
#endif
    return {};
}

bool read_key(const fs::path& path, SipHash128::Key& key) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }
    file.read(reinterpret_cast<char*>(key.data()), static_cast<std::streamsize>(key.size()));
    return file.gcount() == static_cast<std::streamsize>(key.size());
}

void store_le(std::uint64_t value, std::uint8_t* bytes, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
        bytes[i] = static_cast<std::uint8_t>(value >> (8 * i));
    }
}

std::uint64_t load_le(const std::uint8_t* bytes, std::size_t size) {
    std::uint64_t value = 0;
    for (std::size_t i = size; i > 0; --i) {
        value = (value << 8) | bytes[i - 1];
    }
    return value;
}

void append_le(std::vector<std::uint8_t>& bytes, std::uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        bytes.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
    }
}

}  // namespace

SipHash128::Key load_witness_key() {
    const fs::path path = witness_key_path();
    SipHash128::Key key;
    if (!path.empty() && read_key(path, key)) {
        return key;
    }
    std::random_device random;
    for (std::size_t i = 0; i < key.size(); i += 4) {
        const std::uint32_t value = random();
        std::memcpy(key.data() + i, &value, sizeof(value));
    }
    if (!path.empty()) {
        std::error_code error;
        fs::create_directories(path.parent_path(), error);
        const fs::path temp = path.string() + ".tmp" + std::to_string(random());
        {
            std::ofstream file(temp, std::ios::binary | std::ios::trunc);
            file.write(reinterpret_cast<const char*>(key.data()), static_cast<std::streamsize>(key.size()));
        }
        fs::rename(temp, path, error);
        fs::remove(temp, error);
        // another process may have created the key at the same time; whichever landed is the key
        SipHash128::Key stored;
        if (read_key(path, stored)) {
            return stored;
        }
    }
    GENAI_WARN("The cache store witness key could not be stored in '" + path.string() +
               "': blocks stored by this process cannot be verified after it exits.");
    return key;
}

std::array<std::uint8_t, WitnessRecord::SIZE> WitnessRecord::encode() const {
    std::array<std::uint8_t, SIZE> bytes{};
    bytes[0] = VERSION;
    bytes[1] = flags;
    store_le(payload_crc, bytes.data() + 4, 4);
    store_le(expires_at, bytes.data() + 8, 8);
    store_le(digest.lo, bytes.data() + 16, 8);
    store_le(digest.hi, bytes.data() + 24, 8);
    return bytes;
}

bool WitnessRecord::decode(const std::uint8_t* bytes, std::size_t size, WitnessRecord& record) {
    if (size != SIZE || !bytes || bytes[0] != VERSION) {
        return false;
    }
    record.flags = bytes[1];
    record.payload_crc = static_cast<std::uint32_t>(load_le(bytes + 4, 4));
    record.expires_at = load_le(bytes + 8, 8);
    record.digest.lo = load_le(bytes + 16, 8);
    record.digest.hi = load_le(bytes + 24, 8);
    return true;
}

SequenceIdentity::SequenceIdentity(std::size_t block_size, const SipHash128::Key& witness_key)
    : m_block_size(block_size),
      m_witness_key(witness_key) {}

void SequenceIdentity::extend(const SequenceGroup& group, const Sequence& sequence, std::size_t num_blocks) {
    for (std::size_t block = m_block_digests.size(); block < num_blocks; ++block) {
        const std::size_t begin = block * m_block_size;
        const std::size_t end = begin + m_block_size;
        m_content.clear();
        if (group.get_sequence_group_type() == SequenceGroupType::TOKENS) {
            const auto& prompt = group.get_prompt_ids();
            const auto& generated = sequence.get_generated_ids();
            OPENVINO_ASSERT(end <= prompt.size() + generated.size(),
                            "Cache store: block ", block, " is not complete, the sequence has ",
                            prompt.size() + generated.size(), " tokens");
            m_content.push_back(TOKEN_CONTENT);
            for (std::size_t position = begin; position < end; ++position) {
                const std::int64_t token =
                    position < prompt.size() ? prompt[position] : generated[position - prompt.size()];
                append_le(m_content, static_cast<std::uint64_t>(token));
            }
        } else {
            const auto& input = group.get_input_embeds();
            const auto& generated = sequence.get_generated_ids_embeds();
            OPENVINO_ASSERT(end <= input.size() + generated.size(),
                            "Cache store: block ", block, " is not complete, the sequence has ",
                            input.size() + generated.size(), " embeddings");
            m_content.push_back(EMBEDDING_CONTENT);
            for (std::size_t position = begin; position < end; ++position) {
                const std::vector<float>& row =
                    position < input.size() ? input[position] : generated[position - input.size()];
                const auto* row_bytes = reinterpret_cast<const std::uint8_t*>(row.data());
                m_content.insert(m_content.end(), row_bytes, row_bytes + row.size() * sizeof(float));
            }
        }
        m_block_digests.push_back(murmur3_128(m_content.data(), m_content.size()));

        SipHash128 witness_chain(m_witness_key);
        witness_chain.update(block == 0 ? Digest128{0, 0} : m_witness_chain.back());
        witness_chain.update(m_content.data(), m_content.size());
        m_witness_chain.push_back(witness_chain.finish());
    }
}

const std::vector<Digest128>& SequenceIdentity::chain(const Digest128& root, std::size_t num_blocks) {
    OPENVINO_ASSERT(num_blocks <= m_block_digests.size(), "Cache store: block ", num_blocks - 1, " is not hashed yet");
    auto it = std::find_if(m_chains.begin(), m_chains.end(), [&root](const auto& chain) {
        return chain.first == root;
    });
    if (it == m_chains.end()) {
        m_chains.emplace_back(root, std::vector<Digest128>{});
        it = std::prev(m_chains.end());
    }
    std::vector<Digest128>& chain = it->second;
    for (std::size_t block = chain.size(); block < num_blocks; ++block) {
        Murmur3Builder link;
        link.add(block == 0 ? root : chain.back()).add(m_block_digests[block]);
        chain.push_back(link.finish());
    }
    return chain;
}

Digest128 SequenceIdentity::key(const Digest128& root, std::size_t block, const std::string& group) {
    const std::vector<Digest128>& links = chain(root, block + 1);
    Murmur3Builder key;
    key.add(links[block]).add(group);
    return key.finish();
}

const Digest128& SequenceIdentity::witness_chain(std::size_t block) const {
    OPENVINO_ASSERT(block < m_witness_chain.size(), "Cache store: block ", block, " is not hashed yet");
    return m_witness_chain[block];
}

Digest128 compute_witness_digest(const SipHash128::Key& witness_key,
                                 const Digest128& root,
                                 const std::string& group,
                                 std::size_t end_position,
                                 const Digest128& witness_chain,
                                 const WitnessRecord& record) {
    SipHash128 digest(witness_key);
    digest.update("ov.genai.kvstore.witness.v1");
    digest.update_u64(group.size()).update(group);
    digest.update_u64(end_position);
    digest.update(witness_chain).update(root);
    digest.update_u64(WitnessRecord::VERSION).update_u64(record.flags);
    digest.update_u64(record.payload_crc).update_u64(record.expires_at);
    return digest.finish();
}

}  // namespace ov::genai::cache_store
