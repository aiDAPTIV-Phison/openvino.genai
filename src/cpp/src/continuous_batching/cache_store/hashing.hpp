// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "openvino/genai/cache_store/provider_abi.hpp"

namespace ov::genai::cache_store {

/**
 * @brief 128-bit digest in the store ABI layout: `lo` is the first 64-bit half of the digest, `hi` the second.
 */
using Digest128 = OvKVCacheContentHash;

inline bool operator==(const Digest128& lhs, const Digest128& rhs) {
    return lhs.lo == rhs.lo && lhs.hi == rhs.hi;
}

inline bool operator!=(const Digest128& lhs, const Digest128& rhs) {
    return !(lhs == rhs);
}

/**
 * @brief MurmurHash3_x64_128 with seed 0, the host's content-key hash. Stable across builds and platforms (the input is
 * read little-endian). `lo` is the reference implementation's h1, `hi` its h2.
 */
Digest128 murmur3_128(const void* data, std::size_t size);

/**
 * @brief Collects the bytes of a MurmurHash3 input that arrives in pieces, then hashes them in one pass.
 */
class Murmur3Builder {
public:
    Murmur3Builder& add(const void* data, std::size_t size);
    Murmur3Builder& add(std::string_view text) {
        return add(text.data(), text.size());
    }
    Murmur3Builder& add(const Digest128& digest);
    Murmur3Builder& add_u64(std::uint64_t value);

    Digest128 finish() const;

private:
    std::vector<std::uint8_t> m_bytes;
};

/**
 * @brief Incremental SipHash-2-4 with a 128-bit output (the reference "siphash" with outlen 16), the host's keyed
 * witness digest. The key is 16 bytes.
 */
class SipHash128 {
public:
    using Key = std::array<std::uint8_t, 16>;

    explicit SipHash128(const Key& key);

    SipHash128& update(const void* data, std::size_t size);
    SipHash128& update(std::string_view text) {
        return update(text.data(), text.size());
    }
    SipHash128& update(const Digest128& digest);
    SipHash128& update_u64(std::uint64_t value);

    /** @return The digest; `lo` holds output bytes 0-7, `hi` bytes 8-15 (both little-endian). */
    Digest128 finish() const;

    /** @return The 64-bit SipHash-2-4 (reference outlen 8), used only to check the implementation against the paper. */
    static std::uint64_t hash64(const Key& key, const void* data, std::size_t size);

private:
    SipHash128(const Key& key, bool output_128);

    void compress(std::uint64_t message);

    std::uint64_t m_v0, m_v1, m_v2, m_v3;
    std::uint64_t m_tail = 0;
    std::size_t m_tail_size = 0;
    std::uint64_t m_total_size = 0;
    bool m_output_128;
};

/** @brief CRC-32C (Castagnoli), as used for iSCSI and ext4; `crc32c("123456789") == 0xE3069283`. */
std::uint32_t crc32c(const void* data, std::size_t size, std::uint32_t crc = 0);

}  // namespace ov::genai::cache_store
