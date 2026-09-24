// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/cache_store/hashing.hpp"

#include <cstring>

namespace ov::genai::cache_store {

namespace {

inline std::uint64_t load_u64_le(const std::uint8_t* bytes) {
    std::uint64_t value = 0;
    for (int i = 7; i >= 0; --i) {
        value = (value << 8) | bytes[i];
    }
    return value;
}

inline void store_u64_le(std::uint64_t value, std::uint8_t* bytes) {
    for (int i = 0; i < 8; ++i) {
        bytes[i] = static_cast<std::uint8_t>(value >> (8 * i));
    }
}

inline std::uint64_t rotl64(std::uint64_t value, int bits) {
    return (value << bits) | (value >> (64 - bits));
}

inline std::uint64_t fmix64(std::uint64_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return k;
}

inline void sip_round(std::uint64_t& v0, std::uint64_t& v1, std::uint64_t& v2, std::uint64_t& v3) {
    v0 += v1;
    v1 = rotl64(v1, 13);
    v1 ^= v0;
    v0 = rotl64(v0, 32);
    v2 += v3;
    v3 = rotl64(v3, 16);
    v3 ^= v2;
    v0 += v3;
    v3 = rotl64(v3, 21);
    v3 ^= v0;
    v2 += v1;
    v1 = rotl64(v1, 17);
    v1 ^= v2;
    v2 = rotl64(v2, 32);
}

struct Crc32cTables {
    std::uint32_t table[8][256];

    Crc32cTables() {
        constexpr std::uint32_t polynomial = 0x82F63B78u;  // CRC-32C, reflected
        for (std::uint32_t byte = 0; byte < 256; ++byte) {
            std::uint32_t crc = byte;
            for (int bit = 0; bit < 8; ++bit) {
                crc = (crc & 1) ? (crc >> 1) ^ polynomial : crc >> 1;
            }
            table[0][byte] = crc;
        }
        for (std::uint32_t byte = 0; byte < 256; ++byte) {
            for (int slice = 1; slice < 8; ++slice) {
                const std::uint32_t previous = table[slice - 1][byte];
                table[slice][byte] = (previous >> 8) ^ table[0][previous & 0xFF];
            }
        }
    }
};

const Crc32cTables& crc32c_tables() {
    static const Crc32cTables tables;
    return tables;
}

}  // namespace

Digest128 murmur3_128(const void* data, std::size_t size) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    const std::size_t num_blocks = size / 16;
    std::uint64_t h1 = 0;
    std::uint64_t h2 = 0;
    constexpr std::uint64_t c1 = 0x87c37b91114253d5ULL;
    constexpr std::uint64_t c2 = 0x4cf5ad432745937fULL;

    for (std::size_t i = 0; i < num_blocks; ++i) {
        std::uint64_t k1 = load_u64_le(bytes + 16 * i);
        std::uint64_t k2 = load_u64_le(bytes + 16 * i + 8);

        k1 *= c1;
        k1 = rotl64(k1, 31);
        k1 *= c2;
        h1 ^= k1;
        h1 = rotl64(h1, 27);
        h1 += h2;
        h1 = h1 * 5 + 0x52dce729;

        k2 *= c2;
        k2 = rotl64(k2, 33);
        k2 *= c1;
        h2 ^= k2;
        h2 = rotl64(h2, 31);
        h2 += h1;
        h2 = h2 * 5 + 0x38495ab5;
    }

    const std::uint8_t* tail = bytes + 16 * num_blocks;
    std::uint64_t k1 = 0;
    std::uint64_t k2 = 0;
    const std::size_t tail_size = size & 15;
    for (std::size_t i = tail_size; i > 8; --i) {
        k2 ^= static_cast<std::uint64_t>(tail[i - 1]) << (8 * (i - 9));
    }
    if (tail_size > 8) {
        k2 *= c2;
        k2 = rotl64(k2, 33);
        k2 *= c1;
        h2 ^= k2;
    }
    for (std::size_t i = tail_size < 8 ? tail_size : 8; i > 0; --i) {
        k1 ^= static_cast<std::uint64_t>(tail[i - 1]) << (8 * (i - 1));
    }
    if (tail_size > 0) {
        k1 *= c1;
        k1 = rotl64(k1, 31);
        k1 *= c2;
        h1 ^= k1;
    }

    h1 ^= static_cast<std::uint64_t>(size);
    h2 ^= static_cast<std::uint64_t>(size);
    h1 += h2;
    h2 += h1;
    h1 = fmix64(h1);
    h2 = fmix64(h2);
    h1 += h2;
    h2 += h1;
    return Digest128{h1, h2};
}

Murmur3Builder& Murmur3Builder::add(const void* data, std::size_t size) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    m_bytes.insert(m_bytes.end(), bytes, bytes + size);
    return *this;
}

Murmur3Builder& Murmur3Builder::add(const Digest128& digest) {
    add_u64(digest.lo);
    return add_u64(digest.hi);
}

Murmur3Builder& Murmur3Builder::add_u64(std::uint64_t value) {
    std::uint8_t bytes[8];
    store_u64_le(value, bytes);
    return add(bytes, sizeof(bytes));
}

Digest128 Murmur3Builder::finish() const {
    return murmur3_128(m_bytes.data(), m_bytes.size());
}

SipHash128::SipHash128(const Key& key) : SipHash128(key, true) {}

SipHash128::SipHash128(const Key& key, bool output_128) : m_output_128(output_128) {
    const std::uint64_t k0 = load_u64_le(key.data());
    const std::uint64_t k1 = load_u64_le(key.data() + 8);
    m_v0 = 0x736f6d6570736575ULL ^ k0;
    m_v1 = 0x646f72616e646f6dULL ^ k1;
    m_v2 = 0x6c7967656e657261ULL ^ k0;
    m_v3 = 0x7465646279746573ULL ^ k1;
    if (m_output_128) {
        m_v1 ^= 0xee;
    }
}

void SipHash128::compress(std::uint64_t message) {
    m_v3 ^= message;
    sip_round(m_v0, m_v1, m_v2, m_v3);
    sip_round(m_v0, m_v1, m_v2, m_v3);
    m_v0 ^= message;
}

SipHash128& SipHash128::update(const void* data, std::size_t size) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    m_total_size += size;
    // complete a partial word left from the previous update
    while (size > 0 && m_tail_size > 0) {
        m_tail |= static_cast<std::uint64_t>(*bytes) << (8 * m_tail_size);
        ++bytes;
        --size;
        if (++m_tail_size == 8) {
            compress(m_tail);
            m_tail = 0;
            m_tail_size = 0;
        }
    }
    for (; size >= 8; bytes += 8, size -= 8) {
        compress(load_u64_le(bytes));
    }
    for (; size > 0; ++bytes, --size) {
        m_tail |= static_cast<std::uint64_t>(*bytes) << (8 * m_tail_size);
        ++m_tail_size;
    }
    return *this;
}

SipHash128& SipHash128::update(const Digest128& digest) {
    update_u64(digest.lo);
    return update_u64(digest.hi);
}

SipHash128& SipHash128::update_u64(std::uint64_t value) {
    std::uint8_t bytes[8];
    store_u64_le(value, bytes);
    return update(bytes, sizeof(bytes));
}

Digest128 SipHash128::finish() const {
    std::uint64_t v0 = m_v0, v1 = m_v1, v2 = m_v2, v3 = m_v3;
    const std::uint64_t last = (m_total_size << 56) | m_tail;
    v3 ^= last;
    sip_round(v0, v1, v2, v3);
    sip_round(v0, v1, v2, v3);
    v0 ^= last;
    v2 ^= m_output_128 ? 0xee : 0xff;
    for (int i = 0; i < 4; ++i) {
        sip_round(v0, v1, v2, v3);
    }
    Digest128 digest{v0 ^ v1 ^ v2 ^ v3, 0};
    if (m_output_128) {
        v1 ^= 0xdd;
        for (int i = 0; i < 4; ++i) {
            sip_round(v0, v1, v2, v3);
        }
        digest.hi = v0 ^ v1 ^ v2 ^ v3;
    }
    return digest;
}

std::uint64_t SipHash128::hash64(const Key& key, const void* data, std::size_t size) {
    SipHash128 hasher(key, false);
    hasher.update(data, size);
    return hasher.finish().lo;
}

std::uint32_t crc32c(const void* data, std::size_t size, std::uint32_t crc) {
    const auto& tables = crc32c_tables().table;
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    crc = ~crc;
    for (; size >= 8; bytes += 8, size -= 8) {
        const std::uint64_t word = load_u64_le(bytes) ^ crc;
        crc = tables[7][word & 0xFF] ^ tables[6][(word >> 8) & 0xFF] ^ tables[5][(word >> 16) & 0xFF] ^
              tables[4][(word >> 24) & 0xFF] ^ tables[3][(word >> 32) & 0xFF] ^ tables[2][(word >> 40) & 0xFF] ^
              tables[1][(word >> 48) & 0xFF] ^ tables[0][word >> 56];
    }
    for (; size > 0; ++bytes, --size) {
        crc = (crc >> 8) ^ tables[0][(crc ^ *bytes) & 0xFF];
    }
    return ~crc;
}

}  // namespace ov::genai::cache_store
