// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "continuous_batching/cache_store/hashing.hpp"

namespace ov::genai {
class SequenceGroup;
class Sequence;
}  // namespace ov::genai

namespace ov::genai::cache_store {

/**
 * @brief The host-private key of the witness digest. Kept in the user's cache directory, outside every store directory,
 * so a store provider cannot read it; created on first use. When it cannot be stored, a key for this process only is
 * used, and stored blocks become misses after a restart.
 */
SipHash128::Key load_witness_key();

/**
 * @brief The witness the host persists with every block (32 bytes, little-endian): format version, flags, payload
 * CRC32C, expiry (Unix seconds, 0 = never) and a keyed digest over the block's content, position, channel, namespace
 * and the fields before it.
 */
struct WitnessRecord {
    static constexpr std::size_t SIZE = 32;
    static constexpr std::uint8_t VERSION = 1;
    static constexpr std::uint8_t FLAG_PAYLOAD_CRC = 1;

    std::uint8_t flags = 0;
    std::uint32_t payload_crc = 0;
    std::uint64_t expires_at = 0;
    Digest128 digest{0, 0};

    std::array<std::uint8_t, SIZE> encode() const;
    /** @return False when @p bytes is not a record of this format. */
    static bool decode(const std::uint8_t* bytes, std::size_t size, WitnessRecord& record);
};

/**
 * @brief The digest a block's witness must carry: keyed with the host-private key over the channel, the token position
 * the block ends at, its witness chain value, the namespace's chain root and @p record's fields before the digest.
 */
Digest128 compute_witness_digest(const SipHash128::Key& witness_key,
                                 const Digest128& root,
                                 const std::string& group,
                                 std::size_t end_position,
                                 const Digest128& witness_chain,
                                 const WitnessRecord& record);

/**
 * @brief Content identity of one sequence, computed by the host from its tokens (or embeddings) block by block and
 * cached as the sequence grows (ARCHITECTURE.md §5):
 * - `D[b]`, a MurmurHash3 digest of block b's content;
 * - `W[b]`, a SipHash chain keyed with the host-private key over the raw content (the witness chain);
 * - `C[b]`, a MurmurHash3 chain over `D[b]` per namespace root, from which block keys are derived.
 */
class SequenceIdentity {
public:
    SequenceIdentity(std::size_t block_size, const SipHash128::Key& witness_key);

    /// Hashes blocks [num_blocks(), @p num_blocks) of the sequence; their tokens must all exist.
    void extend(const SequenceGroup& group, const Sequence& sequence, std::size_t num_blocks);

    std::size_t num_blocks() const {
        return m_block_digests.size();
    }

    /// @return The store key hash of block @p block in channel @p group of the namespace with chain root @p root.
    Digest128 key(const Digest128& root, std::size_t block, const std::string& group);

    /// @return `W[block]`, the witness chain up to and including @p block.
    const Digest128& witness_chain(std::size_t block) const;

private:
    const std::vector<Digest128>& chain(const Digest128& root, std::size_t num_blocks);

    std::size_t m_block_size;
    SipHash128::Key m_witness_key;
    std::vector<Digest128> m_block_digests;
    std::vector<Digest128> m_witness_chain;
    std::vector<std::pair<Digest128, std::vector<Digest128>>> m_chains;
    std::vector<std::uint8_t> m_content;
};

}  // namespace ov::genai::cache_store
