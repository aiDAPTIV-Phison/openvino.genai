// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>

namespace ov::genai::cache_store {

class StagingPool;

/**
 * @brief A range of the staging pool, owned by one transfer. Returned to the pool when destroyed; the host keeps a lease
 * alive until the provider call that uses its memory has returned, or its async token is terminal.
 */
class StagingLease {
public:
    StagingLease() = default;
    StagingLease(StagingLease&& other) noexcept;
    StagingLease& operator=(StagingLease&& other) noexcept;
    StagingLease(const StagingLease&) = delete;
    StagingLease& operator=(const StagingLease&) = delete;
    ~StagingLease();

    std::uint8_t* data() const {
        return m_data;
    }

    std::size_t size() const {
        return m_size;
    }

    explicit operator bool() const {
        return m_data != nullptr;
    }

private:
    friend class StagingPool;
    StagingLease(StagingPool* pool, std::size_t offset, std::size_t reserved, std::uint8_t* data, std::size_t size);
    void release();

    StagingPool* m_pool = nullptr;
    std::size_t m_offset = 0;
    std::size_t m_reserved = 0;
    std::uint8_t* m_data = nullptr;
    std::size_t m_size = 0;
};

/**
 * @brief The host staging pool (`CacheStoreConfig::staging_bytes`): one page-aligned host allocation from which every
 * block moving between the device cache and a store provider takes a lease. Leases are page aligned.
 */
class StagingPool {
public:
    static constexpr std::size_t ALIGNMENT = 4096;

    explicit StagingPool(std::size_t bytes);
    ~StagingPool();

    StagingPool(const StagingPool&) = delete;
    StagingPool& operator=(const StagingPool&) = delete;

    /** @return A lease of @p bytes, or an empty lease when the pool has no free range that large right now. */
    StagingLease try_acquire(std::size_t bytes);

    std::uint8_t* base() const {
        return m_base;
    }

    std::size_t capacity() const {
        return m_capacity;
    }

    std::size_t free_bytes() const;

    /** @return How many leases were returned so far; with wait_for_release, a wait for space without missing a release. */
    std::uint64_t release_count() const;

    /** @brief Blocks until more than @p seen leases have been returned in total. */
    void wait_for_release(std::uint64_t seen);

private:
    friend class StagingLease;
    void release(std::size_t offset, std::size_t reserved);

    std::uint8_t* m_base = nullptr;
    std::size_t m_capacity = 0;
    mutable std::mutex m_mutex;
    std::map<std::size_t, std::size_t> m_free_ranges;  // offset -> length, coalesced
    std::size_t m_free_bytes = 0;
    std::uint64_t m_releases = 0;
    std::condition_variable m_released;
};

}  // namespace ov::genai::cache_store
