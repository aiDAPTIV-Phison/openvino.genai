// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/cache_store/staging_pool.hpp"

#include <cstdlib>
#include <iterator>
#include <new>

#include "openvino/core/except.hpp"

#ifdef _WIN32
#    include <malloc.h>
#endif

namespace ov::genai::cache_store {

namespace {

std::size_t round_up(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

std::uint8_t* allocate_aligned(std::size_t bytes) {
#ifdef _WIN32
    void* memory = _aligned_malloc(bytes, StagingPool::ALIGNMENT);
#else
    void* memory = std::aligned_alloc(StagingPool::ALIGNMENT, bytes);
#endif
    if (!memory) {
        throw std::bad_alloc();
    }
    return static_cast<std::uint8_t*>(memory);
}

void free_aligned(std::uint8_t* memory) {
#ifdef _WIN32
    _aligned_free(memory);
#else
    std::free(memory);
#endif
}

}  // namespace

StagingLease::StagingLease(StagingPool* pool, std::size_t offset, std::size_t reserved, std::uint8_t* data, std::size_t size)
    : m_pool(pool),
      m_offset(offset),
      m_reserved(reserved),
      m_data(data),
      m_size(size) {}

StagingLease::StagingLease(StagingLease&& other) noexcept
    : m_pool(other.m_pool),
      m_offset(other.m_offset),
      m_reserved(other.m_reserved),
      m_data(other.m_data),
      m_size(other.m_size) {
    other.m_pool = nullptr;
    other.m_data = nullptr;
    other.m_size = 0;
}

StagingLease& StagingLease::operator=(StagingLease&& other) noexcept {
    if (this != &other) {
        release();
        m_pool = other.m_pool;
        m_offset = other.m_offset;
        m_reserved = other.m_reserved;
        m_data = other.m_data;
        m_size = other.m_size;
        other.m_pool = nullptr;
        other.m_data = nullptr;
        other.m_size = 0;
    }
    return *this;
}

StagingLease::~StagingLease() {
    release();
}

void StagingLease::release() {
    if (m_pool) {
        m_pool->release(m_offset, m_reserved);
        m_pool = nullptr;
        m_data = nullptr;
        m_size = 0;
    }
}

StagingPool::StagingPool(std::size_t bytes) : m_capacity(round_up(bytes, ALIGNMENT)) {
    OPENVINO_ASSERT(m_capacity > 0, "Cache store staging pool cannot be empty");
    m_base = allocate_aligned(m_capacity);
    m_free_ranges.emplace(0, m_capacity);
    m_free_bytes = m_capacity;
}

StagingPool::~StagingPool() {
    free_aligned(m_base);
}

StagingLease StagingPool::try_acquire(std::size_t bytes) {
    if (bytes == 0) {
        return {};
    }
    const std::size_t reserved = round_up(bytes, ALIGNMENT);
    std::lock_guard<std::mutex> lock(m_mutex);
    // first fit
    for (auto it = m_free_ranges.begin(); it != m_free_ranges.end(); ++it) {
        if (it->second < reserved) {
            continue;
        }
        const std::size_t offset = it->first;
        const std::size_t remaining = it->second - reserved;
        m_free_ranges.erase(it);
        if (remaining > 0) {
            m_free_ranges.emplace(offset + reserved, remaining);
        }
        m_free_bytes -= reserved;
        return StagingLease(this, offset, reserved, m_base + offset, bytes);
    }
    return {};
}

std::size_t StagingPool::free_bytes() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_free_bytes;
}

void StagingPool::release(std::size_t offset, std::size_t reserved) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto inserted = m_free_ranges.emplace(offset, reserved).first;
    // merge with the following range
    auto next = std::next(inserted);
    if (next != m_free_ranges.end() && inserted->first + inserted->second == next->first) {
        inserted->second += next->second;
        m_free_ranges.erase(next);
    }
    // merge with the preceding range
    if (inserted != m_free_ranges.begin()) {
        auto previous = std::prev(inserted);
        if (previous->first + previous->second == inserted->first) {
            previous->second += inserted->second;
            m_free_ranges.erase(inserted);
        }
    }
    m_free_bytes += reserved;
}

}  // namespace ov::genai::cache_store
