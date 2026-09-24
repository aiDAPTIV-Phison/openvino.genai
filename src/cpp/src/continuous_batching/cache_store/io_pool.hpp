// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace ov::genai::cache_store {

/**
 * @brief The host I/O pool (`CacheStoreConfig::io_threads`): every call into a store provider runs on one of its threads,
 * never while the caller holds a host lock. Tasks run in submission order on the first free thread.
 */
class IoPool {
public:
    explicit IoPool(std::size_t num_threads);
    /// Runs every queued task to completion, then joins the threads.
    ~IoPool();

    IoPool(const IoPool&) = delete;
    IoPool& operator=(const IoPool&) = delete;

    /// @return A future that becomes ready when @p task has run; it carries an exception the task threw.
    std::future<void> submit(std::function<void()> task);

    /// Blocks until every task submitted so far has run.
    void drain();

private:
    void run();

    std::mutex m_mutex;
    std::condition_variable m_task_available;
    std::condition_variable m_idle;
    std::queue<std::packaged_task<void()>> m_tasks;
    std::size_t m_running = 0;
    bool m_stopping = false;
    std::vector<std::thread> m_threads;
};

}  // namespace ov::genai::cache_store
