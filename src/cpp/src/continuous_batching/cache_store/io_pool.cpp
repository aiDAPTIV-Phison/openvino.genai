// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/cache_store/io_pool.hpp"

namespace ov::genai::cache_store {

IoPool::IoPool(std::size_t num_threads) {
    m_threads.reserve(num_threads);
    for (std::size_t i = 0; i < num_threads; ++i) {
        m_threads.emplace_back(&IoPool::run, this);
    }
}

IoPool::~IoPool() {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_stopping = true;
    }
    m_task_available.notify_all();
    for (std::thread& thread : m_threads) {
        thread.join();
    }
}

std::future<void> IoPool::submit(std::function<void()> task) {
    std::packaged_task<void()> packaged(std::move(task));
    std::future<void> done = packaged.get_future();
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_tasks.push(std::move(packaged));
    }
    m_task_available.notify_one();
    return done;
}

void IoPool::drain() {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_idle.wait(lock, [this] {
        return m_tasks.empty() && m_running == 0;
    });
}

void IoPool::run() {
    while (true) {
        std::packaged_task<void()> task;
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_task_available.wait(lock, [this] {
                return m_stopping || !m_tasks.empty();
            });
            if (m_tasks.empty()) {
                return;  // stopping, and nothing is left to run
            }
            task = std::move(m_tasks.front());
            m_tasks.pop();
            ++m_running;
        }
        // a packaged task stores what the task throws in its future
        task();
        // release what the task holds (staging leases of a write) before it counts as done for drain()
        task = std::packaged_task<void()>();
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            --m_running;
        }
        m_idle.notify_all();
    }
}

}  // namespace ov::genai::cache_store
