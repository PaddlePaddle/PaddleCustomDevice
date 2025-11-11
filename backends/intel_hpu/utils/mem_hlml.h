// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#include "utils/hlml_shm.h"

/**
 * Memory reporter for HLML infrastructure. It is de facto communication
 * mechanism between framework and hl-smi command.
 *
 * The purpose of communication is to allow framework send precise information
 * about memory usage.
 */
class HlMlMemoryReporter {
 public:
  class Error : public std::runtime_error {
   public:
    static constexpr std::size_t MAXLEN = 256;
    Error(const char* operation, int error);

   private:
    char buffer[MAXLEN];
  };

  /**
   * Creates memory reporter for shared memory object associated with
   * given device.
   *
   */
  explicit HlMlMemoryReporter(int device_index);

  /**
   * During destruction we cleanup resources.
   */
  ~HlMlMemoryReporter();

  /**
   * Publish memory usage into shared object
   */
  void PublishMemory(std::uint64_t bytes);

  /**
   * Updates timestamp in shared object
   */
  void PublishTimestamp(std::uint64_t timestamp);

  /**
   * Get path
   */
  std::string GetPath() const;

 private:
  std::string MakePath(int device_index);
  int OpenSharedObject();
  hlml_shm_data* PrepareSharedObject(int fd);
  hlml_shm_data* MmapSharedObject();

  std::string m_path;
  hlml_shm_data* m_data = nullptr;
  int m_fd = -1;
};

/**
 * Background thread to update shared object with actual
 * value how much memory is used.
 */
class HlMlMemoryUpdater {
 public:
  static constexpr int INTERVAL = 1;
  HlMlMemoryUpdater(std::shared_ptr<HlMlMemoryReporter> reporter,
                    std::function<std::uint64_t()> get_used_memory);
  ~HlMlMemoryUpdater();

  void stop();

 private:
  void thread_main();

  std::atomic_bool m_quit{false};
  std::shared_ptr<HlMlMemoryReporter> m_reporter;
  std::function<std::uint64_t()> m_get_used_memory;
  std::thread m_thread;
};
