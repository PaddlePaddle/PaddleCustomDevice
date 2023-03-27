// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
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

#include <cnpapi.h>
#include <cnpapi_cndrv_id.h>

#include <mutex>
#include <unordered_map>

#include "paddle/phi/api/profiler/trace_event.h"
#include "paddle/phi/backends/device_ext.h"
#include "runtime/os_info.h"

struct ActivityBuffer {
  ActivityBuffer(uint64_t* addr, size_t size) : addr(addr), valid_size(size) {}
  uint64_t* addr;
  size_t valid_size;
};

void ProcessCnpapiActivityRecord(
    const cnpapiActivity* record,
    uint64_t start_ns,
    const std::unordered_map<uint32_t, uint64_t> tid_mapping,
    C_Profiler collector);

class Tracer {
 public:
  static Tracer& Instance() {
    static Tracer instance;
    return instance;
  }

  void AllocateBuffer(uint64_t** buffer, size_t* size);
  void ProduceBuffer(uint64_t* buffer, size_t valid_size);
  std::vector<ActivityBuffer> ConsumeBuffers();
  void ReleaseBuffer(uint64_t* buffer);

 private:
  Tracer() {}

  std::mutex activity_buffer_lock_;
  std::vector<ActivityBuffer> activity_buffers_;
};
