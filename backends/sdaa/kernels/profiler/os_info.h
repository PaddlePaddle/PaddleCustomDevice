// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <stdlib.h>
#include <unistd.h>

#include <cassert>
#include <iostream>
#include <unordered_map>

// All kinds of Ids for OS thread
struct ThreadId {
  uint64_t std_tid = 0;   // std::hash<std::thread::id>
  uint64_t sys_tid = 0;   // OS-specific, Linux: gettid
  uint32_t sdaa_tid = 0;  // thread_id used by SDAA SDPTI
};

class InternalThreadId {
 public:
  InternalThreadId();

  const ThreadId &GetTid() const { return id_; }

 private:
  ThreadId id_;
};

uint32_t GetProcessId();

void *AlignMalloc(size_t size, size_t alignment);

void AlignFree(void *mem_ptr);
