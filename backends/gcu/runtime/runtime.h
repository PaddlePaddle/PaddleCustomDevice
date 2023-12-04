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
#include <tops/tops_ext.h>

#include <string>

#include "paddle/phi/extension.h"

#define RT_DISALLOW_COPY_AND_ASSIGN(TypeName)     \
  TypeName(const TypeName &) = delete;            \
  TypeName(const TypeName &&) = delete;           \
  TypeName &operator=(const TypeName &) = delete; \
  TypeName &operator=(const TypeName &&) = delete

#define CHECK_COMMON(func, success)                                      \
  do {                                                                   \
    auto ret = (func);                                                   \
    if (ret != success) {                                                \
      std::cout << "[ERROR]" << __FILE__ << ":" << __LINE__ << ", Call " \
                << #func << " failed, ret:" << ret << std::endl;         \
      exit(-1);                                                          \
    }                                                                    \
  } while (false)

#define RT_CHECK(func) CHECK_COMMON(func, topsSuccess)
#define ECCL_CHECK(func) CHECK_COMMON(func, ecclSuccess)

class GcuDeviceGuard {
 public:
  explicit GcuDeviceGuard(int device) {
    RT_CHECK(topsGetDevice(&device_));
    if (device_ != device) {
      RT_CHECK(topsSetDevice(device));
      reset_device_ = true;
    }
  }

  ~GcuDeviceGuard() {
    if (reset_device_) {
      RT_CHECK(topsSetDevice(device_));
    }
  }

  GcuDeviceGuard() = delete;
  RT_DISALLOW_COPY_AND_ASSIGN(GcuDeviceGuard);

 private:
  int device_;
  bool reset_device_ = false;
};

void DeAllocScatter(void *ptr);

static bool UseScatterMemory() {
  static bool use_scatter_memory =
      (std::getenv("ENFLAME_ENABLE_PT_JIT_AOT_MIXED") != nullptr &&
       std::getenv("FLAGS_use_system_allocator") != nullptr)
          ? (std::string(std::getenv("ENFLAME_ENABLE_PT_JIT_AOT_MIXED")) ==
                 "true" &&
             std::string(std::getenv("FLAGS_use_system_allocator")) == "true")
          : false;
  return use_scatter_memory;
}

C_Status AsyncMemCpyH2D(const C_Device device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size);

C_Status AsyncMemCpyD2H(const C_Device device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size);

C_Status AsyncMemCpyD2D(const C_Device device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size);

std::string DumpHbm(void *ptr);

C_Status InitResource(const int32_t device_id);

void FinalizeResource();
