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

#include <acl/acl.h>
#include <acl/acl_prof.h>
#include <hccl/hccl.h>
#include <hccl/hccl_types.h>

#include "paddle/phi/extension.h"

#define RUNTIME_CHECK(func, success)                                          \
  do {                                                                        \
    auto acl_ret = func;                                                      \
    if (acl_ret != success) {                                                 \
      std::cerr << "Call " << #func << " failed : " << acl_ret << " at file " \
                << __FILE__ << " line " << __LINE__ << std::endl;             \
      {                                                                       \
        const char *aclRecentErrMsg = nullptr;                                \
        aclRecentErrMsg = aclGetRecentErrMsg();                               \
        if (aclRecentErrMsg != nullptr) {                                     \
          printf("%s\n", aclRecentErrMsg);                                    \
        } else {                                                              \
          printf("Failed to get recent error message.\n");                    \
        }                                                                     \
      }                                                                       \
      exit(-1);                                                               \
    }                                                                         \
  } while (0)

#define ACL_CHECK(func) RUNTIME_CHECK(func, ACL_ERROR_NONE)
#define HCCL_CHECK(func) RUNTIME_CHECK(func, HCCL_SUCCESS)

C_Status MemCpyH2D(const C_Device device,
                   void *dst,
                   const void *src,
                   size_t size);
C_Status MemCpyD2D(const C_Device device,
                   void *dst,
                   const void *src,
                   size_t size);
C_Status MemCpyD2H(const C_Device device,
                   void *dst,
                   const void *src,
                   size_t size);
C_Status AsyncMemCpyH2D(const C_Device device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size);
C_Status AsyncMemCpyD2D(const C_Device device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size);
C_Status AsyncMemCpyD2H(const C_Device device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size);

class AscendProfiler {
 public:
  static AscendProfiler &Instance() {
    static AscendProfiler ins;
    return ins;
  }

  AscendProfiler() {}

  ~AscendProfiler() {
    destroy_step_info();
    destroy_config();
  }

  void update_config(std::vector<uint32_t> device_ids,
                     aclprofAicoreMetrics metrics,
                     aclprofAicoreEvents *events,
                     uint64_t type) {
    devices_ids_ = device_ids;
    metrics_ = metrics;
    data_type_ = type;
  }

  void destroy_config() {
    if (config_) {
      ACL_CHECK(aclprofDestroyConfig(config_));
      config_ = nullptr;
    }
  }

  aclprofConfig *get_config() {
    if (config_ == nullptr) {
      config_ = aclprofCreateConfig(devices_ids_.data(),
                                    devices_ids_.size(),
                                    metrics_,
                                    nullptr,
                                    data_type_);
    }
    return config_;
  }

  aclprofStepInfo *get_step_info() {
    if (step_info_ == nullptr) {
      step_info_ = aclprofCreateStepInfo();
    }
    return step_info_;
  }

  void destroy_step_info() {
    if (step_info_) {
      aclprofDestroyStepInfo(step_info_);
      step_info_ = nullptr;
    }
  }

  void update_stream(aclrtStream stream) {
    if (stream_ == nullptr) {
      stream_ = stream;
      if (step_info_) {
        ACL_CHECK(aclprofGetStepTimestamp(step_info_, ACL_STEP_START, stream_));
      }
    }
  }

  aclrtStream get_stream() { return stream_; }

  void clear_stream() { stream_ = nullptr; }

  void start() {
    if (!start_) {
      ACL_CHECK(aclrtSynchronizeDevice());
      ACL_CHECK(aclprofStart(AscendProfiler::Instance().get_config()));
      start_ = true;
    }
  }

  void stop() {
    if (start_) {
      ACL_CHECK(aclrtSynchronizeDevice());
      ACL_CHECK(aclprofStop(AscendProfiler::Instance().get_config()));
      start_ = false;
    }
  }

  void step_start() {
    get_step_info();
    if (stream_ && step_info_) {
      ACL_CHECK(aclprofGetStepTimestamp(step_info_, ACL_STEP_START, stream_));
    }
  }

  void step_stop() {
    if (stream_ && step_info_) {
      ACL_CHECK(aclprofGetStepTimestamp(step_info_, ACL_STEP_END, stream_));
    }
    destroy_step_info();
  }

 private:
  std::vector<uint32_t> devices_ids_;
  aclprofAicoreMetrics metrics_;
  uint64_t data_type_;

  aclprofConfig *config_ = nullptr;
  aclprofStepInfo *step_info_ = nullptr;
  aclrtStream stream_ = nullptr;

  bool start_ = false;
};
