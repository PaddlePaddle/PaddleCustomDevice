// Copyright (c) 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All
// rights reserved.
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

#include <glog/logging.h>
#include <mccl.h>
#include <musa.h>
#include <musa_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mutex>
#include <optional>

#include "paddle/common/exception.h"
#include "paddle/phi/backends/device_ext.h"

namespace musa {

#if defined(PADDLE_WITH_MUSA)
#define MUSA_CHECK(EXPR)                                          \
  do {                                                            \
    const musaError_t __err = EXPR;                               \
    if (__err != musaSuccess) {                                   \
      PD_CHECK(false, "MUSA error: ", musaGetErrorString(__err)); \
    }                                                             \
  } while (0);

#define MUSA_CHECK_WARN(EXPR)                               \
  do {                                                      \
    const musaError_t __err = EXPR;                         \
    if (PD_UNLIKELY(__err != musaSuccess)) {                \
      auto error_unused = musaGetLastError();               \
      (void)error_unused;                                   \
      PD_WARN("MUSA warning: ", musaGetErrorString(__err)); \
    }                                                       \
  } while (0);

#define MUDNN_CHECK(rst, msg)                   \
  PD_CHECK(rst == ::musa::dnn::Status::SUCCESS, \
           __FUNCTION__,                        \
           " MUDNN failed at: ",                \
           msg);

#if defined(PADDLE_WITH_MCCL)

static std::string GetMcclVer() {
  static std::once_flag mcclGetVersionFlag;
  static std::string versionString;
  std::call_once(mcclGetVersionFlag, []() {
    int version = 0;
    mcclResult_t status = mcclGetVersion(&version);
    if (status != mcclSuccess || version < 100) {
      versionString = "Unknown MCCL version";
    } else {
      const int majorBase = version < 2900 ? 1000 : 10000;
      const int minorBase = 100;
      int mcclMajor = version / majorBase;
      int mcclMinor = (version % majorBase) / minorBase;
      int mcclPatch = version % minorBase;
      versionString = std::to_string(mcclMajor) + "." +
                      std::to_string(mcclMinor) + "." +
                      std::to_string(mcclPatch);
    }
  });
  return versionString;
}

extern inline std::string McclGetErrorWithVersion(mcclResult_t error) {
  return std::string(mcclGetErrorString(error)) + ", MCCL version " +
         GetMcclVer();
}

extern inline std::string GetMcclErrorDetailStr(
    mcclResult_t error, std::optional<std::string> processGroupFailureReason) {
  if (processGroupFailureReason != std::nullopt) {
    return *processGroupFailureReason;
  }
  return McclGetErrorWithVersion(error);
}

#define MCCL_CHECK(cmd)                                                       \
  do {                                                                        \
    mcclResult_t result = cmd;                                                \
    if (result != mcclSuccess) {                                              \
      std::string err =                                                       \
          "MCCL error at: " + std::string(__FILE__) + ":" +                   \
          std::to_string(__LINE__) + ", " + McclGetErrorWithVersion(result) + \
          "\n" +                                                              \
          GetMcclErrorDetailStr(result, "call mccl failed at paddle-musa");   \
      PD_CHECK(false, err);                                                   \
    }                                                                         \
  } while (0);

// Macro to print and abort on a non-successful MCCL return value.
#define MCCL_ASSERT(cmd)                                 \
  do {                                                   \
    mcclResult_t result = cmd;                           \
    if (result != mcclSuccess) {                         \
      std::string err = McclGetErrorWithVersion(result); \
      fprintf(stderr,                                    \
              "[Error] MCCL error at: %s:%d, %s\n",      \
              __FILE__,                                  \
              __LINE__,                                  \
              err.c_str());                              \
      abort();                                           \
    }                                                    \
  } while (0);
#endif  // PADDLE_WITH_MCCL

#define DECLARE_TYPE_FOR_MUSA(GPU_TYPE, CUSTME_TYPE) \
  using GPU_TYPE = CUSTME_TYPE;

DECLARE_TYPE_FOR_MUSA(gpuStream_t, musaStream_t);
DECLARE_TYPE_FOR_MUSA(gpuError_t, musaError_t);
DECLARE_TYPE_FOR_MUSA(gpuEvent_t, musaEvent_t)
DECLARE_TYPE_FOR_MUSA(gpuMemcpyKind, musaMemcpyKind);
DECLARE_TYPE_FOR_MUSA(gpuDeviceProp, musaDeviceProp);
// DECLARE_TYPE_FOR_GPU(dnnDataType_t, musaDataType_t);

#endif  // PADDLE_WITH_MUSA

}  // namespace musa
