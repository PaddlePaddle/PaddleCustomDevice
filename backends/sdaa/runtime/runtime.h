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

#include <sdaa_runtime.h>
#include <sys/syscall.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

#include "glog/logging.h"
#include "kernels/profiler/RecordEvent.h"
#include "kernels/profiler/os_info.h"
#include "kernels/profiler/sdaa_wrapper.h"
#include "paddle/phi/backends/device_ext.h"
#include "tccl.h"      //NOLINT
#include "tecoblas.h"  //NOLINT
#include "tecodnn.h"   // NOLINT

struct CustomSDAAStream {
  sdaaStream_t pStream;
  tecodnnHandle_t dnnHandle;
  tblasHandle_t tblasHandle;
  ~CustomSDAAStream() {
    if (dnnHandle) {
      tecodnnDestroy(dnnHandle);
      dnnHandle = nullptr;
    }
    if (tblasHandle) {
      tblasDestroy(tblasHandle);
      tblasHandle = nullptr;
    }
    if (pStream) {
      sdaaStreamDestroy(pStream);
      pStream = nullptr;
    }
  }
};
typedef CustomSDAAStream *CustomSDAAStream_t;

struct lastCommStream {
  static lastCommStream &Instance() {
    static lastCommStream last_comm_stream;
    return last_comm_stream;
  }

  void update(sdaaStream_t c_stream) {
    if (p_stream == nullptr) {
      VLOG(4) << "previous comm stream is nullptr, set to stream: " << c_stream;
      p_stream = c_stream;
    } else if (p_stream != c_stream) {
      VLOG(4) << "use different comm_stream, prev: " << p_stream
              << " current: " << c_stream;
      VLOG(4) << "sync prev_stream: " << p_stream;
      checkSdaaErrors(sdaaStreamSynchronize(p_stream));
      p_stream = c_stream;
    }
  }

  sdaaStream_t get() { return p_stream; }

 private:
  lastCommStream() = default;
  sdaaStream_t p_stream = nullptr;
};

#define CHECK_SDPTI(call)                             \
  do {                                                \
    SDptiResult err = call;                           \
    if (err != SDPTI_SUCCESS) {                       \
      fprintf(stderr, "%s:%d\n", "SDPTI_ERROR", err); \
      exit(EXIT_FAILURE);                             \
    }                                                 \
  } while (0)

inline tecodnnHandle_t GetHandle(void *stream) {
  return reinterpret_cast<CustomSDAAStream_t>(stream)->dnnHandle;
}
inline sdaaStream_t GetStream(void *cstream) {
  return reinterpret_cast<CustomSDAAStream_t>(cstream)->pStream;
}
inline tblasHandle_t GetBlasHandle(void *stream) {
  return reinterpret_cast<CustomSDAAStream_t>(stream)->tblasHandle;
}

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
C_Status DeviceAllocate(const C_Device device, void **ptr, size_t size);
C_Status DeviceMemSet(const C_Device device,
                      void *ptr,
                      unsigned char value,
                      size_t size);
C_Status HostAllocate(const C_Device device, void **ptr, size_t size);
C_Status DeviceDeallocate(const C_Device device, void *ptr, size_t size);
C_Status DeviceMemStats(const C_Device device,
                        size_t *total_memory,
                        size_t *free_memory);
C_Status SetDevice(const C_Device device);
C_Status DestroyStream(const C_Device device, C_Stream stream);
C_Status CreateEvent(const C_Device device, C_Event *event);
C_Status RecordEvent(const C_Device device, C_Stream stream, C_Event event);
C_Status QueryEvent(const C_Device device, C_Event event);
C_Status DestroyEvent(const C_Device device, C_Event event);

bool isEnvEnable(std::string env_);

int getEnvVal(std::string env_);

inline size_t get_devices_count() {
  int count = 0;
  sdaaGetDeviceCount(&count);
  return static_cast<size_t>(count);
}
