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

#include <musa_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mutex>

#include "paddle/phi/backends/device_base.h"
#include "paddle/phi/backends/device_ext.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/allocator.h"
#include "runtime/utils.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace musa {
namespace device {

class EigenGpuStreamDevice : public Eigen::StreamInterface {
 public:
  EigenGpuStreamDevice() : scratch_(nullptr), semaphore_(nullptr) {
    Eigen::initializeDeviceProp();
  }
  ~EigenGpuStreamDevice() override = default;

  void Reinitialize(gpuStream_t gpu_stream,
                    phi::Allocator *allocator,
                    phi::GPUPlace place) {
    stream_ = gpu_stream;
    allocator_ = allocator;
    device_prop_ = &Eigen::m_deviceProperties[place.device];
  }

  const gpuStream_t &stream() const override { return stream_; }

  const gpuDeviceProp &deviceProperties() const override {
    return *device_prop_;
  }

  void *allocate(size_t num_bytes) const override {
    if (UNLIKELY(num_bytes == 0)) {
      return nullptr;
    }
    auto buf = allocator_->Allocate(num_bytes);
    // VLOG(4) << "Eigen allocated at " << buf->ptr() << " requested "
    //         << num_bytes;
    void *retv = buf->ptr();
    {
      std::lock_guard<std::mutex> lock(mtx_);
      allocations_.emplace(retv, std::move(buf));
    }
    return retv;
  }

  void deallocate(void *buffer) const override {
    if (LIKELY(buffer)) {
      std::lock_guard<std::mutex> lock(mtx_);
      allocations_.erase(buffer);
    }
  }

  void *scratchpad() const override {
    if (scratch_ == nullptr) {
      scratch_ = allocate(Eigen::kGpuScratchSize + sizeof(unsigned int));
    }
    return scratch_;
  }

  unsigned int *semaphore() const override {
    if (semaphore_ == nullptr) {
      char *scratch =
          static_cast<char *>(scratchpad()) + Eigen::kGpuScratchSize;
      semaphore_ = reinterpret_cast<unsigned int *>(scratch);
      MUSA_CHECK(musaMemsetAsync(semaphore_, 0, sizeof(unsigned int), stream_));
    }
    return semaphore_;
  }

 private:
  gpuStream_t stream_;                // not owned;
  phi::Allocator *allocator_;         // not owned;
  const gpuDeviceProp *device_prop_;  // not owned;
  mutable void *scratch_;
  mutable unsigned int *semaphore_;
  mutable std::mutex mtx_;  // to protect allocations_
  mutable std::unordered_map<void *, phi::Allocator::AllocationPtr>
      allocations_;
};

int GetMusaDeviceCount();
C_Status GetMusaDeviceCount(size_t *count);
C_Status GetMusaDeviceCountImpl(size_t *count);

C_Status InitMusaDevice(const C_Device device);
C_Status SetMusaDevice(const C_Device device);
C_Status GetMusaDevice(const C_Device device);
C_Status GetMusaDevicesList(size_t *devices);

C_Status MusaMemCpyH2D(const C_Device device,
                       void *dst,
                       const void *src,
                       size_t size);
C_Status MusaMemCpyD2D(const C_Device device,
                       void *dst,
                       const void *src,
                       size_t size);
C_Status MusaMemCpyD2H(const C_Device device,
                       void *dst,
                       const void *src,
                       size_t size);
C_Status MusaMemCpyP2P(const C_Device dst_device,
                       const C_Device src_device,
                       void *dst,
                       const void *src,
                       size_t size);
C_Status MusaMemCpyH2DAsync(const C_Device device,
                            C_Stream stream,
                            void *dst,
                            const void *src,
                            size_t size);
C_Status MusaMemCpyD2HAsync(const C_Device device,
                            C_Stream stream,
                            void *dst,
                            const void *src,
                            size_t size);
C_Status MusaMemCpyH2DAsync(const C_Device device,
                            C_Stream stream,
                            void *dst,
                            const void *src,
                            size_t size);
C_Status MusaMemCpyD2DAsync(const C_Device device,
                            C_Stream stream,
                            void *dst,
                            const void *src,
                            size_t size);
C_Status MusaMemCpyP2PAsync(const C_Device dst_device,
                            const C_Device src_device,
                            C_Stream stream,
                            void *dst,
                            const void *src,
                            size_t size);

C_Status MusaMemAllocate(const C_Device device, void **ptr, size_t size);
C_Status MusaMemAllocateHost(const C_Device device, void **ptr, size_t size);
C_Status MusaMemDeallocate(const C_Device device, void *ptr, size_t size);
C_Status MusaMemDeallocateHost(const C_Device device, void *ptr, size_t size);
C_Status CreateMusaStream(const C_Device device, C_Stream *stream);
C_Status DestroyMusaStream(const C_Device device, C_Stream stream);
C_Status CreateMusaEvent(const C_Device device, C_Event *event);
C_Status RecordMusaEvent(const C_Device device, C_Stream stream, C_Event event);
C_Status DestroyMusaEvent(const C_Device device, C_Event event);
C_Status SyncMusaDevice(const C_Device device);
C_Status SyncMusaStream(const C_Device device, C_Stream stream);
C_Status SyncMusaEvent(const C_Device device, C_Event event);
C_Status MusaStreamWaitEvent(const C_Device device,
                             C_Stream stream,
                             C_Event event);

C_Status GetMusaDeviceProperties(const C_Device device,
                                 void *device_properties);
C_Status GetMusaRuntimeVersion(const C_Device device, size_t *version);
C_Status GetMusaDriverVersion(const C_Device device, size_t *version);
C_Status GetMusaMultiProcessorCount(const C_Device device, size_t *result);
C_Status GetMusaMaxThreadsPerMP(const C_Device device, size_t *result);
C_Status GetMusaMaxThreadsPerBlock(const C_Device device, size_t *result);
C_Status GetMusaMaxGridDimSize(const C_Device device,
                               std::array<unsigned int, 3> *result);
C_Status GetMusaComputeCapability(const C_Device device, size_t *result);

C_Status MusaInitEigenDevice(const C_Place place,
                             C_EigenDevice *eigen_device,
                             C_Stream stream,
                             C_Allocator allocator);
C_Status MusaDestroyEigenDevice(const C_Device device,
                                C_EigenDevice *eigen_device);

}  // namespace device

}  // namespace musa
