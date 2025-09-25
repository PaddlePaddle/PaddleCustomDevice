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

#include "musa_device_handler.h"  // NOLINT

// #include "paddle/phi/core/enforce.h"

namespace musa {

namespace device {

int GetMusaDeviceCount() {
  static size_t count = 0;
  static std::once_flag call_flag;

  std::call_once(call_flag, []() {
    auto ret = GetMusaDeviceCountImpl(&count);
    if (ret != C_SUCCESS) {
      VLOG(2) << "Get musa device count failed, ret = " << ret;
    }
  });

  return static_cast<int>(count);
}

C_Status GetMusaDeviceCount(size_t *count) {
  return GetMusaDeviceCountImpl(count);
}

C_Status GetMusaDeviceCountImpl(size_t *count) {
  static int mu_count = 0;
  std::once_flag call_flag;

  std::call_once(call_flag, []() {
    int driverVersion = 0;
    MUSA_CHECK(musaDriverGetVersion(&driverVersion))
    if (driverVersion == 0) {
      VLOG(2) << "GPU Driver Version can't be detected. No GPU driver!";
      return C_FAILED;
    }

    const auto *musa_visible_devices = std::getenv("MUSA_VISIBLE_DEVICES");
    if (musa_visible_devices != nullptr) {
      std::string musa_visible_devices_str(musa_visible_devices);
      if (!musa_visible_devices_str.empty()) {
        musa_visible_devices_str.erase(
            0, musa_visible_devices_str.find_first_not_of('\''));
        musa_visible_devices_str.erase(
            musa_visible_devices_str.find_last_not_of('\'') + 1);
        musa_visible_devices_str.erase(
            0, musa_visible_devices_str.find_first_not_of('\"'));
        musa_visible_devices_str.erase(
            musa_visible_devices_str.find_last_not_of('\"') + 1);
      }
      if (std::all_of(musa_visible_devices_str.begin(),
                      musa_visible_devices_str.end(),
                      [](char ch) { return ch == ' '; })) {
        VLOG(2) << "MUSA_VISIBLE_DEVICES is set to be "
                   "empty. No GPU detected.";
        return C_ERROR;
      }
    }
    MUSA_CHECK(musaGetDeviceCount(&mu_count));
    return C_SUCCESS;
  });
  *count = mu_count;
  return C_SUCCESS;
}

C_Status GetMusaDevicesList(size_t *devices) {
  int count = musa::device::GetMusaDeviceCount();

  // TODO(jihong.zhong): it should be realloc devices ptr, check it alloc mem by
  // malloc or calloc before
  for (size_t i = 0; i < count; ++i) {
    devices[i] = i;
  }

  return C_SUCCESS;
}

C_Status InitMusaDevice(const C_Device device) {
  if (!device || device->id < 0) {
    return C_ERROR;
  }
  MUSA_CHECK(musaSetDevice(device->id))

  return C_SUCCESS;
}

C_Status SetMusaDevice(const C_Device device) {
  int id = device->id;
  // global_current_device = device->id;
  PADDLE_ENFORCE_LT(
      id,
      GetMusaDeviceCount(),
      phi::errors::InvalidArgument("Device id must be less than GPU count, "
                                   "but received id is: %d. GPU count is: %d.",
                                   id,
                                   GetMusaDeviceCount()));

  MUSA_CHECK(musaSetDevice(id))

  return C_SUCCESS;
}

C_Status GetMusaDevice(const C_Device device) {
  if (!device) {
    return C_ERROR;
  }
  int dev_id;
  MUSA_CHECK(musaGetDevice(&dev_id));

  device->id = dev_id;
  return C_SUCCESS;
}

C_Status MusaMemCpyH2D(const C_Device device,
                       void *dst,
                       const void *src,
                       size_t size) {
  if (0 == size) {
    return C_SUCCESS;
  }
  MUSA_CHECK(musaSetDevice(device->id))

  MUSA_CHECK(musaMemcpy(dst, src, size, musaMemcpyHostToDevice))

  return C_SUCCESS;
}

C_Status MusaMemCpyD2D(const C_Device device,
                       void *dst,
                       const void *src,
                       size_t size) {
  MUSA_CHECK(musaSetDevice(device->id))

  MUSA_CHECK(musaMemcpy(dst, src, size, musaMemcpyDeviceToDevice))

  return C_SUCCESS;
}

C_Status MusaMemCpyD2H(const C_Device device,
                       void *dst,
                       const void *src,
                       size_t size) {
  if (nullptr == device || dst == nullptr || src == nullptr || size <= 0) {
    VLOG(2) << "input params is error" << std::endl;
    return C_ERROR;
  }

  MUSA_CHECK(musaSetDevice(device->id))

  MUSA_CHECK(musaMemcpy(dst, src, size, musaMemcpyDeviceToHost))

  return C_SUCCESS;
}

C_Status MusaMemCpyP2P(const C_Device dst_device,
                       const C_Device src_device,
                       void *dst,
                       const void *src,
                       size_t size) {
  MUSA_CHECK(musaMemcpyPeer(dst, dst_device->id, src, src_device->id, size));

  return C_SUCCESS;
}

C_Status MusaMemAllocate(const C_Device device, void **ptr, size_t size) {
  MUSA_CHECK(musaSetDevice(device->id));

  MUSA_CHECK(musaMalloc(ptr, size));

  return C_SUCCESS;
}

C_Status MusaMemAllocateHost(const C_Device device, void **ptr, size_t size) {
  MUSA_CHECK(musaSetDevice(device->id))

  MUSA_CHECK(musaMallocHost(ptr, size))

  return C_SUCCESS;
}

C_Status MusaMemDeallocate(const C_Device device, void *ptr, size_t size) {
  MUSA_CHECK(musaSetDevice(device->id))

  MUSA_CHECK(musaFree(ptr))

  return C_SUCCESS;
}

C_Status MusaMemDeallocateHost(const C_Device device, void *ptr, size_t size) {
  MUSA_CHECK(musaSetDevice(device->id));

  MUSA_CHECK(musaFreeHost(ptr));

  return C_SUCCESS;
}

C_Status CreateMusaStream(const C_Device device, C_Stream *stream) {
  MUSA_CHECK(musaSetDevice(device->id))

  musaStream_t musa_stream{nullptr};
  MUSA_CHECK(musaStreamCreate(&musa_stream))

  *stream = (C_Stream)musa_stream;

  return C_SUCCESS;
}

C_Status DestroyMusaStream(const C_Device device, C_Stream stream) {
  MUSA_CHECK(musaSetDevice(device->id));

  musaStream_t musa_stream = (musaStream_t)stream;
  MUSA_CHECK(musaStreamDestroy(musa_stream));

  return C_SUCCESS;
}

C_Status CreateMusaEvent(const C_Device device, C_Event *event) {
  if (nullptr == device || nullptr == event) {
    return C_ERROR;
  }

  MUSA_CHECK(musaSetDevice(device->id));

  musaEvent_t evt;
  MUSA_CHECK(musaEventCreate(&evt));

  *event = NULL;
  *event = (C_Event)evt;

  return C_SUCCESS;
}

C_Status RecordMusaEvent(const C_Device device,
                         C_Stream stream,
                         C_Event event) {
  if (nullptr == device || nullptr == event) {
    return C_ERROR;
  }

  MUSA_CHECK(musaSetDevice(device->id));

  MUSA_CHECK(musaEventRecord(musaEvent_t(event), musaStream_t(stream)));

  return C_SUCCESS;
}

C_Status DestroyMusaEvent(const C_Device device, C_Event event) {
  if (nullptr == device || nullptr == event) {
    return C_ERROR;
  }

  MUSA_CHECK(musaSetDevice(device->id));

  MUSA_CHECK(musaEventDestroy(musaEvent_t(event)));

  return C_SUCCESS;
}

C_Status SyncMusaDevice(const C_Device device) {
  MUSA_CHECK(musaSetDevice(device->id));

  MUSA_CHECK(musaDeviceSynchronize());

  return C_SUCCESS;
}

C_Status SyncMusaStream(const C_Device device, C_Stream stream) {
  MUSA_CHECK(musaSetDevice(device->id))

  musaStream_t musa_stream = (musaStream_t)stream;
  MUSA_CHECK(musaStreamSynchronize(musa_stream))

  return C_SUCCESS;
}

C_Status SyncMusaEvent(const C_Device device, C_Event event) {
  if (nullptr == device || nullptr == event) {
    return C_ERROR;
  }

  MUSA_CHECK(musaSetDevice(device->id));

  MUSA_CHECK(musaEventSynchronize(musaEvent_t(event)));

  return C_SUCCESS;
}

C_Status MusaStreamWaitEvent(const C_Device device,
                             C_Stream stream,
                             C_Event event) {
  if (nullptr == device || nullptr == event) {
    return C_ERROR;
  }

  MUSA_CHECK(musaSetDevice(device->id));

  MUSA_CHECK(musaStreamWaitEvent(musaStream_t(stream), musaEvent_t(event), 0));

  return C_SUCCESS;
}

C_Status MusaMemCpyH2DAsync(const C_Device device,
                            C_Stream stream,
                            void *dst,
                            const void *src,
                            size_t size) {
  if (0 == size) {
    return C_SUCCESS;
  }
  if (nullptr == dst || nullptr == src) {
    return C_ERROR;
  }

  MUSA_CHECK(musaSetDevice(device->id));

  MUSA_CHECK(musaMemcpyAsync(dst, src, size, musaMemcpyHostToDevice));

  return C_SUCCESS;
}

C_Status MusaMemCpyD2HAsync(const C_Device device,
                            C_Stream stream,
                            void *dst,
                            const void *src,
                            size_t size) {
  if (0 == size) {
    return C_SUCCESS;
  }
  if (nullptr == dst || nullptr == src) {
    return C_ERROR;
  }

  MUSA_CHECK(musaSetDevice(device->id));

  MUSA_CHECK(musaMemcpyAsync(dst, src, size, musaMemcpyDeviceToHost));

  return C_SUCCESS;
}

C_Status MusaMemCpyD2DAsync(const C_Device device,
                            C_Stream stream,
                            void *dst,
                            const void *src,
                            size_t size) {
  if (0 == size) {
    VLOG(2) << "musa memcpy successful by zero size: " << dst << " " << src
            << " " << size;  // NOLINT
    return C_SUCCESS;
  }
  if (nullptr == dst || nullptr == src) {
    return C_ERROR;
  }

  MUSA_CHECK(musaSetDevice(device->id));

  MUSA_CHECK(musaMemcpyAsync(dst, src, size, musaMemcpyDeviceToDevice));
  return C_SUCCESS;
}

C_Status MusaMemCpyP2PAsync(const C_Device dst_device,
                            const C_Device src_device,
                            C_Stream stream,
                            void *dst,
                            const void *src,
                            size_t size) {
  MUSA_CHECK(musaMemcpyPeerAsync(
      dst, dst_device->id, src, src_device->id, size, musaStream_t(stream)));
  return C_SUCCESS;
}

C_Status GetMusaDeviceProperties(const C_Device device,
                                 void *device_properties) {
  static std::once_flag call_flag;
  static std::vector<musaDeviceProp> dev_props_vec;

  int dev_id = device->id;
  if (dev_id < 0) MUSA_CHECK(musaGetDevice(&dev_id));

  std::call_once(call_flag, [&] {
    auto dev_cnt = GetMusaDeviceCount();
    dev_props_vec.resize(dev_cnt);

    for (int i = 0; i < dev_cnt; ++i) {
      MUSA_CHECK(musaGetDeviceProperties(&dev_props_vec[dev_id], dev_id));
    }
  });

  const auto &src = dev_props_vec[dev_id];
  phi::DeviceProp dst;
  dst.name = src.name;
  dst.deviceMajor = src.major;
  dst.deviceMinor = src.minor;
  dst.totalGlobalMem = src.totalGlobalMem;
  dst.multiProcessorCount = src.multiProcessorCount;
  dst.isMultiGpuBoard = static_cast<bool>(src.isMultiGpuBoard);
  dst.integrated = static_cast<bool>(src.integrated);

  phi::DeviceProp *prop = static_cast<phi::DeviceProp *>(device_properties);
  *prop = std::move(dst);

  return C_SUCCESS;
}

C_Status GetMusaComputeCapability(const C_Device device, size_t *result) {
  int major_ver{0}, minor_ver{0};
  MUSA_CHECK(musaDeviceGetAttribute(
      &major_ver, musaDevAttrComputeCapabilityMajor, device->id));
  MUSA_CHECK(musaDeviceGetAttribute(
      &minor_ver, musaDevAttrComputeCapabilityMinor, device->id));
  *result = static_cast<int>(major_ver * 10 + minor_ver);
  return C_SUCCESS;
}

C_Status GetMusaDriverVersion(const C_Device device, size_t *version) {
  int driver_version{0};
  musaError_t status = musaDriverGetVersion(&driver_version);
  *version = driver_version;
  return C_SUCCESS;
}

C_Status GetMusaRuntimeVersion(const C_Device device, size_t *version) {
  int runtime_version{0};
  MUSA_CHECK(musaRuntimeGetVersion(&runtime_version));
  *version = static_cast<size_t>(runtime_version);
  return C_SUCCESS;
}

C_Status GetMusaMultiProcessorCount(const C_Device device, size_t *result) {
  int mp_cnt{0};
  MUSA_CHECK(musaDeviceGetAttribute(
      &mp_cnt, musaDevAttrMultiProcessorCount, device->id));
  *result = static_cast<size_t>(mp_cnt);
  return C_SUCCESS;
}

C_Status GetMusaMaxThreadsPerMP(const C_Device device, size_t *result) {
  int max_threads_per_mp{0};
  MUSA_CHECK(musaDeviceGetAttribute(
      &max_threads_per_mp, musaDevAttrMaxThreadsPerMultiProcessor, device->id));
  *result = static_cast<size_t>(max_threads_per_mp);
  return C_SUCCESS;
}

C_Status GetMusaMaxThreadsPerBlock(const C_Device device, size_t *result) {
  int max_threads_per_block{0};
  MUSA_CHECK(musaDeviceGetAttribute(
      &max_threads_per_block, musaDevAttrMaxThreadsPerBlock, device->id));
  *result = max_threads_per_block;
  return C_SUCCESS;
}

C_Status GetMusaMaxGridDimSize(const C_Device device,
                               std::array<unsigned int, 3> *result) {
  int x{0}, y{0}, z{0};
  MUSA_CHECK(musaDeviceGetAttribute(&x, musaDevAttrMaxGridDimX, device->id));
  MUSA_CHECK(musaDeviceGetAttribute(&y, musaDevAttrMaxGridDimY, device->id));
  MUSA_CHECK(musaDeviceGetAttribute(&z, musaDevAttrMaxGridDimZ, device->id));
  *result = {static_cast<unsigned int>(x),
             static_cast<unsigned int>(y),
             static_cast<unsigned int>(z)};
  return C_SUCCESS;
}
C_Status MusaInitEigenDevice(const C_Place place,
                             C_EigenDevice *eigen_device,
                             C_Stream stream,
                             C_Allocator allocator) {
  PADDLE_ENFORCE_NOT_NULL(
      allocator,
      common::errors::InvalidArgument(
          "The allocator for eigen device is nullptr. It must not be null."));

  phi::Place *phi_place = (phi::Place *)(place);
  musaStream_t musa_stream = (musaStream_t)stream;
  phi::Allocator *phi_allocator = (phi::Allocator *)allocator;
  EigenGpuStreamDevice *eigen_stream_ = new EigenGpuStreamDevice();

  eigen_stream_->Reinitialize(musa_stream, phi_allocator, *phi_place);

  Eigen::GpuDevice *eigen_device_ = new Eigen::GpuDevice(eigen_stream_);
  *eigen_device = reinterpret_cast<C_EigenDevice>(eigen_device_);

  return C_SUCCESS;
}

C_Status MusaDestroyEigenDevice(const C_Device device,
                                C_EigenDevice *eigen_device) {
  if (nullptr == eigen_device) {
    VLOG(2) << "eigen device is nullptr.";
    return C_ERROR;
  }

  Eigen::GpuDevice *gpu_device =
      reinterpret_cast<Eigen::GpuDevice *>(*eigen_device);

  delete gpu_device;
  *eigen_device = nullptr;

  VLOG(4) << "destroyed Eigen::GpuDevice. at musa backend";
  return C_SUCCESS;
}

}  // namespace device

}  // namespace musa
