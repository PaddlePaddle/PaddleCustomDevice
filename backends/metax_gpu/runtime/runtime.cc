// 2024 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights
// Reserved. Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include <cuda_runtime.h>
#include <cupti.h>
#include <errno.h>
#include <fcntl.h>
#include <nccl.h>
#include <semaphore.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "../cinn/cinn_interface.h"
#include "glog/logging.h"
#include "paddle/fluid/platform/profiler/cuda_tracer.h"
#include "paddle/fluid/platform/profiler/cupti_data_process.h"
#include "paddle/phi/api/profiler/trace_event_collector.h"
#include "paddle/phi/backends/custom/cuda_graph.h"
#include "paddle/phi/backends/device_base.h"
#include "paddle/phi/backends/device_ext.h"
#include "paddle/phi/backends/dynload/cublasLt.h"
#include "paddle/phi/backends/dynload/cupti.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/allocator.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/os_info.cc"  //NOLINT
#include "paddle/phi/core/os_info.h"
#include "paddle/phi/core/platform/device/gpu/gpu_info.cc"  //NOLINT
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"
#include "paddle/phi/core/platform/profiler/utils.cc"  //NOLINT
#include "paddle/phi/core/platform/profiler/utils.h"
#include "passes/pattern_passes.h"
#include "runtime/process_cupti_data.cc"  //NOLINT
#include "unsupported/Eigen/CXX11/Tensor"
#define MEMORY_FRACTION 0.5f

static int global_current_device = 0;

const char *const DeviceType = "metax_gpu";
const char *const SubDeviceType = "v0.1";

#ifdef WITH_CINN
namespace paddle {
namespace custom_device {
namespace metax {
void InitCinnInterface(C_DeviceInterface *interface);
}
}  // namespace custom_device
}  // namespace paddle
#endif

namespace phi {

namespace internal {
inline ncclDataType_t PDDataTypeToNcclDataType(C_DataType type) {
  if (type == C_DataType::FLOAT32) {
    return ncclFloat32;
  } else if (type == C_DataType::BFLOAT16) {
    return ncclBfloat16;
  } else if (type == C_DataType::UINT8 || type == C_DataType::BOOL) {
    return ncclUint8;
  } else if (type == C_DataType::UINT32) {
    return ncclUint32;
  } else if (type == C_DataType::UINT64) {
    return ncclUint64;
  } else if (type == C_DataType::INT8) {
    return ncclInt8;
  } else if (type == C_DataType::INT32) {
    return ncclInt32;
  } else if (type == C_DataType::INT64) {
    return ncclInt64;
  } else if (type == C_DataType::FLOAT16) {
    return ncclFloat16;
  } else if (type == C_DataType::FLOAT64) {
    return ncclFloat64;
  } else {
    LOG(ERROR) << "Datatype " << type << " in nccl is not supported.";
  }
  return ncclFloat32;
}

#define NCCL_CHECK(cmd)                                                        \
  do {                                                                         \
    ncclResult_t r = cmd;                                                      \
    if (r != ncclSuccess) {                                                    \
      PADDLE_THROW(common::errors::External("Failed, NCCL error %s:%d '%s'\n", \
                                            __FILE__,                          \
                                            __LINE__,                          \
                                            ncclGetErrorString(r)));           \
    }                                                                          \
  } while (0)

#define MCCL_CHECK(cmd)                                                        \
  do {                                                                         \
    mcclResult_t r = cmd;                                                      \
    if (r != ncclSuccess) {                                                    \
      PADDLE_THROW(common::errors::External("Failed, MCCL error %s:%d '%s'\n", \
                                            __FILE__,                          \
                                            __LINE__,                          \
                                            mcclGetErrorString(r)));           \
    }                                                                          \
  } while (0)

class EigenGpuStreamDevice : public Eigen::StreamInterface {
 public:
  EigenGpuStreamDevice()
      : stream_(nullptr),
        allocator_(nullptr),
        device_prop_(nullptr),
        scratch_(nullptr),
        semaphore_(nullptr),
        allocations_() {
    Eigen::GetGpuDeviceProperties();
    // Eigen::initializeDeviceProp();
  }
  ~EigenGpuStreamDevice() override = default;

  void Reinitialize(cudaStream_t cuda_stream,
                    Allocator *allocator,
                    CustomPlace place) {
    stream_ = cuda_stream;
    place_ = place;
    allocator_ = allocator;
    // device_prop_ = &Eigen::m_deviceProperties[place.device];
    device_prop_ = &Eigen::GetGpuDeviceProperties(place.device);
  }

  const cudaStream_t &stream() const override { return stream_; }

  const gpuDeviceProp &deviceProperties() const override {
    return *device_prop_;
  }

  void *allocate(size_t num_bytes) const override {
    if (UNLIKELY(num_bytes == 0)) {
      return nullptr;
    }
    auto buf = allocator_->Allocate(num_bytes);
    VLOG(4) << "Eigen allocated at " << buf->ptr() << " requested "
            << num_bytes;
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
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(
          hipMemsetAsync(semaphore_, 0, sizeof(unsigned int), stream()));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaMemsetAsync(semaphore_, 0, sizeof(unsigned int), stream()));
#endif
    }
    return semaphore_;
  }

 private:
  CustomPlace place_;
  cudaStream_t stream_;               // not owned;
  Allocator *allocator_;              // not owned;
  const gpuDeviceProp *device_prop_;  // not owned;
  mutable void *scratch_;
  mutable unsigned int *semaphore_;
  mutable std::mutex mtx_;  // to protect allocations_
  mutable std::unordered_map<void *, Allocator::AllocationPtr> allocations_;
};

}  // namespace internal
}  // namespace phi

C_Status InitEigenDevice(const C_Place place,
                         C_EigenDevice *eigen_device,
                         C_Stream stream,
                         C_Allocator allocator) {
  cudaStream_t stream_t = (cudaStream_t)stream;
  phi::Allocator *allocator_t = (phi::Allocator *)allocator;
  phi::Place *place_t = (phi::Place *)(place);
  VLOG(4) << "allocator: " << allocator;
  VLOG(4) << "allocator is nullptr " << (allocator == nullptr);
  VLOG(4) << "stream: " << stream;
  VLOG(4) << "stream is nullptr " << (stream == nullptr);
  VLOG(4) << "place is nullptr " << (place == nullptr);
  PADDLE_ENFORCE_NOT_NULL(
      allocator,
      common::errors::InvalidArgument(
          "The allocator for eigen device is nullptr. It must not be null."));
  phi::internal::EigenGpuStreamDevice *eigen_stream_ =
      new phi::internal::EigenGpuStreamDevice();
  eigen_stream_->Reinitialize(stream_t, allocator_t, *place_t);
  Eigen::GpuDevice *eigen_device_ = new Eigen::GpuDevice(eigen_stream_);
  *eigen_device = reinterpret_cast<C_EigenDevice>(eigen_device_);
  VLOG(4) << "eigen_device:" << eigen_device;
  return C_SUCCESS;
}

C_Status DestroyEigenDevice(const C_Device device,
                            C_EigenDevice *eigen_device) {
  if (eigen_device == nullptr) {
    VLOG(4) << "Invalid eigen_device pointer (nullptr).";
    return C_ERROR;
  }

  Eigen::GpuDevice *gpu_device =
      reinterpret_cast<Eigen::GpuDevice *>(*eigen_device);

  delete gpu_device;

  *eigen_device = nullptr;

  VLOG(4) << "destroyed Eigen::GpuDevice.";
  return C_SUCCESS;
}

C_Status Init() {
  std::cout << "matex_gpu plugin";
  return C_SUCCESS;
}

C_Status GetComputeCapability(const C_Device device,
                              size_t *compute_capability) {
  int id = device->id;
  int major, minor;
  auto major_error_code =
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, id);
  auto minor_error_code =
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, id);

  *compute_capability = major * 10 + minor;
  VLOG(4) << "CUDA Device " << id
          << " compute capability: " << *compute_capability;
  return C_SUCCESS;
}

C_Status GetDevicesCount(size_t *count) {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  // *count = (size_t)device_count;
  *count = static_cast<size_t>(device_count);
  return C_SUCCESS;
}

C_Status GetDevicesList(size_t *devices) {
  size_t count = 0;
  C_Status status = GetDevicesCount(&count);
  if (status != C_SUCCESS) {
    return status;
  }
  for (size_t i = 0; i < count; ++i) {
    devices[i] = i;
  }
  return C_SUCCESS;
}

static std::once_flag g_device_props_size_init_flag;
static std::vector<std::unique_ptr<std::once_flag>> g_device_props_init_flags;
static std::vector<cudaDeviceProp> g_device_props;
static std::vector<cudaError_t> g_device_props_init_errors;

C_Status GetDeviceProperties(const C_Device device, void *device_properties) {
  int id = device->id;
  C_Status init_status = C_SUCCESS;
  if (id == -1) {
    cudaGetDevice(&id);
  }

  std::call_once(g_device_props_size_init_flag, [&] {
    size_t count = 0;
    init_status = GetDevicesCount(&count);
    int gpu_num = count;

    g_device_props_init_flags.resize(gpu_num);
    g_device_props.resize(gpu_num);
    g_device_props_init_errors.resize(gpu_num, cudaSuccess);

    for (int i = 0; i < gpu_num; ++i) {
      g_device_props_init_flags[i] = std::make_unique<std::once_flag>();
    }
  });

  if (init_status != C_SUCCESS) {
    VLOG(0) << "GetDevicesCount failed: " << init_status;
    return C_ERROR;
  }

  if (id < 0 || id >= static_cast<int>(g_device_props.size())) {
    VLOG(0) << "device id: " << id << " out of range";
    return C_ERROR;
  }

  std::call_once(*(g_device_props_init_flags[id]), [&] {
    cudaError_t ret = cudaGetDeviceProperties(&g_device_props[id], id);
    g_device_props_init_errors[id] = ret;
  });

  if (g_device_props_init_errors[id] != cudaSuccess) {
    return C_ERROR;
  }

  phi::DeviceProp *prop = static_cast<phi::DeviceProp *>(device_properties);
  const cudaDeviceProp &src = g_device_props[id];

  using DeviceProp = phi::DeviceProp;
  prop->~DeviceProp();
  new (prop) DeviceProp();

  prop->name = src.name;
  prop->deviceMajor = src.major;
  prop->deviceMinor = src.minor;
  prop->totalGlobalMem = src.totalGlobalMem;
  prop->multiProcessorCount = src.multiProcessorCount;
  prop->isMultiGpuBoard = src.isMultiGpuBoard;
  prop->integrated = (src.integrated != 0);

  return C_SUCCESS;
}

C_Status GetRuntimeVersion(const C_Device device, size_t *version) {
  int runtime_version = 0;
  cudaError_t status = cudaRuntimeGetVersion(&runtime_version);
  *version = runtime_version;
  return C_SUCCESS;
}

C_Status GetDriverVersion(const C_Device device, size_t *version) {
  int driver_version = 0;
  cudaError_t status = cudaDriverGetVersion(&driver_version);
  *version = driver_version;
  return C_SUCCESS;
}

C_Status GetMultiProcessors(const C_Device device, size_t *multi_process) {
  int id = device->id;
  int count = 0;
  cudaError_t status =
      cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, id);
  *multi_process = count;
  return C_SUCCESS;
}

C_Status GetMaxThreadsPerMultiProcessor(const C_Device device,
                                        size_t *threads_per_mp) {
  int id = device->id;
  int count = 0;
  cudaError_t status = cudaDeviceGetAttribute(
      &count, cudaDevAttrMaxThreadsPerMultiProcessor, id);
  *threads_per_mp = count;
  return C_SUCCESS;
}

C_Status GetMaxThreadsPerBlock(const C_Device device,
                               size_t *threads_per_block) {
  int id = device->id;
  int count = 0;
  cudaError_t status =
      cudaDeviceGetAttribute(&count, cudaDevAttrMaxThreadsPerBlock, id);
  *threads_per_block = count;
  return C_SUCCESS;
}

C_Status GetMaxSharedMemPerBlock(const C_Device device,
                                 size_t *shared_mem_per_block) {
  int id = device->id;
  int count = 0;
  cudaError_t status =
      cudaDeviceGetAttribute(&count, cudaDevAttrMaxSharedMemoryPerBlock, id);
  *shared_mem_per_block = 65534;
  return C_SUCCESS;
}

C_Status GetWarpSize(const C_Device device, size_t *warp_size) {
  int id = device->id;
  int size = 0;
  cudaError_t status = cudaDeviceGetAttribute(&size, cudaDevAttrWarpSize, id);
  *warp_size = size;
  return C_SUCCESS;
}

C_Status GetMaxRegistersPerMultiProcessor(const C_Device device,
                                          size_t *registers_per_mp) {
  int id = device->id;
  int count = 0;
  cudaError_t status = cudaDeviceGetAttribute(
      &count, cudaDevAttrMaxRegistersPerMultiprocessor, id);
  *registers_per_mp = count;
  return C_SUCCESS;
}

C_Status GetPreferredVectorWidth(const C_Device device,
                                 size_t *vector_alignment) {
  int id = device->id;
  // int count = 0;
  // cudaError_t status =
  //     cudaDeviceGetAttribute(&count, cudaDevAttrMaxSharedMemoryPerBlock, id);
  // *vector_alignment = count;
  *vector_alignment = 128;
  return C_SUCCESS;
}

C_Status GetMaxBlocksPerMultiProcessor(const C_Device device,
                                       size_t *blocks_per_mp) {
  int id = device->id;
  int count = 0;
  cudaError_t status =
      cudaDeviceGetAttribute(&count, cudaDevAttrMaxBlocksPerMultiprocessor, id);
  *blocks_per_mp = count;
  return C_SUCCESS;
}

C_Status GetMaxGridDimSize(const C_Device device,
                           std::array<unsigned int, 3> *grid_dim_size) {
  int id = device->id;
  std::array<unsigned int, 3> ret = {};
  int size;
  auto error_code_x = cudaDeviceGetAttribute(&size, cudaDevAttrMaxGridDimX, id);
  ret[0] = size;
  auto error_code_y = cudaDeviceGetAttribute(&size, cudaDevAttrMaxGridDimY, id);
  ret[1] = size;
  auto error_code_z = cudaDeviceGetAttribute(&size, cudaDevAttrMaxGridDimZ, id);
  ret[2] = size;

  *grid_dim_size = ret;
  return C_SUCCESS;
}

C_Status GetMaxBlockDimSize(const C_Device device,
                            std::array<unsigned int, 3> *block_dim_size) {
  int id = device->id;
  std::array<unsigned int, 3> ret = {};
  int size;
  auto error_code_x =
      cudaDeviceGetAttribute(&size, cudaDevAttrMaxBlockDimX, id);
  ret[0] = size;
  auto error_code_y =
      cudaDeviceGetAttribute(&size, cudaDevAttrMaxBlockDimY, id);
  ret[1] = size;
  auto error_code_z =
      cudaDeviceGetAttribute(&size, cudaDevAttrMaxBlockDimZ, id);
  ret[2] = size;

  *block_dim_size = ret;
  return C_SUCCESS;
}

C_Status InitDevice(const C_Device device) {
  if (!device || device->id < 0) {
    return C_ERROR;
  }

  cudaError_t err;

  if ((err = cudaSetDevice(device->id)) != cudaSuccess) {
    return C_ERROR;
  }

  return C_SUCCESS;
}

C_Status SetDevice(const C_Device device) {
  if (device == nullptr) {
    return C_ERROR;
  }
  cudaError_t err = cudaSetDevice(device->id);
  return (err == cudaSuccess) ? C_SUCCESS : C_ERROR;
}

C_Status GetDevice(const C_Device device) {
  if (!device) {
    return C_ERROR;
  }

  cudaError_t err;
  int dev_id;

  if ((err = cudaGetDevice(&dev_id)) != cudaSuccess) {
    return C_ERROR;
  }

  device->id = dev_id;
  return C_SUCCESS;
}

C_Status DestroyDevice(const C_Device device) {
  if (device == NULL) {
    return C_ERROR;
  }

  return C_SUCCESS;
}

C_Status Finalize() { return C_SUCCESS; }

C_Status MemCpyH2D(const C_Device device,
                   void *dst,
                   const void *src,
                   size_t size) {
  VLOG(2) << "MemCpyH2D: " << dst << " " << src << " " << size;
  if (size == 0) {
    return C_SUCCESS;
  }
  cudaError_t cudaErr = cudaSetDevice(device->id);
  if (cudaErr != cudaSuccess) {
    VLOG(2) << "Failed to set device: " << device->id
            << ", Error: " << cudaGetErrorString(cudaErr);
    return C_ERROR;
  }
  VLOG(2) << "setdevice: " << device->id;
  cudaErr = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
  if (cudaErr != cudaSuccess) {
    VLOG(2) << "cudaMemcpy failed: " << cudaGetErrorString(cudaErr);
    return C_ERROR;
  }
  VLOG(2) << "cudamemcpy successful: " << dst << " " << src << " "
          << size;  // NOLINT
  return C_SUCCESS;
}

C_Status MemCpyD2D(const C_Device device,
                   void *dst,
                   const void *src,
                   size_t size) {
  cudaError_t err;

  err = cudaSetDevice(device->id);
  if (err != cudaSuccess) {
    return C_ERROR;
  }

  err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);

  if (err == cudaSuccess) {
    VLOG(2) << "cudamemcpy successful: " << dst << " " << src << " "
            << size;  // NOLINT
    return C_SUCCESS;
  } else {
    return C_ERROR;
  }
}

C_Status MemCpyD2H(const C_Device device,
                   void *dst,
                   const void *src,
                   size_t size) {
  if (device == NULL || dst == NULL || src == NULL || size == 0) {
    return C_ERROR;
  }

  cudaError_t cudaErr;

  cudaErr = cudaSetDevice(device->id);
  if (cudaErr != cudaSuccess) {
    return C_ERROR;
  }

  cudaErr = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);

  if (cudaErr != cudaSuccess) {
    return C_ERROR;
  }
  VLOG(2) << "cudamemcpy successful: " << dst << " " << src << " "
          << size;  // NOLINT
  return C_SUCCESS;
}

C_Status MemCpy(const C_Device device,
                void *dst,
                const void *src,
                size_t size) {
  return C_ERROR;
}

C_Status AsyncMemCpyH2D(const C_Device device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size) {
  if (size == 0) {
    return C_SUCCESS;
  }

  if (dst == NULL || src == NULL) {
    return C_ERROR;
  }
  cudaError_t cudaErr = cudaSetDevice(device->id);
  if (cudaErr != cudaSuccess) {
    return C_ERROR;
  }

  cudaErr = cudaMemcpyAsync(
      dst, src, size, cudaMemcpyHostToDevice, (cudaStream_t)stream);
  if (cudaErr != cudaSuccess) {
    return C_ERROR;
  }

  return C_SUCCESS;
}

C_Status AsyncMemCpyD2H(const C_Device device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size) {
  if (size == 0) {
    return C_SUCCESS;
  }

  if (dst == NULL || src == NULL) {
    return C_ERROR;
  }

  cudaError_t cudaErr = cudaSetDevice(device->id);
  if (cudaErr != cudaSuccess) {
    return C_ERROR;
  }

  cudaErr = cudaMemcpyAsync(
      dst, src, size, cudaMemcpyDeviceToHost, (cudaStream_t)stream);
  if (cudaErr != cudaSuccess) {
    return C_ERROR;
  }

  return C_SUCCESS;
}

C_Status AsyncMemCpyD2D(const C_Device device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size) {
  if (size == 0) {
    VLOG(2) << "cudamemcpy successful: " << dst << " " << src << " "
            << size;  // NOLINT
    return C_SUCCESS;
  }

  if (dst == NULL || src == NULL) {
    return C_ERROR;
  }

  cudaError_t cudaErr = cudaSetDevice(device->id);
  if (cudaErr != cudaSuccess) {
    return C_ERROR;
  }

  cudaErr = cudaMemcpyAsync(
      dst, src, size, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
  if (cudaErr != cudaSuccess) {
    return C_ERROR;
  }
  VLOG(2) << "cudamemcpy successful: " << dst << " " << src << " "
          << size;  // NOLINT
  return C_SUCCESS;
}

C_Status MemCpyP2P(const C_Device dst_device,
                   const C_Device src_device,
                   void *dst,
                   const void *src,
                   size_t size) {
  return C_ERROR;
}

C_Status AsyncMemCpyP2P(const C_Device dst_device,
                        const C_Device src_device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size) {
  return C_ERROR;
}

C_Status Allocate(const C_Device device, void **ptr, size_t size) {
  cudaError_t err;
  *ptr = NULL;

  err = cudaSetDevice(device->id);
  if (err != cudaSuccess) {
    return C_ERROR;
  }

  err = cudaMalloc(ptr, size);
  if (err != cudaSuccess) {
    *ptr = NULL;
    return C_ERROR;
  }

  return C_SUCCESS;
}
C_Status AllocateHost(const C_Device device, void **ptr, size_t size) {
  cudaError_t err;
  *ptr = NULL;

  err = cudaSetDevice(device->id);
  if (err != cudaSuccess) {
    return C_ERROR;
  }

  err = cudaMallocHost(ptr, size);
  if (err != cudaSuccess) {
    *ptr = NULL;
    return C_ERROR;
  }

  return C_SUCCESS;
}

C_Status Deallocate(const C_Device device, void *ptr, size_t size) {
  cudaSetDevice(device->id);
  cudaFree(ptr);
  return C_SUCCESS;
}
C_Status DeallocateHost(const C_Device device, void *ptr, size_t size) {
  cudaSetDevice(device->id);
  cudaFreeHost(ptr);
  return C_SUCCESS;
}

C_Status CreateStream(const C_Device device, C_Stream *stream) {
  cudaError_t err;
  cudaStream_t cuda_stream = NULL;

  err = cudaSetDevice(device->id);
  if (err != cudaSuccess) {
    return C_ERROR;
  }

  err = cudaStreamCreate(&cuda_stream);
  if (err != cudaSuccess) {
    return C_ERROR;
  }

  *stream = (C_Stream)cuda_stream;
  return C_SUCCESS;
}

C_Status DestroyStream(const C_Device device, C_Stream stream) {
  cudaError_t err;
  cudaStream_t cuda_stream = (cudaStream_t)stream;

  err = cudaSetDevice(device->id);
  if (err != cudaSuccess) {
    return C_ERROR;
  }

  err = cudaStreamDestroy(cuda_stream);

  if (err != cudaSuccess) {
    return C_ERROR;
  }

  return C_SUCCESS;
}

C_Status CreateEvent(const C_Device device, C_Event *event) {
  if (device == NULL || event == NULL) {
    return C_ERROR;
  }

  *event = NULL;

  cudaError_t cuda_status;

  cuda_status = cudaSetDevice(device->id);
  if (cuda_status != cudaSuccess) {
    return C_ERROR;
  }

  cudaEvent_t evt;
  cuda_status = cudaEventCreate(&evt);
  if (cuda_status != cudaSuccess) {
    return C_ERROR;
  }

  *event = (C_Event)evt;
  return C_SUCCESS;
}

C_Status RecordEvent(const C_Device device, C_Stream stream, C_Event event) {
  if (device == NULL || event == NULL) {
    return C_ERROR;
  }

  cudaError_t cuda_status;

  cuda_status = cudaSetDevice(device->id);
  if (cuda_status != cudaSuccess) {
    return C_ERROR;
  }

  cuda_status = cudaEventRecord(cudaEvent_t(event), cudaStream_t(stream));
  if (cuda_status != cudaSuccess) {
    return C_ERROR;
  }

  return C_SUCCESS;
}

C_Status DestroyEvent(const C_Device device, C_Event event) {
  if (device == NULL || event == NULL) {
    return C_ERROR;
  }

  cudaError_t cuda_status;

  cuda_status = cudaSetDevice(device->id);
  if (cuda_status != cudaSuccess) {
    return C_ERROR;
  }

  cuda_status = cudaEventDestroy(cudaEvent_t(event));
  if (cuda_status != cudaSuccess) {
    return C_ERROR;
  }

  return C_SUCCESS;
}

C_Status SyncDevice(const C_Device device) {
  cudaError_t err;

  err = cudaSetDevice(device->id);
  if (err != cudaSuccess) {
    return C_ERROR;
  }

  err = cudaDeviceSynchronize();
  cudaError_t sync_err = err;

  if (sync_err != cudaSuccess) {
    return C_ERROR;
  }

  return C_SUCCESS;
}

C_Status SyncStream(const C_Device device, C_Stream stream) {
  cudaError_t err;
  cudaStream_t cuda_stream = (cudaStream_t)stream;

  err = cudaSetDevice(device->id);
  if (err != cudaSuccess) {
    return C_ERROR;
  }

  err = cudaStreamSynchronize(cuda_stream);

  if (err != cudaSuccess) {
    return C_ERROR;
  }

  return C_SUCCESS;
}

C_Status SyncEvent(const C_Device device, C_Event event) {
  if (device == NULL || event == NULL) {
    return C_ERROR;
  }

  cudaError_t cuda_status;

  cuda_status = cudaSetDevice(device->id);
  if (cuda_status != cudaSuccess) {
    return C_ERROR;
  }
  cuda_status = cudaEventSynchronize(cudaEvent_t(event));
  if (cuda_status != cudaSuccess) {
    return C_ERROR;
  }

  return C_SUCCESS;
}

C_Status StreamWaitEvent(const C_Device device,
                         C_Stream stream,
                         C_Event event) {
  if (device == NULL || event == NULL) {
    return C_ERROR;
  }

  cudaError_t cuda_status;

  cuda_status = cudaSetDevice(device->id);
  if (cuda_status != cudaSuccess) {
    return C_ERROR;
  }

  cuda_status =
      cudaStreamWaitEvent(cudaStream_t(stream), cudaEvent_t(event), 0);
  if (cuda_status != cudaSuccess) {
    return C_ERROR;
  }

  if (cuda_status != cudaSuccess) {
    return C_ERROR;
  }

  return C_SUCCESS;
}

C_Status DeviceMinChunkSize(const C_Device device, size_t *size) {
  VLOG(10) << "Runtime: GPU min chunk size is " << (1 << 8);
  *size = 1 << 8;
  return C_SUCCESS;
}

C_Status DeviceMaxChunkSize(const C_Device device, size_t *size) {
  return C_ERROR;
}
ncclRedOp_t PDReduceOpToNcclReduceOp(C_CCLReduceOp op) {
  if (op == C_CCLReduceOp::MIN) {
    return ncclMin;
  } else if (op == C_CCLReduceOp::MAX) {
    return ncclMax;
  } else if (op == C_CCLReduceOp::SUM) {
    return ncclSum;
  } else if (op == C_CCLReduceOp::PRODUCT) {
    return ncclProd;
  } else if (op == C_CCLReduceOp::AVG) {
    return ncclAvg;
  } else {
    LOG(ERROR) << "Reduceop " << op << " in nccl is not supported.";
  }
}

void BufferRequestedCallback(uint8_t **buffer,
                             size_t *size,
                             size_t *max_num_records) {
  Tracer::Instance().AllocateBuffer(buffer, size);
  *max_num_records = 0;
}

void BufferCompletedCallback(CUcontext ctx,
                             uint32_t stream_id,
                             uint8_t *buffer,
                             size_t size,
                             size_t valid_size) {
  Tracer::Instance().ProduceBuffer(buffer, valid_size);
  size_t dropped = 0;
  CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, stream_id, &dropped));
  if (dropped != 0) {
    LOG(WARNING) << "Stream " << stream_id << " Dropped " << dropped
                 << " activity records";
  }
}

int ProcessCuptiActivity(C_Profiler prof, uint64_t tracing_start_ns_) {
  int record_cnt = 0;
  CUPTI_CALL(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
  auto mapping = details::CreateThreadIdMapping();
  std::vector<ActivityBuffer> buffers = Tracer::Instance().ConsumeBuffers();
  for (auto &buffer : buffers) {
    if (buffer.addr == nullptr || buffer.valid_size == 0) {
      continue;
    }
    CUpti_Activity *record = nullptr;
    while (true) {
      CUptiResult status =
          cuptiActivityGetNextRecord(buffer.addr, buffer.valid_size, &record);
      if (status == CUPTI_SUCCESS) {
        ProcessCuptiActivityRecord(record, tracing_start_ns_, mapping, prof);
        ++record_cnt;
      } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
        break;
      } else {
        CUPTI_CALL(status);
      }
    }

    Tracer::Instance().ReleaseBuffer(buffer.addr);
    // ReleaseBuffer(buffer.addr);
  }
  return record_cnt;
}

C_Status ProfilerInitialize(C_Profiler prof, void **user_data) {
  return C_SUCCESS;
}

C_Status ProfilerFinalize(C_Profiler prof, void *user_data) {
  // CUPTI_CALL(cuptiRelease());
  return C_SUCCESS;
}

C_Status ProfilerPrepare(C_Profiler prof, void *user_data) {
  CUPTI_CALL(cuptiActivityRegisterCallbacks(BufferRequestedCallback,
                                            BufferCompletedCallback));

  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
  VLOG(3) << "enable cupti activity";
  return C_SUCCESS;
}

C_Status ProfilerStart(C_Profiler prof, void *user_data) {
  Tracer::Instance().ConsumeBuffers();
  return C_SUCCESS;
}

C_Status ProfilerStop(C_Profiler prof, void *user_data) {
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DRIVER));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMSET));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME));
  VLOG(3) << "disable cupti activity";
  return C_SUCCESS;
}

C_Status ProfilerCollectData(C_Profiler prof,
                             uint64_t tracing_start_ns_,
                             void *user_data) {
  ProcessCuptiActivity(prof, tracing_start_ns_);
  return C_SUCCESS;
}

C_Status XcclGetUniqueIdSize(size_t *size) {
  *size = sizeof(ncclUniqueId);
  return C_SUCCESS;
}

C_Status XcclGetUniqueId(C_CCLRootId *unique_id) {
  if (unique_id->sz != sizeof(ncclUniqueId)) {
    LOG(ERROR) << "unique_id->sz must be equal sizeof(ncclUniqueId)";
    return C_FAILED;
  }
  NCCL_CHECK(
      ncclGetUniqueId(reinterpret_cast<ncclUniqueId *>(unique_id->data)));

  return C_SUCCESS;
}

C_Status XcclCommInitRank(size_t nranks,
                          C_CCLRootId *unique_id,
                          size_t rank,
                          C_CCLComm *comm) {
  NCCL_CHECK(
      ncclCommInitRank(reinterpret_cast<ncclComm_t *>(comm),
                       nranks,
                       *(reinterpret_cast<ncclUniqueId *>(unique_id->data)),
                       rank));
  VLOG(4) << "[NCCL] comm inited: " << reinterpret_cast<ncclComm_t>(*comm);
  return C_SUCCESS;
}

C_Status XcclDestroyComm(C_CCLComm comm) {
  NCCL_CHECK(ncclCommDestroy(reinterpret_cast<ncclComm_t>(comm)));
  return C_SUCCESS;
}

C_Status XcclAllReduce(void *send_buf,
                       void *recv_buf,
                       size_t count,
                       C_DataType data_type,
                       C_CCLReduceOp op,
                       C_CCLComm comm,
                       C_Stream stream) {
  NCCL_CHECK(ncclAllReduce(send_buf,
                           recv_buf,
                           count,
                           phi::internal::PDDataTypeToNcclDataType(data_type),
                           PDReduceOpToNcclReduceOp(op),
                           reinterpret_cast<ncclComm_t>(comm),
                           reinterpret_cast<cudaStream_t>(stream)));
  return C_SUCCESS;
}

C_Status XcclBroadcast(void *buf,
                       size_t count,
                       C_DataType data_type,
                       size_t root,
                       C_CCLComm comm,
                       C_Stream stream) {
  NCCL_CHECK(ncclBroadcast(static_cast<const void *>(buf),
                           buf,
                           count,
                           phi::internal::PDDataTypeToNcclDataType(data_type),
                           root,
                           reinterpret_cast<ncclComm_t>(comm),
                           reinterpret_cast<cudaStream_t>(stream)));
  return C_SUCCESS;
}

C_Status XcclReduce(void *send_buf,
                    void *recv_buf,
                    size_t count,
                    C_DataType data_type,
                    C_CCLReduceOp op,
                    size_t root,
                    C_CCLComm comm,
                    C_Stream stream) {
  NCCL_CHECK(ncclReduce(send_buf,
                        recv_buf,
                        count,
                        phi::internal::PDDataTypeToNcclDataType(data_type),
                        PDReduceOpToNcclReduceOp(op),
                        root,
                        reinterpret_cast<ncclComm_t>(comm),
                        reinterpret_cast<cudaStream_t>(stream)));
  return C_SUCCESS;
}

C_Status XcclAllGather(void *send_buf,
                       void *recv_buf,
                       size_t count,
                       C_DataType data_type,
                       C_CCLComm comm,
                       C_Stream stream) {
  NCCL_CHECK(ncclAllGather(send_buf,
                           recv_buf,
                           count,
                           phi::internal::PDDataTypeToNcclDataType(data_type),
                           reinterpret_cast<ncclComm_t>(comm),
                           reinterpret_cast<cudaStream_t>(stream)));
  return C_SUCCESS;
}

C_Status XcclReduceScatter(void *send_buf,
                           void *recv_buf,
                           size_t count,
                           C_DataType data_type,
                           C_CCLReduceOp op,
                           C_CCLComm comm,
                           C_Stream stream) {
  NCCL_CHECK(
      ncclReduceScatter(send_buf,
                        recv_buf,
                        count,
                        phi::internal::PDDataTypeToNcclDataType(data_type),
                        PDReduceOpToNcclReduceOp(op),
                        reinterpret_cast<ncclComm_t>(comm),
                        reinterpret_cast<cudaStream_t>(stream)));
  return C_SUCCESS;
}

C_Status XcclGroupStart() {
  NCCL_CHECK(ncclGroupStart());
  return C_SUCCESS;
}

C_Status XcclGroupEnd() {
  NCCL_CHECK(ncclGroupEnd());
  return C_SUCCESS;
}

C_Status XcclSend(void *send_buf,
                  size_t count,
                  C_DataType data_type,
                  size_t dest_rank,
                  C_CCLComm comm,
                  C_Stream stream) {
  NCCL_CHECK(ncclSend(send_buf,
                      count,
                      phi::internal::PDDataTypeToNcclDataType(data_type),
                      dest_rank,
                      reinterpret_cast<ncclComm_t>(comm),
                      reinterpret_cast<cudaStream_t>(stream)));
  return C_SUCCESS;
}

C_Status XcclRecv(void *recv_buf,
                  size_t count,
                  C_DataType data_type,
                  size_t src_rank,
                  C_CCLComm comm,
                  C_Stream stream) {
  NCCL_CHECK(ncclRecv(recv_buf,
                      count,
                      phi::internal::PDDataTypeToNcclDataType(data_type),
                      src_rank,
                      reinterpret_cast<ncclComm_t>(comm),
                      reinterpret_cast<cudaStream_t>(stream)));
  return C_SUCCESS;
}
C_Status Xccl_all_to_all(const void **send_buf,
                         const size_t *send_count,
                         const C_DataType *send_dtype,
                         void **recv_buf,
                         const size_t *recv_count,
                         const C_DataType *recv_dtype,
                         size_t rank,
                         size_t nranks,
                         C_CCLComm comm,
                         C_Stream stream) {
  NCCL_CHECK(ncclGroupStart());
  for (size_t i = 0; i < nranks; ++i) {
    if (send_count[i] != 0)
      NCCL_CHECK(
          ncclSend(send_buf[i],
                   send_count[i],
                   phi::internal::PDDataTypeToNcclDataType(send_dtype[i]),
                   i,
                   reinterpret_cast<ncclComm_t>(comm),
                   reinterpret_cast<cudaStream_t>(stream)));
    if (recv_count[i] != 0)
      NCCL_CHECK(
          ncclRecv(recv_buf[i],
                   recv_count[i],
                   phi::internal::PDDataTypeToNcclDataType(recv_dtype[i]),
                   i,
                   reinterpret_cast<ncclComm_t>(comm),
                   reinterpret_cast<cudaStream_t>(stream)));
  }
  NCCL_CHECK(ncclGroupEnd());
  return C_SUCCESS;
}

C_Status InitBlasHandle(const C_Device device,
                        C_BLASHandle *blas_handle,
                        C_Stream stream) {
  PADDLE_RETRY_CUDA_SUCCESS(phi::dynload::cublasCreate(
      reinterpret_cast<cublasHandle_t *>(blas_handle)));
  PADDLE_RETRY_CUDA_SUCCESS(phi::dynload::cublasSetStream(
      *reinterpret_cast<cublasHandle_t *>(blas_handle),
      reinterpret_cast<cudaStream_t>((stream))));
  return C_SUCCESS;
}

C_Status InitBlasLtHandle(const C_Device device,
                          C_BLASLtHandle *blaslt_handle) {
  phi::dynload::cublasLtCreate(
      reinterpret_cast<cublasLtHandle_t *>(blaslt_handle));
  return C_SUCCESS;
}

C_Status DestroyBlasLtHandle(const C_Device device,
                             C_BLASLtHandle blaslt_handle) {
  if (blaslt_handle != nullptr) {
    phi::dynload::cublasLtDestroy(
        reinterpret_cast<cublasLtHandle_t>(blaslt_handle));
    blaslt_handle = nullptr;
  }
  return C_SUCCESS;
}

C_Status DestroyBlasHandle(const C_Device device, C_BLASHandle blas_handle) {
  if (blas_handle != nullptr) {
    phi::dynload::cublasDestroy(reinterpret_cast<cublasHandle_t>(blas_handle));
    blas_handle = nullptr;
  }
  return C_SUCCESS;
}

C_Status BlasSetMathMode(const C_Device device,
                         C_BLASHandle blas_handle,
                         int math_mode) {
  if (math_mode == 1) {
    PADDLE_RETRY_CUDA_SUCCESS(phi::dynload::cublasSetMathMode(
        reinterpret_cast<cublasHandle_t>(blas_handle), CUBLAS_TENSOR_OP_MATH));
  } else if (math_mode == 2) {
    PADDLE_RETRY_CUDA_SUCCESS(phi::dynload::cublasSetMathMode(
        reinterpret_cast<cublasHandle_t>(blas_handle),
        CUBLAS_TF32_TENSOR_OP_MATH));
  } else {
    PADDLE_RETRY_CUDA_SUCCESS(phi::dynload::cublasSetMathMode(
        reinterpret_cast<cublasHandle_t>(blas_handle), CUBLAS_DEFAULT_MATH));
  }
  return C_SUCCESS;
}

C_Status IsFloat16Supported(const C_Device device, bool *supported) {
  *supported = true;
  return C_SUCCESS;
}

C_Status IsBFloat16Supported(const C_Device device, bool *supported) {
  *supported = true;
  return C_SUCCESS;
}

C_Status IsDNNSupported(const C_Device device, bool *supported) {
  *supported = true;
  return C_SUCCESS;
}

C_Status InitDnnHandle(const C_Device device,
                       C_DNNHandle *dnn_handle,
                       C_Stream stream) {
  if (phi::dynload::HasCUDNN()) {
    PADDLE_RETRY_CUDA_SUCCESS(phi::dynload::cudnnCreate(
        reinterpret_cast<cudnnHandle_t *>(dnn_handle)));
    PADDLE_RETRY_CUDA_SUCCESS(phi::dynload::cudnnSetStream(
        *reinterpret_cast<cudnnHandle_t *>(dnn_handle),
        reinterpret_cast<cudaStream_t>(stream)));
    return C_SUCCESS;
  } else {
    *dnn_handle = nullptr;
    LOG(WARNING) << "cudnn library not found! setting dnn_handle to nullptr.";
    return C_SUCCESS;
  }
}

C_Status DestroyDnnHandle(const C_Device device, C_DNNHandle dnn_handle) {
  if (dnn_handle != nullptr) {
    phi::dynload::cudnnDestroy(reinterpret_cast<cudnnHandle_t>(dnn_handle));
    dnn_handle = nullptr;
  }
  return C_SUCCESS;
}

C_Status CudaStreamBeginCapture(const C_Device device,
                                C_Stream stream,
                                C_StreamCaptureMode mode) {
  if (cudaStreamBeginCapture(cudaStream_t(stream),
                             cudaStreamCaptureMode(mode)) != cudaSuccess)
    return C_ERROR;
  return C_SUCCESS;
}

C_Status CudaStreamEndCaptrue(const C_Device device,
                              C_Stream stream,
                              C_CudaGraph *pGraph) {
  if (cudaStreamEndCapture(cudaStream_t(stream),
                           reinterpret_cast<cudaGraph_t *>(pGraph)) !=
      cudaSuccess)
    return C_ERROR;
  return C_SUCCESS;
}

C_Status CudaGraphGetNodes(C_CudaGraph graph,
                           C_CudaGraphNode *pNode,
                           size_t *numNodes) {
  if (cudaGraphGetNodes(cudaGraph_t(graph), nullptr, numNodes) != cudaSuccess)
    return C_ERROR;
  return C_SUCCESS;
}

C_Status CudaGraphLaunch(const C_Device device,
                         C_GraphExec exec,
                         C_Stream stream) {
  cudaError_t result =
      cudaGraphLaunch(cudaGraphExec_t(exec), cudaStream_t(stream));
  if (result != cudaSuccess) {
    return C_ERROR;
  }
  return C_SUCCESS;
}

C_Status CudaGraphDestroy(C_CudaGraph graph) {
  if (cudaGraphDestroy(cudaGraph_t(graph)) != cudaSuccess) return C_ERROR;
  return C_SUCCESS;
}

C_Status CudaGraphExecDestroy(C_GraphExec exec) {
  if (cudaGraphExecDestroy(cudaGraphExec_t(exec)) != cudaSuccess)
    return C_ERROR;
  return C_SUCCESS;
}

C_Status CudaGraphInstantiate(C_GraphExec *pExec,
                              C_CudaGraph *pGraph,
                              void **pErrorNode,
                              char *pLogBuffer,
                              size_t bufferSize) {
  if (cudaGraphInstantiateWithFlags(reinterpret_cast<cudaGraphExec_t *>(pExec),
                                    *(reinterpret_cast<cudaGraph_t *>(pGraph)),
                                    cudaGraphInstantiateFlagAutoFreeOnLaunch) !=
      cudaSuccess)
    return C_ERROR;
  return C_SUCCESS;
}

C_Status CudaStreamCaptureInfo(const C_Device device,
                               C_Stream stream,
                               C_StreamCaptureStatus *captureStatus_out,
                               unsigned long long *id_out,  // NOLINT
                               C_CudaGraph *graph_out,
                               C_CudaGraphNode *dependencies_out,
                               void **edgeData_out,
                               size_t *numDependencies_out) {
  if (cudaStreamGetCaptureInfo(
          cudaStream_t(stream),
          reinterpret_cast<cudaStreamCaptureStatus *>(captureStatus_out),
          id_out) != cudaSuccess)
    return C_ERROR;
  return C_SUCCESS;
}

C_Status CudaThreadExchangeStreamCaptureMode(C_StreamCaptureMode *mode) {
  if (cudaThreadExchangeStreamCaptureMode(
          reinterpret_cast<cudaStreamCaptureMode *>(mode)) != cudaSuccess)
    return C_ERROR;
  return C_SUCCESS;
}

C_Status CudaGraphDebugDotPrint(C_CudaGraph graph,
                                const char *path,
                                unsigned int flags) {
  if (cudaGraphDebugDotPrint(cudaGraph_t(graph), path, flags) != cudaSuccess)
    return C_ERROR;
  return C_SUCCESS;
}

C_Status GetParameterSettersForExecGraph(C_CudaGraph graph,
                                         C_GraphHookManager *c_hook) {
  using parameterSetter_t =
      std::function<void(phi::backends::gpu::gpuKernelParams &)>;
  struct SetKernelParamsCtx {
    parameterSetter_t setter_func;
    cudaGraphNode_t node;
    cudaKernelNodeParams params;
  };

  size_t num_nodes;
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaGraphGetNodes(cudaGraph_t(graph), nullptr, &num_nodes));
  std::vector<cudaGraphNode_t> nodes(num_nodes);
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaGraphGetNodes(cudaGraph_t(graph), nodes.data(), &num_nodes));

  std::vector<SetKernelParamsCtx *> cts_all;
  std::vector<C_GraphExecuterSetter> hooks;
  for (auto node : nodes) {
    cudaGraphNodeType pType;
    cudaError_t result = cudaGraphNodeGetType(node, &pType);
    assert(result == CUDA_SUCCESS);
    if (pType == cudaGraphNodeTypeKernel) {
      cudaKernelNodeParams params = {};
      result = cudaGraphKernelNodeGetParams(node, &params);
      assert(result == CUDA_SUCCESS);
      phi::backends::gpu::gpuKernelParams kernel_params(params.kernelParams);
      auto kernel =
          phi::backends::gpu::CUDAGraphNodeLauncher::Instance()
              .parameterSetters.find(static_cast<cudaFunction_t>(params.func));
      if (kernel != phi::backends::gpu::CUDAGraphNodeLauncher::Instance()
                        .parameterSetters.end()) {
        auto launchSequence = kernel->second;
        unsigned int id = kernel_params.As<int>(0);
        auto parameterSetter = launchSequence.find(id);
        if (parameterSetter != launchSequence.end()) {
          auto setter = parameterSetter->second;
          SetKernelParamsCtx *ctx = new SetKernelParamsCtx;
          ctx->node = node;
          ctx->params = params;
          ctx->setter_func = setter;
          cts_all.emplace_back(ctx);
          hooks.emplace_back([](C_GraphExec exec_graph, void *userdate) {
            SetKernelParamsCtx *tmp =
                reinterpret_cast<SetKernelParamsCtx *>(userdate);
            phi::backends::gpu::gpuKernelParams kernel_params(
                tmp->params.kernelParams);
            tmp->setter_func(kernel_params);
            cudaGraphExecKernelNodeSetParams(
                reinterpret_cast<cudaGraphExec_t>(exec_graph),
                tmp->node,
                &tmp->params);
          });
        } else {
          PADDLE_THROW(common::errors::InvalidArgument(
              "Error: does not find launch id"));
        }
      }
    }
  }

  c_hook->size = cts_all.size();
  if (cts_all.size() != 0) {
    c_hook->size = cts_all.size();
    c_hook->hooks = reinterpret_cast<C_GraphExecuterSetter *>(
        malloc(sizeof(C_GraphExecuterSetter) * cts_all.size()));
    c_hook->user_data =
        reinterpret_cast<void **>(malloc(sizeof(void *) * cts_all.size()));
    std::memcpy(c_hook->hooks,
                hooks.data(),
                cts_all.size() * sizeof(C_GraphExecuterSetter));
    std::memcpy(
        c_hook->user_data, cts_all.data(), cts_all.size() * sizeof(void *));
  }
  return C_SUCCESS;
}

void InitPlugin(CustomRuntimeParams *params) {
  PADDLE_CUSTOM_RUNTIME_CHECK_VERSION(params);
  params->device_type = const_cast<char *>(DeviceType);
  params->sub_device_type = const_cast<char *>(SubDeviceType);

  memset(reinterpret_cast<void *>(params->interface),
         0,
         sizeof(C_DeviceInterface));

  params->interface->cuda_stream_begin_capture = CudaStreamBeginCapture;
  params->interface->cuda_stream_end_captrue = CudaStreamEndCaptrue;
  params->interface->cuda_graph_launch = CudaGraphLaunch;
  params->interface->cuda_graph_destroy = CudaGraphDestroy;
  params->interface->cuda_graph_exec_destroy = CudaGraphExecDestroy;
  params->interface->cuda_graph_instantiate = CudaGraphInstantiate;
  params->interface->cuda_graph_get_nodes = CudaGraphGetNodes;
  params->interface->cuda_stream_capture_info = CudaStreamCaptureInfo;
  params->interface->get_parameter_setter_for_exec_graph =
      GetParameterSettersForExecGraph;
  params->interface->cuda_thread_exchange_stream_capthure_mode =
      CudaThreadExchangeStreamCaptureMode;
  params->interface->cuda_graph_debug_dot_print = CudaGraphDebugDotPrint;
  params->interface->get_compute_capability = GetComputeCapability;
  params->interface->get_device_properties = GetDeviceProperties;
  params->interface->get_runtime_version = GetRuntimeVersion;
  params->interface->get_driver_version = GetDriverVersion;
  params->interface->get_multi_process = GetMultiProcessors;
  params->interface->get_max_threads_per_mp = GetMaxThreadsPerMultiProcessor;
  params->interface->get_max_threads_per_block = GetMaxThreadsPerBlock;
  params->interface->get_max_shared_mem_per_block = GetMaxSharedMemPerBlock;
  params->interface->get_max_blocks_per_mp = GetMaxBlocksPerMultiProcessor;
  params->interface->get_warp_size = GetWarpSize;
  params->interface->get_max_registers_per_mp =
      GetMaxRegistersPerMultiProcessor;
  params->interface->get_vector_width = GetPreferredVectorWidth;
  params->interface->get_max_grid_dim_size = GetMaxGridDimSize;
  params->interface->get_max_block_dim_size = GetMaxBlockDimSize;

  params->interface->init_device = InitDevice;
  params->interface->set_device = SetDevice;
  params->interface->get_device = GetDevice;
  params->interface->deinit_device = DestroyDevice;

  params->interface->create_stream = CreateStream;
  params->interface->destroy_stream = DestroyStream;

  params->interface->create_event = CreateEvent;
  params->interface->destroy_event = DestroyEvent;
  params->interface->record_event = RecordEvent;

  params->interface->synchronize_device = SyncDevice;
  params->interface->synchronize_stream = SyncStream;
  params->interface->synchronize_event = SyncEvent;
  params->interface->stream_wait_event = StreamWaitEvent;

  params->interface->memory_copy_h2d = MemCpyH2D;
  params->interface->memory_copy_d2d = MemCpyD2D;
  params->interface->memory_copy_d2h = MemCpyD2H;
  params->interface->memory_copy_p2p = MemCpyP2P;
  params->interface->async_memory_copy_h2d = AsyncMemCpyH2D;
  params->interface->async_memory_copy_d2d = AsyncMemCpyD2D;
  params->interface->async_memory_copy_d2h = AsyncMemCpyD2H;
  params->interface->async_memory_copy_p2p = nullptr;
  params->interface->device_memory_allocate = Allocate;
  params->interface->host_memory_allocate = AllocateHost;
  params->interface->unified_memory_allocate = nullptr;
  params->interface->device_memory_deallocate = Deallocate;
  params->interface->host_memory_deallocate = DeallocateHost;
  params->interface->unified_memory_deallocate = nullptr;

  params->interface->get_device_count = GetDevicesCount;
  params->interface->get_device_list = GetDevicesList;

  params->interface->device_memory_stats = nullptr;
  params->interface->device_min_chunk_size = DeviceMinChunkSize;
  params->interface->device_max_chunk_size = DeviceMaxChunkSize;

  params->interface->init_eigen_device = InitEigenDevice;
  params->interface->destroy_eigen_device = DestroyEigenDevice;

  params->interface->is_dnn_supported = IsDNNSupported;

  params->interface->is_float16_supported = IsFloat16Supported;

  params->interface->is_bfloat16_supported = IsBFloat16Supported;

  params->interface->init_blas_handle = InitBlasHandle;
  params->interface->init_blaslt_handle = InitBlasLtHandle;
  params->interface->destroy_blas_handle = DestroyBlasHandle;
  params->interface->destroy_blaslt_handle = DestroyBlasLtHandle;
  params->interface->blas_set_math_mode = BlasSetMathMode;
  params->interface->init_dnn_handle = InitDnnHandle;
  params->interface->destroy_dnn_handle = DestroyDnnHandle;

  params->interface->xccl_all_gather = XcclAllGather;
  params->interface->xccl_all_reduce = XcclAllReduce;
  params->interface->xccl_broadcast = XcclBroadcast;
  params->interface->xccl_comm_init_rank = XcclCommInitRank;
  params->interface->xccl_destroy_comm = XcclDestroyComm;
  params->interface->xccl_get_unique_id = XcclGetUniqueId;
  params->interface->xccl_get_unique_id_size = XcclGetUniqueIdSize;
  params->interface->xccl_group_end = XcclGroupEnd;
  params->interface->xccl_group_start = XcclGroupStart;
  params->interface->xccl_recv = XcclRecv;
  params->interface->xccl_reduce = XcclReduce;
  params->interface->xccl_reduce_scatter = XcclReduceScatter;
  params->interface->xccl_send = XcclSend;
  params->interface->xccl_all_to_all = Xccl_all_to_all;

  // profiler
  params->interface->profiler_collect_trace_data = ProfilerCollectData;
  params->interface->profiler_initialize = ProfilerInitialize;
  params->interface->profiler_finalize = ProfilerFinalize;
  params->interface->profiler_start_tracing = ProfilerStart;
  params->interface->profiler_stop_tracing = ProfilerStop;
  params->interface->profiler_prepare_tracing = ProfilerPrepare;

  // PIR pass pipeline
  params->pir_default_passes = reinterpret_cast<void *>(
      const_cast<std::vector<std::string> *>(GetPirMetaxGpuPasses()));

  // CINN interface init
#ifdef WITH_CINN
  if (params->interface) {
    paddle::custom_device::metax::InitCinnInterface(params->interface);
    LOG(INFO) << "[MetaX] CINN Interface registered successfully.";
  }
#endif
}
