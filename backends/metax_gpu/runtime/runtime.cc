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
#define EIGEN_USE_GPU
#include <cuda_runtime.h>
#include <errno.h>
#include <fcntl.h>
#include <semaphore.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>

#include "glog/logging.h"
#include "paddle/phi/backends/device_ext.h"
#include "unsupported/Eigen/CXX11/Tensor"

#define MEMORY_FRACTION 0.5f

static int global_current_device = 0;

// C_Status Init() {
//   std::cout << "custom_cpu plugin compiled with ";
// #ifdef __clang__
//   std::cout << "clang\n";
// #else
//   std::cout << "gcc\n";
// #endif
//   return C_SUCCESS;
// }
C_Status GetComputeCapability(const C_Device device,
                              size_t *compute_capability) {
  int id = device->id;
  int major, minor;
  auto major_error_code =
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, id);
  auto minor_error_code =
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, id);

  *compute_capability = major * 10 + minor;
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

C_Status InitDevice(const C_Device device) {
  if (!device || device->id < 0) {
    return C_ERROR;
  }

  cudaError_t err;
  int current_dev;

  if ((err = cudaSetDevice(device->id)) != cudaSuccess) {
    return C_ERROR;
  }

  if ((err = cudaGetDevice(&current_dev)) != cudaSuccess) {
    return C_ERROR;
  }

  return (current_dev == device->id) ? C_SUCCESS : C_ERROR;
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

  // free(device->id);

  return C_SUCCESS;
}

C_Status Finalize() { return C_SUCCESS; }

C_Status GetDevicesCount(size_t *count) {
  *count = 4;
  return C_SUCCESS;
}

C_Status GetDevicesList(size_t *devices) {
  devices[0] = 0;
  devices[1] = 1;
  devices[2] = 2;
  devices[3] = 3;
  return C_SUCCESS;
}

C_Status MemCpyH2D(const C_Device device,
                   void *dst,
                   const void *src,
                   size_t size) {
  if (dst == NULL || src == NULL) {
    return C_ERROR;
  }

  if (size == 0) {
    return C_SUCCESS;
  }

  cudaError_t cudaErr = cudaSetDevice(device->id);
  if (cudaErr != cudaSuccess) {
    return C_ERROR;
  }

  cudaErr = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
  if (cudaErr != cudaSuccess) {
    return C_ERROR;
  }

  return C_SUCCESS;
  return C_SUCCESS;
}

C_Status MemCpyD2D(const C_Device device,
                   void *dst,
                   const void *src,
                   size_t size) {
  cudaError_t err;
  int original_device;
  cudaError_t err_restore;

  err = cudaSetDevice(device->id);
  if (err != cudaSuccess) {
    return C_ERROR;
  }

  err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);

  if (err == cudaSuccess) {
    return C_SUCCESS;
  } else {
    return C_ERROR;
  }
}

// C_Status MemCpyD2D(const C_Device device,
//                    void *dst,
//                    const void *src,
//                    size_t size) {
//     return C_ERROR;
//     if (device == NULL || dst == NULL || src == NULL || size == 0) {
//         return C_ERROR;
//     }

//     cudaError_t cudaErr;

//     int currentDeviceId;
//     cudaErr = cudaGetDevice(&currentDeviceId);
//     if (cudaErr != cudaSuccess) {
//         return C_ERROR;
//     }

//     if (currentDeviceId == device->id) {
//         cudaErr = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
//         if (cudaErr != cudaSuccess) {
//             return C_ERROR;
//         }
//         return C_SUCCESS;
//     }

//     void* hostBuffer = NULL;
//     cudaErr = cudaMallocHost(&hostBuffer, size);
//     if (cudaErr != cudaSuccess) {
//         return C_ERROR;
//     }

//     cudaErr = cudaMemcpy(hostBuffer, src, size, cudaMemcpyDeviceToHost);
//     if (cudaErr != cudaSuccess) {
//         cudaFreeHost(hostBuffer);
//         return C_ERROR;
//     }

//     cudaErr = cudaSetDevice(device->id);
//     if (cudaErr != cudaSuccess) {
//         cudaFreeHost(hostBuffer);
//         return C_ERROR;
//     }

//     cudaErr = cudaMemcpy(dst, hostBuffer, size, cudaMemcpyHostToDevice);

//     cudaFreeHost(hostBuffer);

//     cudaSetDevice(currentDeviceId);

//     if (cudaErr != cudaSuccess) {
//         return C_ERROR;
//     }

//     return C_SUCCESS;
// }

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

  return C_SUCCESS;
}

C_Status MemCpy(const C_Device device,
                void *dst,
                const void *src,
                size_t size) {
  memcpy(dst, src, size);
  return C_ERROR;
}

C_Status AsyncMemCpy(const C_Device device,
                     C_Stream stream,
                     void *dst,
                     const void *src,
                     size_t size) {
  memcpy(dst, src, size);
  return C_ERROR;
}

C_Status MemCpyP2P(const C_Device dst_device,
                   const C_Device src_device,
                   void *dst,
                   const void *src,
                   size_t size) {
  memcpy(dst, src, size);
  return C_ERROR;
}

C_Status AsyncMemCpyP2P(const C_Device dst_device,
                        const C_Device src_device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size) {
  memcpy(dst, src, size);
  return C_ERROR;
}

C_Status Allocate(const C_Device device, void **ptr, size_t size) {
  cudaError_t err;
  // int original_device;
  *ptr = NULL;

  // err = cudaGetDevice(&original_device);
  // if (err != cudaSuccess) {
  //     return C_ERROR;
  // }

  err = cudaSetDevice(device->id);
  if (err != cudaSuccess) {
    return C_ERROR;
  }

  err = cudaMalloc(ptr, size);
  if (err != cudaSuccess) {
    // cudaSetDevice(original_device);
    *ptr = NULL;
    return C_ERROR;
  }

  // cudaError_t restore_err = cudaSetDevice(original_device);
  // if (restore_err != cudaSuccess) {
  //     cudaError_t temp_err = cudaSetDevice(device->id);
  //     if (temp_err == cudaSuccess) {
  //         cudaFree(*ptr);
  //         *ptr = NULL;
  //     }
  //     return C_ERROR;
  // }

  return C_SUCCESS;
}

C_Status Deallocate(const C_Device device, void *ptr, size_t size) {
  cudaSetDevice(device->id);
  cudaFree(ptr);
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

  int previous_device;
  cudaError_t cuda_status;

  cuda_status = cudaGetDevice(&previous_device);
  if (cuda_status != cudaSuccess) {
    return C_ERROR;
  }

  cuda_status = cudaSetDevice(device->id);
  if (cuda_status != cudaSuccess) {
    return C_ERROR;
  }

  cudaEvent_t evt;
  cuda_status = cudaEventCreate(&evt);
  if (cuda_status != cudaSuccess) {
    cudaSetDevice(previous_device);
    return C_ERROR;
  }

  cuda_status = cudaSetDevice(previous_device);
  if (cuda_status != cudaSuccess) {
    cudaSetDevice(device->id);
    cudaEventDestroy(evt);
    return C_ERROR;
  }

  *event = (C_Event)evt;
  return C_SUCCESS;
}

C_Status RecordEvent(const C_Device device, C_Stream stream, C_Event event) {
  if (device == NULL || event == NULL) {
    return C_ERROR;
  }

  int previous_device;
  cudaError_t cuda_status;

  cuda_status = cudaGetDevice(&previous_device);
  if (cuda_status != cudaSuccess) {
    return C_ERROR;
  }

  cuda_status = cudaSetDevice(device->id);
  if (cuda_status != cudaSuccess) {
    return C_ERROR;
  }

  cuda_status = cudaEventRecord(cudaEvent_t(event), cudaStream_t(stream));
  if (cuda_status != cudaSuccess) {
    cudaSetDevice(previous_device);
    return C_ERROR;
  }

  cuda_status = cudaSetDevice(previous_device);
  if (cuda_status != cudaSuccess) {
    return C_ERROR;
  }

  return C_SUCCESS;
}

C_Status DestroyEvent(const C_Device device, C_Event event) {
  if (device == NULL || event == NULL) {
    return C_ERROR;
  }

  int previous_device;
  cudaError_t cuda_status;

  cuda_status = cudaGetDevice(&previous_device);
  if (cuda_status != cudaSuccess) {
    return C_ERROR;
  }

  cuda_status = cudaSetDevice(device->id);
  if (cuda_status != cudaSuccess) {
    return C_ERROR;
  }

  cuda_status = cudaEventDestroy(cudaEvent_t(event));
  if (cuda_status != cudaSuccess) {
    cudaSetDevice(previous_device);
    return C_ERROR;
  }

  cuda_status = cudaSetDevice(previous_device);
  if (cuda_status != cudaSuccess) {
    return C_ERROR;
  }

  return C_SUCCESS;
}

C_Status SyncDevice(const C_Device device) {
  cudaError_t err;
  // int original_device;

  // err = cudaGetDevice(&original_device);
  // if (err != cudaSuccess) {
  //     return C_ERROR;
  // }

  err = cudaSetDevice(device->id);
  if (err != cudaSuccess) {
    return C_ERROR;
  }

  err = cudaDeviceSynchronize();
  cudaError_t sync_err = err;

  // cudaError_t restore_err = cudaSetDevice(original_device);

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

  int previous_device;
  cudaError_t cuda_status;

  cuda_status = cudaGetDevice(&previous_device);
  if (cuda_status != cudaSuccess) {
    return C_ERROR;
  }
  cuda_status = cudaSetDevice(device->id);
  if (cuda_status != cudaSuccess) {
    return C_ERROR;
  }
  cuda_status = cudaEventSynchronize(cudaEvent_t(event));
  if (cuda_status != cudaSuccess) {
    cudaSetDevice(previous_device);
    return C_ERROR;
  }

  cuda_status = cudaSetDevice(previous_device);
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

  int previous_device;
  cudaError_t cuda_status;

  cuda_status = cudaGetDevice(&previous_device);
  if (cuda_status != cudaSuccess) {
    return C_ERROR;
  }

  cuda_status = cudaSetDevice(device->id);
  if (cuda_status != cudaSuccess) {
    return C_ERROR;
  }

  cuda_status =
      cudaStreamWaitEvent(cudaStream_t(stream), cudaEvent_t(event), 0);
  if (cuda_status != cudaSuccess) {
    cudaSetDevice(previous_device);
    return C_ERROR;
  }

  cuda_status = cudaSetDevice(previous_device);
  if (cuda_status != cudaSuccess) {
    return C_ERROR;
  }

  return C_SUCCESS;
}

C_Status VisibleDevices(size_t *devices) { return C_ERROR; }

// modified for eigen test
C_Status InitEigenDevice(const C_Device device,
                         Eigen::GpuDevice *eigen_device) {
  // cudaStream_t stream;
  //   cudaError_t cuda_err = cudaStreamCreate(&stream);
  //   if (cuda_err != cudaSuccess) {
  //       return C_ERROR;
  //   }

  //   Eigen::GpuStreamDevice* stream_device = new
  //   Eigen::GpuStreamDevice(&stream);

  //   if (stream_device == nullptr) {
  //       cudaStreamDestroy(stream);
  //       return C_ERROR;
  //   }

  //   new (eigen_device) Eigen::GpuDevice(stream_device);
  return C_SUCCESS;
}

C_Status DestoryEigenDevice(const C_Device device,
                            Eigen::GpuDevice *eigen_device) {
  if (eigen_device == nullptr) return C_SUCCESS;

  eigen_device->~GpuDevice();

  return C_SUCCESS;
}

// C_Status DeviceMemStats(const C_Device device,
//                         size_t *total_memory,
//                         size_t *free_memory) {
//   float memusage;
//   FILE *fp;
//   char buffer[1024];
//   size_t byte_read;
//   char *pos;

//   fp = fopen("/proc/meminfo", "r");
//   byte_read = fread(buffer, 1, sizeof(buffer), fp);
//   fclose(fp);
//   buffer[byte_read] = '\0';
//   pos = strstr(buffer, "MemTotal:");
//   sscanf(pos, "MemTotal: %lu kB", total_memory);
//   pos = strstr(pos, "MemFree:");
//   sscanf(pos, "MemFree: %lu kB", free_memory);
//   *total_memory = *total_memory * 1024;
//   *free_memory = *free_memory * 1024;
//   *free_memory = *free_memory * MEMORY_FRACTION;

//   return C_SUCCESS;
// }

C_Status DeviceMinChunkSize(const C_Device device, size_t *size) {
  VLOG(10) << "Runtime: GPU min chunk size is " << (1 << 8);
  *size = 1 << 8;
  return C_SUCCESS;
}

// struct C_CCLComm_st {
//   size_t rank;
//   size_t nranks;
//   sem_t *sig;
//   sem_t *sig_2;
//   std::string sig_name;
//   std::string sig_2_name;
// };

// // for unittest
// C_Status XcclGetUniqueIdSize(size_t *sz) {
//   *sz = sizeof(size_t);
//   return C_SUCCESS;
// }

// C_Status XcclGetUniqueId(C_CCLRootId *unique_id) {
//   auto ptr = reinterpret_cast<int8_t *>(unique_id->data);
//   for (auto i = 0; i < unique_id->sz - 1; ++i) {
//     ptr[i] = static_cast<int8_t>(std::rand() % ('z' - 'a') + 'a');
//   }
//   ptr[unique_id->sz - 1] = '\0';
//   return C_SUCCESS;
// }

// C_Status XcclCommInitRank(size_t ranks,
//                           C_CCLRootId *unique_id,
//                           size_t rank,
//                           C_CCLComm *comm) {
//   auto sig = sem_open(static_cast<char *>(unique_id->data), O_CREAT, 0644,
//   0); auto sig_2 =
//       sem_open(static_cast<char *>(unique_id->data) + 1, O_CREAT, 0644, 0);
//   *comm =
//       new C_CCLComm_st({rank,
//                         ranks,
//                         sig,
//                         sig_2,
//                         std::string(static_cast<char *>(unique_id->data)),
//                         std::string(static_cast<char *>(unique_id->data) +
//                         1)});
//   return C_SUCCESS;
// }

// C_Status XcclDestroyComm(C_CCLComm comm) {
//   if (comm) {
//     sem_unlink(comm->sig_name.c_str());
//     sem_unlink(comm->sig_2_name.c_str());
//     delete comm;
//   }
//   return C_SUCCESS;
// }

// C_Status XcclAllReduce(void *send_buf,
//                        void *recv_buf,
//                        size_t count,
//                        C_DataType data_type,
//                        C_CCLReduceOp op,
//                        C_CCLComm comm,
//                        C_Stream stream) {
//   sem_post(comm->sig);

//   if (comm->rank == 0) {
//     for (auto i = 0; i < comm->nranks; ++i) {
//       sem_wait(comm->sig);
//     }

//     for (auto i = 0; i < comm->nranks; ++i) {
//       sem_post(comm->sig_2);
//     }
//   }

//   sem_wait(comm->sig_2);
//   return C_SUCCESS;
// }

// C_Status XcclBroadcast(void *buf,
//                        size_t count,
//                        C_DataType data_type,
//                        size_t root,
//                        C_CCLComm comm,
//                        C_Stream stream) {
//   sem_post(comm->sig);

//   if (comm->rank == 0) {
//     for (auto i = 0; i < comm->nranks; ++i) {
//       sem_wait(comm->sig);
//     }

//     for (auto i = 0; i < comm->nranks; ++i) {
//       sem_post(comm->sig_2);
//     }
//   }

//   sem_wait(comm->sig_2);
//   return C_SUCCESS;
// }

// C_Status ProfilerInitialize(C_Profiler prof, void **user_data) {
//   return C_SUCCESS;
// }

// C_Status ProfilerFinalize(C_Profiler prof, void *user_data) {
//   return C_SUCCESS;
// }

// C_Status ProfilerPrepare(C_Profiler prof, void *user_data) { return
// C_SUCCESS; }

// C_Status ProfilerStart(C_Profiler prof, void *user_data) { return C_SUCCESS;
// }

// C_Status ProfilerStop(C_Profiler prof, void *user_data) { return C_SUCCESS; }

// C_Status ProfilerCollectData(C_Profiler prof,
//                              uint64_t start_ns,
//                              void *user_data) {
//   return C_SUCCESS;
// }

void InitPlugin(CustomRuntimeParams *params) {
  PADDLE_CUSTOM_RUNTIME_CHECK_VERSION(params);
  params->device_type = "metax_gpu";
  params->sub_device_type = "v0.1";

  memset(reinterpret_cast<void *>(params->interface),
         0,
         sizeof(C_DeviceInterface));

  params->interface->get_compute_capability = GetComputeCapability;
  params->interface->get_runtime_version = GetRuntimeVersion;
  params->interface->get_driver_version = GetDriverVersion;
  params->interface->get_multi_process = GetMultiProcessors;
  params->interface->get_max_threads_per_mp = GetMaxThreadsPerMultiProcessor;
  params->interface->get_max_threads_per_block = GetMaxThreadsPerBlock;
  params->interface->get_max_grid_dim_size = GetMaxGridDimSize;
  //   params->interface->initialize = Init;
  //   params->interface->finalize = Finalize;

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
  // params->interface->async_memory_copy_h2d = AsyncMemCpy;
  // params->interface->async_memory_copy_d2d = AsyncMemCpyD2D;
  // params->interface->async_memory_copy_d2h = AsyncMemCpy;
  // params->interface->async_memory_copy_p2p = AsyncMemCpyP2P;
  params->interface->device_memory_allocate = Allocate;
  //   params->interface->host_memory_allocate = Allocate;
  //   params->interface->unified_memory_allocate = Allocate;
  params->interface->device_memory_deallocate = Deallocate;
  //   params->interface->host_memory_deallocate = Deallocate;
  //   params->interface->unified_memory_deallocate = Deallocate;

  params->interface->get_device_count = GetDevicesCount;
  params->interface->get_device_list = GetDevicesList;

  params->interface->init_eigen_device = InitEigenDevice;
  params->interface->destory_eigen_device = DestoryEigenDevice;
  //   params->interface->device_memory_stats = DeviceMemStats;
  params->interface->device_min_chunk_size = DeviceMinChunkSize;

  //   params->interface->xccl_get_unique_id_size = XcclGetUniqueIdSize;
  //   params->interface->xccl_get_unique_id = XcclGetUniqueId;
  //   params->interface->xccl_comm_init_rank = XcclCommInitRank;
  //   params->interface->xccl_destroy_comm = XcclDestroyComm;
  //   params->interface->xccl_all_reduce = XcclAllReduce;
  //   params->interface->xccl_broadcast = XcclBroadcast;

  //   params->interface->profiler_collect_trace_data = ProfilerCollectData;
  //   params->interface->profiler_initialize = ProfilerInitialize;
  //   params->interface->profiler_finalize = ProfilerFinalize;
  //   params->interface->profiler_start_tracing = ProfilerStart;
  //   params->interface->profiler_stop_tracing = ProfilerStop;
  //   params->interface->profiler_prepare_tracing = ProfilerPrepare;
}
