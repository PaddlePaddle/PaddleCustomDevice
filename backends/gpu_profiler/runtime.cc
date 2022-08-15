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

#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "cuda.h"
#include "cupti.h"
#include "paddle/phi/backends/custom/trace_event.h"
#include "paddle/phi/backends/device_ext.h"
#include "process_data.h"

#define CUDA_CHECK(x) x
#define CUPTI_CALL(call)                                                   \
  do {                                                                     \
    CUptiResult _status = call;                                            \
    if (_status != CUPTI_SUCCESS) {                                        \
      const char *errstr;                                                  \
      cuptiGetResultString(_status, &errstr);                              \
      std::cerr << "Function " << #call << " failed with error " << errstr \
                << std::endl;                                              \
      exit(-1);                                                            \
    }                                                                      \
  } while (0)

#define INTERFACE_UNIMPLEMENT                      \
  throw std::runtime_error(std::string(__func__) + \
                           " is not implemented on gpu_profiler.")

int device_count() {
  int count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  return count;
}

int get_device_count() {
  static int count = device_count();
  return count;
}

C_Status SetDevice(const C_Device device) {
  CUDA_CHECK(cudaSetDevice(device->id));
  return C_SUCCESS;
}

C_Status GetDevice(const C_Device device) {
  CUDA_CHECK(cudaGetDevice(&(device->id)));
  return C_SUCCESS;
}

C_Status GetDevicesCount(size_t *count) {
  *count = get_device_count();
  return C_SUCCESS;
}

C_Status GetDevicesList(size_t *devices) {
  for (auto i = 0; i < get_device_count(); ++i) {
    devices[i] = i;
  }
  return C_SUCCESS;
}

C_Status MemCpy(const C_Device device,
                void *dst,
                const void *src,
                size_t size) {
  INTERFACE_UNIMPLEMENT;
  CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
  return C_SUCCESS;
}

C_Status AsyncMemCpy(const C_Device device,
                     C_Stream stream,
                     void *dst,
                     const void *src,
                     size_t size) {
  INTERFACE_UNIMPLEMENT;
  CUDA_CHECK(cudaMemcpyAsync(dst,
                             src,
                             size,
                             cudaMemcpyHostToDevice,
                             reinterpret_cast<cudaStream_t>(stream)));
  return C_SUCCESS;
}

C_Status MemCpyP2P(const C_Device dst_device,
                   const C_Device src_device,
                   void *dst,
                   const void *src,
                   size_t size) {
  INTERFACE_UNIMPLEMENT;
  return C_SUCCESS;
}

C_Status AsyncMemCpyP2P(const C_Device dst_device,
                        const C_Device src_device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size) {
  INTERFACE_UNIMPLEMENT;
  return C_SUCCESS;
}

C_Status Allocate(const C_Device device, void **ptr, size_t size) {
  INTERFACE_UNIMPLEMENT;
  return C_FAILED;
}

C_Status Deallocate(const C_Device device, void *ptr, size_t size) {
  INTERFACE_UNIMPLEMENT;
  return C_SUCCESS;
}

C_Status CreateStream(const C_Device device, C_Stream *stream) {
  INTERFACE_UNIMPLEMENT;
  return C_SUCCESS;
}

C_Status DestroyStream(const C_Device device, C_Stream stream) {
  INTERFACE_UNIMPLEMENT;
  return C_SUCCESS;
}

C_Status CreateEvent(const C_Device device, C_Event *event) {
  INTERFACE_UNIMPLEMENT;
  return C_SUCCESS;
}

C_Status RecordEvent(const C_Device device, C_Stream stream, C_Event event) {
  INTERFACE_UNIMPLEMENT;

  return C_SUCCESS;
}

C_Status DestroyEvent(const C_Device device, C_Event event) {
  INTERFACE_UNIMPLEMENT;

  return C_SUCCESS;
}

C_Status SyncStream(const C_Device device, C_Stream stream) {
  INTERFACE_UNIMPLEMENT;

  return C_SUCCESS;
}

C_Status SyncEvent(const C_Device device, C_Event event) {
  INTERFACE_UNIMPLEMENT;

  return C_SUCCESS;
}

C_Status StreamWaitEvent(const C_Device device,
                         C_Stream stream,
                         C_Event event) {
  INTERFACE_UNIMPLEMENT;

  return C_SUCCESS;
}

C_Status VisibleDevices(size_t *devices) {
  INTERFACE_UNIMPLEMENT;
  return C_SUCCESS;
}

C_Status DeviceMemStats(const C_Device device,
                        size_t *total_memory,
                        size_t *free_memory) {
  INTERFACE_UNIMPLEMENT;
  return C_SUCCESS;
}

C_Status DeviceMinChunkSize(const C_Device device, size_t *size) {
  *size = 4096;
  return C_SUCCESS;
}

C_Status SyncDevice(const C_Device device) {
  CUDA_CHECK(cudaDeviceSynchronize());
  return C_SUCCESS;
}

///////// CUPTI

void CUPTIAPI BufferRequestedCallback(uint8_t **buffer,
                                      size_t *size,
                                      size_t *max_num_records) {
  Tracer::Instance().AllocateBuffer(buffer, size);
  *max_num_records = 0;
}

void CUPTIAPI BufferCompletedCallback(CUcontext ctx,
                                      uint32_t stream_id,
                                      uint8_t *buffer,
                                      size_t size,
                                      size_t valid_size) {
  Tracer::Instance().ProduceBuffer(buffer, valid_size);
  size_t dropped = 0;
  CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, stream_id, &dropped));
  if (dropped != 0) {
    std::cerr << "Stream " << stream_id << " Dropped " << dropped
              << " activity records";
  }
}

std::unordered_map<uint32_t, uint64_t> CreateThreadIdMapping() {
  std::unordered_map<uint32_t, uint64_t> mapping;
  std::unordered_map<uint64_t, ThreadId> ids = GetAllThreadIds();
  for (const auto &id : ids) {
    mapping[id.second.cupti_tid] = id.second.sys_tid;
  }
  return mapping;
}

int ProcessCuptiActivity(C_Profiler prof, uint64_t tracing_start_ns_) {
  int record_cnt = 0;
  CUPTI_CALL(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
  auto mapping = CreateThreadIdMapping();
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
  }
  return record_cnt;
}

C_Status ProfilerInitialize(C_Profiler prof, void **user_data) {
  return C_SUCCESS;
}

C_Status ProfilerFinalize(C_Profiler prof, void *user_data) {
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
  std::cout << "enable cupti activity\n";
  return C_SUCCESS;
}

C_Status ProfilerStart(C_Profiler prof, void *user_data) {
  Tracer::Instance().ConsumeBuffers();
  return C_SUCCESS;
}

C_Status ProfilerStop(C_Profiler prof, void *user_data) {
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DRIVER));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMSET));
  std::cout << "disable cupti activity\n";
  return C_SUCCESS;
}

C_Status ProfilerCollectData(C_Profiler prof,
                             uint64_t tracing_start_ns_,
                             void *user_data) {
  ProcessCuptiActivity(prof, tracing_start_ns_);
  return C_SUCCESS;
}

void InitPlugin(CustomRuntimeParams *params) {
  PADDLE_CUSTOM_RUNTIME_CHECK_VERSION(params);
  params->device_type = "gpu_profiler";
  params->sub_device_type = "v0.1";

  memset(reinterpret_cast<void *>(params->interface),
         0,
         sizeof(C_DeviceInterface));

  // params->interface->initialize = Init;
  // params->interface->finalize = Finalize;
  // params->interface->init_device = InitDevice;
  // params->interface->deinit_device = DestroyDevice;

  params->interface->set_device = SetDevice;
  params->interface->get_device = GetDevice;
  params->interface->create_stream = CreateStream;
  params->interface->destroy_stream = DestroyStream;
  params->interface->create_event = CreateEvent;
  params->interface->destroy_event = DestroyEvent;
  params->interface->record_event = RecordEvent;
  params->interface->synchronize_stream = SyncStream;
  params->interface->synchronize_event = SyncEvent;
  params->interface->stream_wait_event = StreamWaitEvent;
  params->interface->memory_copy_h2d = MemCpy;
  params->interface->memory_copy_d2d = MemCpy;
  params->interface->memory_copy_d2h = MemCpy;
  params->interface->async_memory_copy_h2d = AsyncMemCpy;
  params->interface->async_memory_copy_d2d = AsyncMemCpy;
  params->interface->async_memory_copy_d2h = AsyncMemCpy;
  params->interface->device_memory_allocate = Allocate;
  params->interface->device_memory_deallocate = Deallocate;
  params->interface->device_memory_stats = DeviceMemStats;
  params->interface->device_min_chunk_size = DeviceMinChunkSize;

  params->interface->get_device_count = GetDevicesCount;
  params->interface->get_device_list = GetDevicesList;
  params->interface->synchronize_device = SyncDevice;
  params->interface->profiler_collect_trace_data = ProfilerCollectData;
  params->interface->profiler_initialize = ProfilerInitialize;
  params->interface->profiler_finalize = ProfilerFinalize;
  params->interface->profiler_start_tracing = ProfilerStart;
  params->interface->profiler_stop_tracing = ProfilerStop;
  params->interface->profiler_prepare_tracing = ProfilerPrepare;
}
