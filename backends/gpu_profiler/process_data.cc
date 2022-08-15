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

#include "process_data.h"

#include <sys/syscall.h>

#include <sstream>
#include <thread>

#include "os_info.h"
#include "thread_data_registry.h"

pid_t gettid() { return syscall(SYS_gettid); }

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

inline std::string demangle(std::string name) { return name; }

float CalculateEstOccupancy(uint32_t DeviceId,
                            uint16_t RegistersPerThread,
                            int32_t StaticSharedMemory,
                            int32_t DynamicSharedMemory,
                            int32_t BlockX,
                            int32_t BlockY,
                            int32_t BlockZ,
                            float BlocksPerSm) {
  float occupancy = 0.0;

  cudaDeviceProp device_property;
  cudaGetDeviceProperties(&device_property, DeviceId);

  cudaOccFuncAttributes occFuncAttr;
  occFuncAttr.maxThreadsPerBlock = INT_MAX;
  occFuncAttr.numRegs = RegistersPerThread;
  occFuncAttr.sharedSizeBytes = StaticSharedMemory;
  occFuncAttr.partitionedGCConfig = PARTITIONED_GC_OFF;
  occFuncAttr.shmemLimitConfig = FUNC_SHMEM_LIMIT_DEFAULT;
  occFuncAttr.maxDynamicSharedSizeBytes = 0;
  const cudaOccDeviceState occDeviceState = {};
  int blockSize = BlockX * BlockY * BlockZ;
  size_t dynamicSmemSize = DynamicSharedMemory;
  cudaOccResult occ_result;
  cudaOccDeviceProp prop(device_property);
  cudaOccError status =
      cudaOccMaxActiveBlocksPerMultiprocessor(&occ_result,
                                              &prop,
                                              &occFuncAttr,
                                              &occDeviceState,
                                              blockSize,
                                              dynamicSmemSize);
  if (status == CUDA_OCC_SUCCESS) {
    if (occ_result.activeBlocksPerMultiprocessor < BlocksPerSm) {
      BlocksPerSm = occ_result.activeBlocksPerMultiprocessor;
    }
    occupancy = BlocksPerSm * blockSize /
                static_cast<float>(device_property.maxThreadsPerMultiProcessor);
  } else {
    std::cerr << "Failed to calculate estimated occupancy, status = " << status
              << std::endl;
  }
  return occupancy;
}

void AddKernelRecord(const CUpti_ActivityKernel4 *kernel,
                     uint64_t start_ns,
                     C_Profiler collector) {
  if (kernel->start < start_ns) {
    return;
  }
  paddle::platform::DeviceTraceEvent event;
  event.name = demangle(kernel->name);
  event.type = paddle::platform::TracerEventType::Kernel;
  event.start_ns = kernel->start;
  event.end_ns = kernel->end;
  event.device_id = kernel->deviceId;
  event.context_id = kernel->contextId;
  event.stream_id = kernel->streamId;
  event.correlation_id = kernel->correlationId;
  event.kernel_info.block_x = kernel->blockX;
  event.kernel_info.block_y = kernel->blockY;
  event.kernel_info.block_z = kernel->blockZ;
  event.kernel_info.grid_x = kernel->gridX;
  event.kernel_info.grid_y = kernel->gridY;
  event.kernel_info.grid_z = kernel->gridZ;
  event.kernel_info.dynamic_shared_memory = kernel->dynamicSharedMemory;
  event.kernel_info.static_shared_memory = kernel->staticSharedMemory;
  event.kernel_info.registers_per_thread = kernel->registersPerThread;
  event.kernel_info.local_memory_per_thread = kernel->localMemoryPerThread;
  event.kernel_info.local_memory_total = kernel->localMemoryTotal;
  event.kernel_info.queued = kernel->queued;
  event.kernel_info.submitted = kernel->submitted;
  event.kernel_info.completed = kernel->completed;

  float blocks_per_sm = 0.0;
  float warps_per_sm = 0.0;
  float occupancy = 0.0;

  constexpr int threads_per_warp = 32;
  cudaDeviceProp device_property;
  cudaGetDeviceProperties(&device_property, kernel->deviceId);
  blocks_per_sm =
      static_cast<float>(event.kernel_info.grid_x * event.kernel_info.grid_y *
                         event.kernel_info.grid_z) /
      device_property.multiProcessorCount;
  warps_per_sm = blocks_per_sm *
                 (event.kernel_info.block_x * event.kernel_info.block_y *
                  event.kernel_info.block_z) /
                 threads_per_warp;
  occupancy = CalculateEstOccupancy(kernel->deviceId,
                                    event.kernel_info.registers_per_thread,
                                    event.kernel_info.static_shared_memory,
                                    event.kernel_info.dynamic_shared_memory,
                                    event.kernel_info.block_x,
                                    event.kernel_info.block_y,
                                    event.kernel_info.block_z,
                                    blocks_per_sm);
  event.kernel_info.blocks_per_sm = blocks_per_sm;
  event.kernel_info.warps_per_sm = warps_per_sm;
  event.kernel_info.occupancy = occupancy;

  profiler_add_device_trace_event(collector, &event);
}

const char *MemcpyKind(uint8_t kind) {
  switch (kind) {
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
      return "MEMCPY_HtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
      return "MEMCPY_DtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
      return "MEMCPY_HtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
      return "MEMCPY_AtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
      return "MEMCPY_AtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
      return "MEMCPY_AtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
      return "MEMCPY_DtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
      return "MEMCPY_DtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
      return "MEMCPY_HtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP:
      return "MEMCPY_PtoP";
    default:
      return "MEMCPY";
  }
}

const char *MemoryKind(uint16_t kind) {
  switch (kind) {
    case CUPTI_ACTIVITY_MEMORY_KIND_UNKNOWN:
      return "Unknown";
    case CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE:
      return "Pageable";
    case CUPTI_ACTIVITY_MEMORY_KIND_PINNED:
      return "Pinned";
    case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE:
      return "Device";
    case CUPTI_ACTIVITY_MEMORY_KIND_ARRAY:
      return "Array";
    case CUPTI_ACTIVITY_MEMORY_KIND_MANAGED:
      return "Managed";
    case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC:
      return "Device Static";
    case CUPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC:
      return "Managed Static";
    default:
      return "Unknown";
  }
}

void AddMemcpyRecord(const CUpti_ActivityMemcpy *memcpy,
                     uint64_t start_ns,
                     C_Profiler collector) {
  if (memcpy->start < start_ns) {
    return;
  }
  paddle::platform::DeviceTraceEvent event;
  event.name = MemcpyKind(memcpy->copyKind);
  event.type = paddle::platform::TracerEventType::Memcpy;
  event.start_ns = memcpy->start;
  event.end_ns = memcpy->end;
  event.device_id = memcpy->deviceId;
  event.context_id = memcpy->contextId;
  event.stream_id = memcpy->streamId;
  event.correlation_id = memcpy->correlationId;
  event.memcpy_info.num_bytes = memcpy->bytes;
  // snprintf(event.memcpy_info.copy_kind, kMemKindMaxLen, "%s",
  //         MemcpyKind(memcpy->copyKind));
  snprintf(event.memcpy_info.src_kind,
           paddle::platform::kMemKindMaxLen,
           "%s",
           MemcpyKind(memcpy->srcKind));
  snprintf(event.memcpy_info.dst_kind,
           paddle::platform::kMemKindMaxLen,
           "%s",
           MemcpyKind(memcpy->dstKind));
  // collector->AddDeviceEvent(std::move(event));
  profiler_add_device_trace_event(collector, &event);
}

void AddMemcpy2Record(const CUpti_ActivityMemcpy2 *memcpy2,
                      uint64_t start_ns,
                      C_Profiler collector) {
  if (memcpy2->start < start_ns) {
    return;
  }
  paddle::platform::DeviceTraceEvent event;
  event.name = MemcpyKind(memcpy2->copyKind);
  event.type = paddle::platform::TracerEventType::Memcpy;
  event.start_ns = memcpy2->start;
  event.end_ns = memcpy2->end;
  event.device_id = memcpy2->deviceId;
  event.context_id = memcpy2->contextId;
  event.stream_id = memcpy2->streamId;
  event.correlation_id = memcpy2->correlationId;
  event.memcpy_info.num_bytes = memcpy2->bytes;
  // snprintf(event.memcpy_info.copy_kind, kMemKindMaxLen, "%s",
  // MemcpyKind(memcpy2->copyKind));
  snprintf(event.memcpy_info.src_kind,
           paddle::platform::kMemKindMaxLen,
           "%s",
           MemcpyKind(memcpy2->srcKind));
  snprintf(event.memcpy_info.dst_kind,
           paddle::platform::kMemKindMaxLen,
           "%s",
           MemcpyKind(memcpy2->dstKind));
  // collector->AddDeviceEvent(std::move(event));
  profiler_add_device_trace_event(collector, &event);
}

void AddMemsetRecord(const CUpti_ActivityMemset *memset,
                     uint64_t start_ns,
                     C_Profiler collector) {
  if (memset->start < start_ns) {
    return;
  }
  paddle::platform::DeviceTraceEvent event;
  event.name = "MEMSET";
  event.type = paddle::platform::TracerEventType::Memset;
  event.start_ns = memset->start;
  event.end_ns = memset->end;
  event.device_id = memset->deviceId;
  event.context_id = memset->contextId;
  event.stream_id = memset->streamId;
  event.correlation_id = memset->correlationId;
  event.memset_info.num_bytes = memset->bytes;
  snprintf(event.memset_info.memory_kind,
           paddle::platform::kMemKindMaxLen,
           "%s",
           MemoryKind(memset->memoryKind));
  event.memset_info.value = memset->value;
  // collector->AddDeviceEvent(std::move(event));
  profiler_add_device_trace_event(collector, &event);
}

class CuptiRuntimeCbidStr {
 public:
  static const CuptiRuntimeCbidStr &GetInstance() {
    static CuptiRuntimeCbidStr inst;
    return inst;
  }

  std::string RuntimeKind(CUpti_CallbackId cbid) const {
    auto iter = cbid_str_.find(cbid);
    if (iter == cbid_str_.end()) {
      return "Runtime API " + std::to_string(cbid);
    }
    return iter->second;
  }

 private:
  CuptiRuntimeCbidStr();

  std::unordered_map<CUpti_CallbackId, std::string> cbid_str_;
};

CuptiRuntimeCbidStr::CuptiRuntimeCbidStr() {
#define REGISTER_RUNTIME_CBID_STR(cbid) \
  cbid_str_[CUPTI_RUNTIME_TRACE_CBID_##cbid] = #cbid
  REGISTER_RUNTIME_CBID_STR(cudaBindTexture_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaConfigureCall_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaDeviceGetAttribute_v5000);
  REGISTER_RUNTIME_CBID_STR(cudaDeviceGetStreamPriorityRange_v5050);
  REGISTER_RUNTIME_CBID_STR(cudaDeviceSynchronize_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaDriverGetVersion_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaEventCreateWithFlags_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaEventDestroy_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaEventDestroy_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaEventQuery_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaEventRecord_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaFreeHost_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaFree_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaFuncGetAttributes_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaGetDeviceCount_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaGetDeviceProperties_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaGetDevice_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaGetErrorString_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaGetLastError_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaHostAlloc_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaHostGetDevicePointer_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaLaunchKernel_v7000);
  REGISTER_RUNTIME_CBID_STR(cudaMallocHost_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaMalloc_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaMemcpyAsync_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaMemcpy_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaMemsetAsync_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaMemset_v3020);
  REGISTER_RUNTIME_CBID_STR(
      cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_v7000);
  REGISTER_RUNTIME_CBID_STR(cudaPeekAtLastError_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaRuntimeGetVersion_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaSetDevice_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaStreamCreate_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaStreamCreateWithFlags_v5000);
  REGISTER_RUNTIME_CBID_STR(cudaStreamCreateWithPriority_v5050);
  REGISTER_RUNTIME_CBID_STR(cudaStreamDestroy_v5050);
  REGISTER_RUNTIME_CBID_STR(cudaStreamSynchronize_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaStreamWaitEvent_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaUnbindTexture_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaSetupArgument_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaLaunch_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaDeviceGetPCIBusId_v4010);
#if CUDA_VERSION >= 9000
  REGISTER_RUNTIME_CBID_STR(cudaLaunchCooperativeKernel_v9000);
  REGISTER_RUNTIME_CBID_STR(cudaLaunchCooperativeKernelMultiDevice_v9000);
#endif
#undef REGISTER_RUNTIME_CBID_STR
}

void AddApiRecord(const CUpti_ActivityAPI *api,
                  uint64_t start_ns,
                  const std::unordered_map<uint32_t, uint64_t> tid_mapping,
                  C_Profiler collector) {
  if (api->start < start_ns) {
    return;
  }
  paddle::platform::RuntimeTraceEvent event;
  event.name = CuptiRuntimeCbidStr::GetInstance().RuntimeKind(api->cbid);
  event.start_ns = api->start;
  event.end_ns = api->end;
  event.process_id = GetProcessId();
  uint64_t tid = 0;
  auto iter = tid_mapping.find(api->threadId);
  if (iter == tid_mapping.end()) {
    tid = gettid();
  } else {
    tid = iter->second;
  }
#ifdef PADDLE_WITH_HIP
  event.thread_id = api->threadId;
#else
  event.thread_id = tid;
#endif
  event.correlation_id = api->correlationId;
  event.callback_id = api->cbid;
  // collector->AddRuntimeEvent(std::move(event));
  profiler_add_runtime_trace_event(collector, &event);
}

void ProcessCuptiActivityRecord(
    const CUpti_Activity *record,
    uint64_t start_ns,
    const std::unordered_map<uint32_t, uint64_t> tid_mapping,
    C_Profiler collector) {
  switch (record->kind) {
    case CUPTI_ACTIVITY_KIND_KERNEL:
    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
      AddKernelRecord(reinterpret_cast<const CUpti_ActivityKernel4 *>(record),
                      start_ns,
                      collector);
      break;
    case CUPTI_ACTIVITY_KIND_MEMCPY:
      AddMemcpyRecord(reinterpret_cast<const CUpti_ActivityMemcpy *>(record),
                      start_ns,
                      collector);
      break;
    case CUPTI_ACTIVITY_KIND_MEMCPY2:
      AddMemcpy2Record(reinterpret_cast<const CUpti_ActivityMemcpy2 *>(record),
                       start_ns,
                       collector);
      break;
    case CUPTI_ACTIVITY_KIND_MEMSET:
      AddMemsetRecord(reinterpret_cast<const CUpti_ActivityMemset *>(record),
                      start_ns,
                      collector);
      break;
    case CUPTI_ACTIVITY_KIND_DRIVER:
    case CUPTI_ACTIVITY_KIND_RUNTIME:
      AddApiRecord(reinterpret_cast<const CUpti_ActivityAPI *>(record),
                   start_ns,
                   tid_mapping,
                   collector);
      break;
    default:
      break;
  }
}

void Tracer::AllocateBuffer(uint8_t **buffer, size_t *size) {
  constexpr size_t kBufSize = 1 << 23;  // 8 MB
  constexpr size_t kBufAlign = 8;       // 8 B
  *buffer = reinterpret_cast<uint8_t *>(AlignedMalloc(kBufSize, kBufAlign));
  *size = kBufSize;
}

void Tracer::ProduceBuffer(uint8_t *buffer, size_t valid_size) {
  std::lock_guard<std::mutex> guard(activity_buffer_lock_);
  activity_buffers_.emplace_back(buffer, valid_size);
}

std::vector<ActivityBuffer> Tracer::ConsumeBuffers() {
  std::vector<ActivityBuffer> buffers;
  {
    std::lock_guard<std::mutex> guard(activity_buffer_lock_);
    buffers.swap(activity_buffers_);
  }
  return buffers;
}

void Tracer::ReleaseBuffer(uint8_t *buffer) { AlignedFree(buffer); }
