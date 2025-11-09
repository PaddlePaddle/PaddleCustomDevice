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

// #include "runtime/process_cupti_data.h"

#include "runtime/process_cupti_data.h"

#include <cupti.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cassert>
#include <cstdio>
#include <sstream>
#include <thread>

#include "paddle/phi/backends/dynload/cupti.h"

pid_t gettid() { return syscall(SYS_gettid); }

// Get system-wide realtime clock in nanoseconds
inline uint64_t PosixInNsec() {
#ifdef _POSIX_C_SOURCE
  struct timespec tp;
  clock_gettime(CLOCK_REALTIME, &tp);
  return tp.tv_sec * 1000 * 1000 * 1000 + tp.tv_nsec;
#else
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return 1000 * (static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec);
#endif
}

inline std::string demangle(std::string name) {
  int status = -4;
  std::unique_ptr<char, void (*)(void*)> res{
      abi::__cxa_demangle(name.c_str(), NULL, NULL, &status), std::free};
  return (status == 0) ? res.get() : name;
}

void AddKernelRecord(const CUpti_ActivityKernel4* kernel,
                     uint64_t start_ns,
                     C_Profiler collector) {
  if (kernel->start < start_ns) {
    return;
  }
  phi::DeviceTraceEvent event;
  event.name = demangle(kernel->name);
  event.type = phi::TracerEventType::Kernel;
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

  profiler_add_device_trace_event(collector, &event);
}

const char* MemcpyKind(uint8_t kind) {
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

void AddMemcpyRecord(const CUpti_ActivityMemcpy* memcpy,
                     uint64_t start_ns,
                     C_Profiler collector) {
  if (memcpy->start < start_ns) {
    return;
  }
  phi::DeviceTraceEvent event;
  event.name = MemcpyKind(memcpy->copyKind);
  event.type = phi::TracerEventType::Memcpy;
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
           phi::kMemKindMaxLen,
           "%s",
           MemcpyKind(memcpy->srcKind));
  snprintf(event.memcpy_info.dst_kind,
           phi::kMemKindMaxLen,
           "%s",
           MemcpyKind(memcpy->dstKind));

  profiler_add_device_trace_event(collector, &event);

  // collector->AddDeviceEvent(std::move(event));
}

void AddMemcpy2Record(const CUpti_ActivityMemcpy2* memcpy2,
                      uint64_t start_ns,
                      C_Profiler collector) {
  if (memcpy2->start < start_ns) {
    return;
  }
  phi::DeviceTraceEvent event;
  event.name = MemcpyKind(memcpy2->copyKind);
  event.type = phi::TracerEventType::Memcpy;
  event.start_ns = memcpy2->start;
  event.end_ns = memcpy2->end;
  event.device_id = memcpy2->deviceId;
  event.context_id = memcpy2->contextId;
  event.stream_id = memcpy2->streamId;
  event.correlation_id = memcpy2->correlationId;
  event.memcpy_info.num_bytes = memcpy2->bytes;

  snprintf(event.memcpy_info.src_kind,
           phi::kMemKindMaxLen,
           "%s",
           MemcpyKind(memcpy2->srcKind));
  snprintf(event.memcpy_info.dst_kind,
           phi::kMemKindMaxLen,
           "%s",
           MemcpyKind(memcpy2->dstKind));

  profiler_add_device_trace_event(collector, &event);
}

void AddMemsetRecord(const CUpti_ActivityMemset* memset,
                     uint64_t start_ns,
                     C_Profiler collector) {
  if (memset->start < start_ns) {
    return;
  }
  phi::DeviceTraceEvent event;
  event.name = "MEMSET";
  // event.type = TracerEventType::Memset;
  event.type = phi::TracerEventType::Memset;
  event.start_ns = memset->start;
  event.end_ns = memset->end;
  event.device_id = memset->deviceId;
  event.context_id = memset->contextId;
  event.stream_id = memset->streamId;
  event.correlation_id = memset->correlationId;
  event.memset_info.num_bytes = memset->bytes;

  // snprintf(event.memset_info.memory_kind,
  //          phi::kMemKindMaxLen,
  //          "%s",
  //          MemoryKind(memset->memoryKind));

  event.memset_info.value = memset->value;

  // collector->AddDeviceEvent(std::move(event));

  profiler_add_device_trace_event(collector, &event);
}

class CuptiRuntimeCbidStr {
 public:
  static const CuptiRuntimeCbidStr& GetInstance() {
    static CuptiRuntimeCbidStr inst;
    return inst;
  }

  std::string RuntimeKind(CUpti_CallbackId cbid) const {
    auto iter = cbid_str_.find(cbid);
    if (iter == cbid_str_.end()) {
      return "metax Runtime API " + std::to_string(cbid);
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
  REGISTER_RUNTIME_CBID_STR(cudaDriverGetVersion_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaRuntimeGetVersion_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaGetDeviceCount_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaGetDeviceProperties_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaChooseDevice_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaGetLastError_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaPeekAtLastError_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaLaunch_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaFuncSetCacheConfig_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaFuncGetAttributes_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaSetDevice_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaGetDevice_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaSetValidDevices_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaSetDeviceFlags_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaMalloc_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaMallocPitch_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaFree_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaMallocArray_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaFreeArray_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaMallocHost_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaFreeHost_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaHostAlloc_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaHostGetDevicePointer_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaHostGetFlags_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaMemGetInfo_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaMemcpy_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaMemcpy2D_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaMemcpyToArray_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaMemcpy2DToArray_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaMemcpyToSymbol_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaMemcpyFromSymbol_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaMemcpyAsync_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaMemcpy2DAsync_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaMemcpyToSymbolAsync_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaMemcpyFromSymbolAsync_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaMemset_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaMemset2D_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaMemsetAsync_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaMemset2DAsync_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaGetSymbolAddress_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaGetSymbolSize_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaBindTexture_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaBindTexture2D_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaBindTextureToArray_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaUnbindTexture_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaStreamCreate_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaStreamDestroy_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaStreamSynchronize_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaStreamQuery_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaEventCreate_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaEventCreateWithFlags_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaEventRecord_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaEventDestroy_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaEventSynchronize_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaEventQuery_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaEventElapsedTime_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaMalloc3D_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaMalloc3DArray_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaMemset3D_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaMemset3DAsync_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaMemcpy3D_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaMemcpy3DAsync_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaStreamWaitEvent_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaPointerGetAttributes_v4000);
  REGISTER_RUNTIME_CBID_STR(cudaHostRegister_v4000);
  REGISTER_RUNTIME_CBID_STR(cudaHostUnregister_v4000);
  REGISTER_RUNTIME_CBID_STR(cudaDeviceCanAccessPeer_v4000);
  REGISTER_RUNTIME_CBID_STR(cudaDeviceEnablePeerAccess_v4000);
  REGISTER_RUNTIME_CBID_STR(cudaDeviceDisablePeerAccess_v4000);
  REGISTER_RUNTIME_CBID_STR(cudaMemcpyPeer_v4000);
  REGISTER_RUNTIME_CBID_STR(cudaMemcpyPeerAsync_v4000);
  REGISTER_RUNTIME_CBID_STR(cudaMemcpy3DPeer_v4000);
  REGISTER_RUNTIME_CBID_STR(cudaMemcpy3DPeerAsync_v4000);
  REGISTER_RUNTIME_CBID_STR(cudaDeviceReset_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaDeviceSynchronize_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaDeviceGetLimit_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaDeviceSetLimit_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaDeviceGetCacheConfig_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaDeviceSetCacheConfig_v3020);
  REGISTER_RUNTIME_CBID_STR(cudaProfilerInitialize_v4000);
  REGISTER_RUNTIME_CBID_STR(cudaProfilerStart_v4000);
  REGISTER_RUNTIME_CBID_STR(cudaProfilerStop_v4000);
  REGISTER_RUNTIME_CBID_STR(cudaDeviceGetByPCIBusId_v4010);
  REGISTER_RUNTIME_CBID_STR(cudaDeviceGetPCIBusId_v4010);
  REGISTER_RUNTIME_CBID_STR(cudaIpcGetEventHandle_v4010);
  REGISTER_RUNTIME_CBID_STR(cudaIpcOpenEventHandle_v4010);
  REGISTER_RUNTIME_CBID_STR(cudaIpcGetMemHandle_v4010);
  REGISTER_RUNTIME_CBID_STR(cudaIpcOpenMemHandle_v4010);
  REGISTER_RUNTIME_CBID_STR(cudaIpcCloseMemHandle_v4010);
  REGISTER_RUNTIME_CBID_STR(cudaFuncSetSharedMemConfig_v4020);
  REGISTER_RUNTIME_CBID_STR(cudaDeviceGetSharedMemConfig_v4020);
  REGISTER_RUNTIME_CBID_STR(cudaDeviceSetSharedMemConfig_v4020);
  REGISTER_RUNTIME_CBID_STR(cudaStreamAddCallback_v5000);
  REGISTER_RUNTIME_CBID_STR(cudaStreamCreateWithFlags_v5000);
  REGISTER_RUNTIME_CBID_STR(cudaDeviceGetAttribute_v5000);
  REGISTER_RUNTIME_CBID_STR(cudaStreamDestroy_v5050);
  REGISTER_RUNTIME_CBID_STR(cudaStreamCreateWithPriority_v5050);
  REGISTER_RUNTIME_CBID_STR(cudaStreamGetPriority_v5050);
  REGISTER_RUNTIME_CBID_STR(cudaStreamGetFlags_v5050);
  REGISTER_RUNTIME_CBID_STR(cudaDeviceGetStreamPriorityRange_v5050);
  REGISTER_RUNTIME_CBID_STR(cudaMallocManaged_v6000);
  REGISTER_RUNTIME_CBID_STR(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor_v6000);
  REGISTER_RUNTIME_CBID_STR(cudaStreamAttachMemAsync_v6000);
  REGISTER_RUNTIME_CBID_STR(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor_v6050);
  REGISTER_RUNTIME_CBID_STR(cudaLaunchKernel_v7000);
  REGISTER_RUNTIME_CBID_STR(cudaGetDeviceFlags_v7000);
  REGISTER_RUNTIME_CBID_STR(
      cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_v7000);
  REGISTER_RUNTIME_CBID_STR(cudaMemRangeGetAttribute_v8000);
  REGISTER_RUNTIME_CBID_STR(cudaMemRangeGetAttributes_v8000);
#if CUDA_VERSION >= 9000
  REGISTER_RUNTIME_CBID_STR(cudaLaunchCooperativeKernel_v9000);
  REGISTER_RUNTIME_CBID_STR(cudaLaunchCooperativeKernelMultiDevice_v9000);
  REGISTER_RUNTIME_CBID_STR(cudaFuncSetAttribute_v9000);
  REGISTER_RUNTIME_CBID_STR(cudaGraphLaunch_v10000);
  REGISTER_RUNTIME_CBID_STR(cudaStreamSetAttribute_v11000);
  REGISTER_RUNTIME_CBID_STR(cudaMallocAsync_v11020);
  REGISTER_RUNTIME_CBID_STR(cudaFreeAsync_v11020);
#endif
#undef REGISTER_RUNTIME_CBID_STR
}

void AddApiRecord(const CUpti_ActivityAPI* api,
                  uint64_t start_ns,
                  const std::unordered_map<uint32_t, uint64_t> tid_mapping,
                  C_Profiler collector) {
  if (api->start < start_ns) {
    return;
  }
  phi::RuntimeTraceEvent event;
  event.name = CuptiRuntimeCbidStr::GetInstance().RuntimeKind(api->cbid);
  event.start_ns = api->start;
  event.end_ns = api->end;
  event.process_id = phi::GetProcessId();
  uint64_t tid = gettid();
  auto iter = tid_mapping.find(api->threadId);
  if (iter == tid_mapping.end()) {
  } else {
    tid = iter->second;
  }

  event.thread_id = tid;

  event.correlation_id = api->correlationId;
  event.callback_id = api->cbid;

  event.type = phi::TracerEventType::CudaRuntime;
  profiler_add_runtime_trace_event(collector, &event);
}

void ProcessCuptiActivityRecord(
    const CUpti_Activity* record,
    uint64_t start_ns,
    const std::unordered_map<uint32_t, uint64_t> tid_mapping,
    C_Profiler collector) {
  switch (record->kind) {  // 差异
    case CUPTI_ACTIVITY_KIND_KERNEL:
    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
      AddKernelRecord(reinterpret_cast<const CUpti_ActivityKernel4*>(record),
                      start_ns,
                      collector);
      break;
    case CUPTI_ACTIVITY_KIND_MEMCPY:
      AddMemcpyRecord(reinterpret_cast<const CUpti_ActivityMemcpy*>(record),
                      start_ns,
                      collector);
      break;
    case CUPTI_ACTIVITY_KIND_MEMCPY2:
      AddMemcpy2Record(reinterpret_cast<const CUpti_ActivityMemcpy2*>(record),
                       start_ns,
                       collector);
      break;
    case CUPTI_ACTIVITY_KIND_MEMSET:
      AddMemsetRecord(reinterpret_cast<const CUpti_ActivityMemset*>(record),
                      start_ns,
                      collector);
      break;
    case CUPTI_ACTIVITY_KIND_DRIVER:
    case CUPTI_ACTIVITY_KIND_RUNTIME:
      AddApiRecord(reinterpret_cast<const CUpti_ActivityAPI*>(record),
                   start_ns,
                   tid_mapping,
                   collector);
      break;
    default:
      break;
  }
}

void* AlignedMalloc(size_t size, size_t alignment) {
  assert(alignment >= sizeof(void*) && (alignment & (alignment - 1)) == 0);
  size = (size + alignment - 1) / alignment * alignment;
#if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
  void* aligned_mem = nullptr;
  if (posix_memalign(&aligned_mem, alignment, size) != 0) {
    aligned_mem = nullptr;
  }
  return aligned_mem;
#elif defined(_WIN32)
  return _aligned_malloc(size, alignment);
#else
  void* mem = malloc(size + alignment);  // NOLINT
  if (mem == nullptr) {
    return nullptr;
  }
  size_t adjust = alignment - reinterpret_cast<uint64_t>(mem) % alignment;
  void* aligned_mem = reinterpret_cast<char*>(mem) + adjust;
  *(reinterpret_cast<void**>(aligned_mem) - 1) = mem;
  assert(reinterpret_cast<uint64_t>(aligned_mem) % alignment == 0);
  return aligned_mem;
#endif
}

void AlignedFree(void* mem_ptr) {
#if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
  free(mem_ptr);
#elif defined(_WIN32)
  _aligned_free(mem_ptr);
#else
  if (mem_ptr) {
    free(*(reinterpret_cast<void**>(mem_ptr) - 1));
  }
#endif
}

void Tracer::AllocateBuffer(uint8_t** buffer, size_t* size) {
  constexpr size_t kBufferSize = 1 << 23;  // 8 MB
  constexpr size_t kBufferAlignSize = 8;
  *buffer =
      reinterpret_cast<uint8_t*>(AlignedMalloc(kBufferSize, kBufferAlignSize));
  *size = kBufferSize;
}

void Tracer::ProduceBuffer(uint8_t* buffer, size_t valid_size) {
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

void Tracer::ReleaseBuffer(uint8_t* buffer) { AlignedFree(buffer); }

const char* MemoryKind(uint16_t kind) {
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

void profiler_add_device_trace_event(C_Profiler prof, void* event) {
  paddle::platform::DeviceTraceEvent de =
      *reinterpret_cast<paddle::platform::DeviceTraceEvent*>(event);
  reinterpret_cast<paddle::platform::TraceEventCollector*>(prof)
      ->AddDeviceEvent(std::move(de));
}

#define CUPTI_CALL(call)                                                     \
  do {                                                                       \
    CUptiResult _status = call;                                              \
    if (_status != CUPTI_SUCCESS) {                                          \
      const char* errstr;                                                    \
      cuptiGetResultString(_status, &errstr);                                \
      LOG(ERROR) << "Function " << #call << " failed with error " << errstr; \
      exit(-1);                                                              \
    }                                                                        \
  } while (0)

namespace details {
std::unordered_map<uint32_t, uint64_t> CreateThreadIdMapping() {
  std::unordered_map<uint32_t, uint64_t> mapping;
  std::unordered_map<uint64_t, phi::ThreadId> ids = phi::GetAllThreadIds();
  for (const auto& id : ids) {
    mapping[id.second.cupti_tid] = id.second.sys_tid;
  }
  return mapping;
}
}  // namespace details
