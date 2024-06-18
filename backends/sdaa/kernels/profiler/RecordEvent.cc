// BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
// reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
// OF SUCH DAMAGE.

#include "RecordEvent.h"

#include <sys/syscall.h>

#include <cassert>
#include <cstdio>
#include <cstring>
#include <thread>

#include "kernels/profiler/sdaa_wrapper.h"
#include "paddle/phi/extension.h"

constexpr size_t kBufSize = 1 << 23;  // 8 MB
constexpr size_t kBufAlign = 8;       // 8 B

pid_t gettid() { return syscall(SYS_gettid); }

std::unordered_map<uint32_t, uint64_t> CreateThreadIdMapping() {
  std::unordered_map<uint32_t, uint64_t> mapping;
  std::unordered_map<uint64_t, phi::ThreadId> ids = phi::GetAllThreadIds();
  for (const auto &id : ids) {
    // sdaa_tid is same with sys_id.
    mapping[id.second.sys_tid] = id.second.sys_tid;
  }
  return mapping;
}

// NOTE: BufferRequested is not used any more, waiting for rewrite to remove
void RecordEvent::BufferRequested() {}

void RecordEvent::BufferCompleted() {
  for (size_t i = 0; i < records_num; ++i) {
    auto &&record = trace_events.front();
    ProcessKernelRecord(record.get());
    trace_events.pop_front();
  }
  records_num = 0;
}

void RecordEvent::init(const TraceEvent &event) {
  std::lock_guard<std::mutex> lock(mutex_);
  EmplaceEvent(std::move(event));
  records_num++;
}

void RecordEvent::AllocateSdptiBuffer(uint8_t **buffer, size_t *size) {
  *buffer = reinterpret_cast<uint8_t *>(AlignMalloc(kBufSize, kBufAlign));
  *size = kBufSize;
}

void RecordEvent::ReleaseSdptiBuffer(uint8_t *buffer) {
  if (buffer) {
    AlignFree(buffer);
  }
}

uint64_t RecordEvent::getBufferSize() { return kBufSize; }

uint64_t RecordEvent::getThread() { return gettid(); }

void RecordEvent::ProcessActivityRecord(
    const SDpti_Activity *record,
    const std::unordered_map<uint32_t, uint64_t> &tid_mapping) {
  switch (record->kind) {
    case SDPTI_ACTIVITY_KIND_RUNTIME:
      AddApiRecord(reinterpret_cast<const SDpti_ActivityAPI *>(record),
                   tid_mapping);
      break;
    case SDPTI_ACTIVITY_KIND_MEMCPY:
      AddMemcpyRecord(reinterpret_cast<const SDpti_ActivityMemcpy *>(record));
      break;
    case SDPTI_ACTIVITY_KIND_MEMSET:
      AddMemsetRecord(reinterpret_cast<const SDpti_ActivityMemset *>(record));
      break;
    case SDPTI_ACTIVITY_KIND_KERNEL:
      AddKernelRecord(reinterpret_cast<const SDpti_ActivityKernel *>(record));
      break;
    case SDPTI_ACTIVITY_KIND_MEMCPY_P2P:
      AddMemcpyP2PRecord(
          reinterpret_cast<const SDpti_ActivityMemcpyP2P *>(record));
      break;
    case SDPTI_ACTIVITY_KIND_MEMOPS:
      AddMemOpsRecord(reinterpret_cast<const SDpti_ActivityMemOps *>(record));
      break;
    default:
      break;
  }
}

void RecordEvent::ProcessKernelRecord(const TraceEvent *record) {
  if (record->start_ns < tracing_start_ns) {
    return;
  }
  phi::RuntimeTraceEvent event;
  size_t pos = record->name.find('(');
  event.name = record->name.substr(0, pos);
  event.start_ns = record->start_ns;
  event.end_ns = record->end_ns;
  event.process_id = GetProcessId();
  event.correlation_id = record->correlation_id;
  event.thread_id = record->thread_id;
  if (GetAttributeDumpMode()) {
    event.msg = record->msg;
  }
  profiler_add_runtime_trace_event(prof, &event);
}

void RecordEvent::AddMemcpyP2PRecord(const SDpti_ActivityMemcpyP2P *record) {
  phi::DeviceTraceEvent event;
  event.name = "MemcpyP2P";
  event.type = phi::TracerEventType::Memcpy;
  event.start_ns = record->start;
  event.end_ns = record->end;
  event.device_id = record->deviceId;
  event.stream_id = record->streamId;
  event.context_id = record->contextId;
  event.correlation_id = record->runtimeCorrelationId;
  event.memcpy_info.num_bytes = record->bytes;
  profiler_add_device_trace_event(prof, &event);
}

void RecordEvent::AddMemOpsRecord(const SDpti_ActivityMemOps *record) {
  phi::DeviceTraceEvent event;
  switch (record->memOpsKind) {
    case SDPTI_ACTIVITY_MEMOPS_WAIT:
      event.name = "WaitValue";
      break;
    case SDPTI_ACTIVITY_MEMOPS_SET:
      event.name = "SetValue";
      break;
    default:
      event.name = "MemOps";
      break;
  }

  event.type = phi::TracerEventType::Kernel;
  event.start_ns = record->start;
  event.end_ns = record->end;
  event.device_id = record->deviceId;
  event.stream_id = record->streamId;
  event.context_id = record->contextId;
  event.correlation_id = record->runtimeCorrelationId;
  profiler_add_device_trace_event(prof, &event);
}

void RecordEvent::AddKernelRecord(const SDpti_ActivityKernel *record) {
  phi::DeviceTraceEvent event;
  event.name = record->name;
  // NOTE(liaotianju): we only need tccl prefix to be replace with xccl
  if (UNLIKELY(event.name.find("tccl") == 0)) {
    event.name.replace(0, 4, COMMUNICATION_KERNEL_PREFIX);
  }
  event.type = phi::TracerEventType::Kernel;
  event.start_ns = record->start;
  event.end_ns = record->end;
  event.device_id = record->deviceId;
  event.stream_id = record->streamId;
  event.context_id = record->contextId;
  event.correlation_id = record->runtimeCorrelationId;
  // NOTE(liaotianju): all the float attribute should be initialized
  event.kernel_info.occupancy = 0.f;
  event.kernel_info.blocks_per_sm = 0.f;
  event.kernel_info.warps_per_sm = 0.f;
  profiler_add_device_trace_event(prof, &event);
}

const char *MemoryKind(uint8_t kind) {
  switch (kind) {
    case SDPTI_ACTIVITY_MEMORY_KIND_PAGEABLE:
      return "Pageable";
    case SDPTI_ACTIVITY_MEMORY_KIND_PINNED:
      return "Pinned";
    case SDPTI_ACTIVITY_MEMORY_KIND_DEVICE:
      return "Device";
    default:
      return "Unknown";
  }
}

std::string MemcpyKind(uint8_t memKind, uint8_t srcKind, uint8_t dstKind) {
  std::stringstream stream;
  switch (memKind) {
    case SDPTI_ACTIVITY_MEMCPY_KIND_HTOD:
      stream << "MEMCPY_HtoD (";
      stream << MemoryKind(srcKind) << ")";
      return stream.str();
    case SDPTI_ACTIVITY_MEMCPY_KIND_DTOH:
      stream << "MEMCPY_DtoH (";
      stream << MemoryKind(dstKind) << ")";
      return stream.str();
    case SDPTI_ACTIVITY_MEMCPY_KIND_DTOD:
      return "MEMCPY_DtoD";
    default:
      return "MEMCPY";
  }
}

void RecordEvent::AddMemcpyRecord(const SDpti_ActivityMemcpy *record) {
  phi::DeviceTraceEvent event;
  event.name = MemcpyKind(record->copyKind, record->srcKind, record->dstKind);
  event.type = phi::TracerEventType::Memcpy;
  event.start_ns = record->start;
  event.end_ns = record->end;
  event.device_id = record->deviceId;
  event.stream_id = record->streamId;
  event.context_id = record->contextId;
  event.correlation_id = record->runtimeCorrelationId;
  event.memcpy_info.num_bytes = record->bytes;
  snprintf(event.memcpy_info.src_kind,
           phi::kMemKindMaxLen,
           "%s",
           MemoryKind(record->srcKind));
  snprintf(event.memcpy_info.dst_kind,
           phi::kMemKindMaxLen,
           "%s",
           MemoryKind(record->dstKind));
  profiler_add_device_trace_event(prof, &event);
}

void RecordEvent::AddMemsetRecord(const SDpti_ActivityMemset *record) {
  phi::DeviceTraceEvent event;
  event.name = "MEMSET";
  event.type = phi::TracerEventType::Memset;
  event.start_ns = record->start;
  event.end_ns = record->end;
  event.device_id = record->deviceId;
  event.stream_id = record->streamId;
  event.context_id = record->contextId;
  event.correlation_id = record->runtimeCorrelationId;
  event.memset_info.num_bytes = record->bytes;
  snprintf(event.memset_info.memory_kind,
           phi::kMemKindMaxLen,
           "%s",
           MemoryKind(record->memoryKind));
  event.memset_info.value = record->value;
  profiler_add_device_trace_event(prof, &event);
}

inline const char *GetApiName(uint32_t callback_id) {
  const char *name;
  auto ret = custom_dynload::sdptiGetCallbackName(
      SDPTI_CB_DOMAIN_RUNTIME_API, callback_id, &name);
  return ret == SDPTI_SUCCESS ? name : "UNKNOWN";
}

void RecordEvent::AddApiRecord(
    const SDpti_ActivityAPI *record,
    const std::unordered_map<uint32_t, uint64_t> &tid_mapping) {
  phi::RuntimeTraceEvent event;
  event.name = GetApiName(record->cbid);
  event.start_ns = record->start;
  event.end_ns = record->end;
  event.process_id = record->processId;
  event.correlation_id = record->correlationId;
  uint64_t tid = 0;
  auto iter = tid_mapping.find(record->threadId);
  if (iter == tid_mapping.end()) {
    tid = gettid();
    VLOG(6) << "dismatched thread_id:" << record->threadId << "->" << tid;
  } else {
    tid = iter->second;
  }
  event.thread_id = tid;
  event.callback_id = record->cbid;
  profiler_add_runtime_trace_event(prof, &event);
}

void RecordEvent::AttributeDumpEnable() { should_dump_info_ = true; }

bool RecordEvent::GetAttributeDumpMode() { return should_dump_info_; }

void RecordEvent::ActivityEnable() { mode_ = true; }

void RecordEvent::ActivityDisable() { mode_ = false; }

bool RecordEvent::GetProfilerMode() { return mode_; }

bool RecordEvent::GetSdptiMode() { return sdpti_mode; }

void RecordEvent::SetSdptiMode(bool mode) { sdpti_mode = mode; }

void RecordEvent::SetProfPara(C_Profiler p, uint64_t tracing_start_ns_) {
  prof = p;
  tracing_start_ns = tracing_start_ns_;
}

RecordEvent::~RecordEvent() {}
