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

#pragma once

#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "dynload/sdpti.h"
#include "glog/logging.h"
#include "kernels/profiler/os_info.h"
#include "paddle/phi/api/profiler/trace_event.h"
#include "paddle/phi/backends/device_ext.h"
#include "sdaa_runtime.h"  // NOLINT

static const char *const COMMUNICATION_KERNEL_PREFIX = "xccl";

struct ActivityBuffer {
  ActivityBuffer(uint8_t *addr, size_t size) : addr(addr), valid_size(size) {}
  uint8_t *addr;
  size_t valid_size;
};

struct TraceEvent {
  TraceEvent() = default;
  TraceEvent(const std::string &name,
             uint64_t start_ns,
             uint64_t end_ns,
             uint64_t device_id,
             uint64_t thread_id,
             uint32_t correlation_id = 0,
             const std::string &msg = "")
      : name(name),
        start_ns(start_ns),
        end_ns(end_ns),
        device_id(device_id),
        thread_id(thread_id),
        correlation_id(correlation_id),
        msg(msg) {}
  std::string name;
  uint64_t start_ns;
  uint64_t end_ns;
  uint64_t device_id;
  uint64_t thread_id;
  uint32_t correlation_id;
  std::string msg;
};

std::unordered_map<uint32_t, uint64_t> CreateThreadIdMapping();

class RecordEvent {
 public:
  static RecordEvent &Instance() {
    static RecordEvent rec;
    return rec;
  }

  RecordEvent()
      : records_num(0),
        mode_(false),
        should_dump_info_(false),
        sdpti_mode(true),
        correlation_id(0),
        tracing_start_ns(-1),
        buffer{nullptr},
        prof{nullptr} {}

  void init(const TraceEvent &event);

  void AllocateSdptiBuffer(uint8_t **buffer, size_t *size);

  void ReleaseSdptiBuffer(uint8_t *buffer);

  uint32_t processName(const std::string &name);

  uint64_t getThread();

  uint64_t getBufferSize();

  void BufferRequested();

  void BufferCompleted();

  bool ActivityGetNextRecord(uint8_t *buffer, TraceEvent *record);

  void ProcessActivityRecord(
      const SDpti_Activity *record,
      const std::unordered_map<uint32_t, uint64_t> &tid_mapping);

  void ProcessKernelRecord(const TraceEvent *record);

  void AddKernelRecord(const SDpti_ActivityKernel *record);

  void AddMemcpyRecord(const SDpti_ActivityMemcpy *record);

  void AddMemsetRecord(const SDpti_ActivityMemset *record);

  void AddApiRecord(const SDpti_ActivityAPI *record,
                    const std::unordered_map<uint32_t, uint64_t> &tid_mapping);

  void AddMemcpyP2PRecord(const SDpti_ActivityMemcpyP2P *record);

  void AddMemOpsRecord(const SDpti_ActivityMemOps *record);

  void ActivityEnable();

  void ActivityDisable();

  void AttributeDumpEnable();

  bool GetAttributeDumpMode();

  void SetProfPara(C_Profiler p, uint64_t tracing_start_ns_);

  bool GetProfilerMode();

  bool GetSdptiMode();

  void SetSdptiMode(bool mode);

  ~RecordEvent();

 private:
  std::mutex mutex_;
  uint8_t *buffer;
  size_t records_num;
  bool mode_;
  bool sdpti_mode;
  bool should_dump_info_;
  uint32_t correlation_id;
  std::vector<std::string> namelist;
  std::unordered_map<std::string, int> name_index;
  std::deque<std::unique_ptr<TraceEvent>> trace_events;
  C_Profiler prof;
  uint64_t tracing_start_ns;

  template <class... Args>
  void EmplaceEvent(Args &&...args) {
    trace_events.emplace_back(
        std::make_unique<TraceEvent>(std::forward<Args>(args)...));
  }
};
