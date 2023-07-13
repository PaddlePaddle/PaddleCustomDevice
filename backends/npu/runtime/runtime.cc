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

#include "runtime/runtime.h"

#include <cstring>
#include <iostream>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "glog/logging.h"
#include "runtime/flags.h"

FLAGS_DEFINE_string(npu_profiling_dir,
                    "ascend_profiling",
                    "ACL profiling output dir");
FLAGS_DEFINE_uint64(npu_profiling_dtypes,
                    ACL_PROF_ACL_API | ACL_PROF_TASK_TIME |
                        ACL_PROF_AICORE_METRICS | ACL_PROF_AICPU |
                        ACL_PROF_HCCL_TRACE | ACL_PROF_RUNTIME_API,
                    "ACL datatypes to profile");
FLAGS_DEFINE_uint64(npu_profiling_metrics,
                    static_cast<uint64_t>(ACL_AICORE_ARITHMETIC_UTILIZATION),
                    "AI Core metric to profile");

thread_local int g_current_device_id(-1);

aclrtStream SecondaryStream::Get(aclrtStream aicore_stream) {
  RUN_CHECK(aicpu_streams.find(aicore_stream) != aicpu_streams.cend());
  return aicpu_streams[aicore_stream];
}

void SecondaryStream::Create(aclrtStream aicore_stream) {
  RUN_CHECK(aicpu_streams.find(aicore_stream) == aicpu_streams.cend());
  aclrtStream aicpu_stream;
  ACL_CHECK(aclrtCreateStream(&aicpu_stream));
  aicpu_streams[aicore_stream] = aicpu_stream;
}

void SecondaryStream::Destroy(aclrtStream aicore_stream) {
  RUN_CHECK(aicpu_streams.find(aicore_stream) != aicpu_streams.cend());
  ACL_CHECK(aclrtDestroyStream(aicpu_streams[aicore_stream]));
  aicpu_streams.erase(aicore_stream);
}

void SecondaryStream::RecordBefore(aclrtStream aicore_stream) {
  static std::list<aclrtEvent> events;

  RUN_CHECK(aicpu_streams.find(aicore_stream) != aicpu_streams.cend());
  auto aicpu_stream = aicpu_streams[aicore_stream];

  for (auto iter = events.begin(); iter != events.end();) {
    auto event = *iter;
    aclrtEventRecordedStatus status = ACL_EVENT_RECORDED_STATUS_COMPLETE;
    ACL_CHECK(aclrtQueryEventStatus(event, &status));
    if (status == ACL_EVENT_RECORDED_STATUS_COMPLETE) {
      ACL_CHECK(aclrtDestroyEvent(event));
      iter = events.erase(iter);
    } else {
      ++iter;
    }
  }
  {
    aclrtEvent event;
    ACL_CHECK(aclrtCreateEvent(&event));
    ACL_CHECK(aclrtRecordEvent(event, aicpu_stream));
    ACL_CHECK(aclrtStreamWaitEvent(aicore_stream, event));
    events.push_back(event);
  }
}

void SecondaryStream::RecordAfter(aclrtStream aicore_stream) {
  static std::list<aclrtEvent> events;

  RUN_CHECK(aicpu_streams.find(aicore_stream) != aicpu_streams.cend());
  auto aicpu_stream = aicpu_streams[aicore_stream];

  for (auto iter = events.begin(); iter != events.end();) {
    auto event = *iter;
    aclrtEventRecordedStatus status = ACL_EVENT_RECORDED_STATUS_COMPLETE;
    ACL_CHECK(aclrtQueryEventStatus(event, &status));
    if (status == ACL_EVENT_RECORDED_STATUS_COMPLETE) {
      ACL_CHECK(aclrtDestroyEvent(event));
      iter = events.erase(iter);
    } else {
      ++iter;
    }
  }
  {
    aclrtEvent event;
    ACL_CHECK(aclrtCreateEvent(&event));
    ACL_CHECK(aclrtRecordEvent(event, aicore_stream));
    ACL_CHECK(aclrtStreamWaitEvent(aicpu_stream, event));
    events.push_back(event);
  }
}

class AlignnedAllocator {
 public:
  void *Alloc(size_t size, size_t align) {
    std::lock_guard<std::mutex> lock(mtx_);
    ProcessEvents();
    void *p = malloc(size + align);
    void *ret =
        reinterpret_cast<void *>(reinterpret_cast<size_t>(p) + align -
                                 (reinterpret_cast<size_t>(p) & (align - 1)));
    recorded_events_[ret] = {p, nullptr};
    return ret;
  }

  void Record(void *p, aclrtEvent event) {
    std::lock_guard<std::mutex> lock(mtx_);
    recorded_events_[p].second = event;
  }

  void ClearEvent() {
    std::lock_guard<std::mutex> lock(mtx_);
    for (auto it = recorded_events_.begin(); it != recorded_events_.end();) {
      aclrtEvent event = it->second.second;
      if (!event) continue;
      ACL_CHECK(aclrtSynchronizeEvent(event));
      void *ptr = it->second.first;
      free(ptr);
      ACL_CHECK(aclrtDestroyEvent(event));
      it = recorded_events_.erase(it);
    }
  }

  void ProcessEvents() {
    for (auto it = recorded_events_.begin(); it != recorded_events_.end();) {
      aclrtEvent event = it->second.second;
      if (!event) continue;
      aclrtEventRecordedStatus status = ACL_EVENT_RECORDED_STATUS_COMPLETE;
      ACL_CHECK(aclrtQueryEventStatus(event, &status));
      if (status == ACL_EVENT_RECORDED_STATUS_COMPLETE) {
        void *ptr = it->second.first;
        free(ptr);
        it = recorded_events_.erase(it);
        ACL_CHECK(aclrtDestroyEvent(event));
      } else {
        ++it;
      }
    }
  }

 private:
  std::unordered_map<void *, std::pair<void *, aclrtEvent>> recorded_events_;
  std::mutex mtx_;
};

class AlignnedAllocatorList {
 public:
  explicit AlignnedAllocatorList(size_t device_count)
      : allocator_list(device_count, nullptr) {}

  void Init(size_t dev_id) { allocator_list[dev_id] = new AlignnedAllocator; }

  void Deinit(size_t dev_id) {
    delete allocator_list[dev_id];
    allocator_list[dev_id] = nullptr;
  }

  AlignnedAllocator *GetAllocator(size_t dev_id) {
    return allocator_list[dev_id];
  }

 private:
  std::vector<AlignnedAllocator *> allocator_list;
};

static AlignnedAllocatorList *global_allocator_list = nullptr;

inline void check_uninitialized_thread(int dev_id) {
  if (g_current_device_id == -1) {
    g_current_device_id = dev_id;
    ACL_CHECK(aclrtSetDevice(dev_id));
  }
}

inline size_t get_current_device_id() {
  check_uninitialized_thread(0);
  ACL_CHECK(aclrtGetDevice(&g_current_device_id));
  return g_current_device_id;
}

inline size_t get_devices_count() {
  uint32_t count = 0;
  aclrtGetDeviceCount(&count);
  return static_cast<size_t>(count);
}

C_Status Init() {
  ACL_CHECK(aclInit(nullptr));
  size_t count = get_devices_count();
  if (count) {
    global_allocator_list = new AlignnedAllocatorList(count);
  }
  return C_SUCCESS;
}

C_Status InitDevice(const C_Device device) {
  ACL_CHECK(aclrtSetDevice(device->id));
  if (global_allocator_list) {
    global_allocator_list->Init(device->id);
  }
  return C_SUCCESS;
}

C_Status SetDevice(const C_Device device) {
  if (g_current_device_id != device->id) {
    ACL_CHECK(aclrtSetDevice(device->id));
    g_current_device_id = device->id;
  }
  return C_SUCCESS;
}

C_Status GetDevice(const C_Device device) {
  device->id = get_current_device_id();
  return C_SUCCESS;
}

C_Status ReleaseDevice(const C_Device device) {
  ACL_CHECK(aclrtSetDevice(device->id));
  if (global_allocator_list) {
    // global_allocator_list->GetAllocator(device->id)->ClearEvent();
    global_allocator_list->Deinit(device->id);
  }
  ACL_CHECK(aclrtResetDevice(device->id));
  return C_SUCCESS;
}

C_Status Finalize() {
  if (global_allocator_list) {
    delete global_allocator_list;
    global_allocator_list = nullptr;
  }
  ACL_CHECK(aclFinalize());
  return C_SUCCESS;
}

C_Status GetDevicesCount(size_t *count) {
  *count = get_devices_count();
  return C_SUCCESS;
}

C_Status GetDevicesList(size_t *device) {
  size_t count = get_devices_count();
  for (size_t dev_id = 0; dev_id < count; dev_id++) {
    device[dev_id] = dev_id;
  }
  return C_SUCCESS;
}

C_Status MemCpyH2D(const C_Device device,
                   void *dst,
                   const void *src,
                   size_t size) {
  if (dst == nullptr && size == 0) return C_SUCCESS;
  ACL_CHECK(aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_HOST_TO_DEVICE));
  return C_SUCCESS;
}

C_Status MemCpyD2D(const C_Device device,
                   void *dst,
                   const void *src,
                   size_t size) {
  ACL_CHECK(aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_DEVICE));
  return C_SUCCESS;
}

C_Status MemCpyD2H(const C_Device device,
                   void *dst,
                   const void *src,
                   size_t size) {
  ACL_CHECK(aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST));
  return C_SUCCESS;
}

// NOTE(wangran16):https://support.huaweicloud.com/aclcppdevg-cann504alpha2infer/aclcppdevg_01_0061.html
C_Status AsyncMemCpyH2D(const C_Device device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size) {
  auto allocator = global_allocator_list->GetAllocator(get_current_device_id());
  void *tmp = allocator->Alloc(size, 64);
  aclrtEvent event;
  ACL_CHECK(aclrtCreateEvent(&event));
  memcpy(tmp, src, size);
  ACL_CHECK(aclrtMemcpyAsync(
      dst, size, tmp, size, ACL_MEMCPY_HOST_TO_DEVICE, (aclrtStream)(stream)));
  ACL_CHECK(aclrtRecordEvent(event, (aclrtStream)(stream)));
  allocator->Record(tmp, event);
  return C_SUCCESS;
}

C_Status AsyncMemCpyD2D(const C_Device device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size) {
  ACL_CHECK(aclrtMemcpyAsync(dst,
                             size,
                             src,
                             size,
                             ACL_MEMCPY_DEVICE_TO_DEVICE,
                             (aclrtStream)(stream)));
  return C_SUCCESS;
}

C_Status AsyncMemCpyD2H(const C_Device device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size) {
  if (device) {
    check_uninitialized_thread(device->id);
  }
  ACL_CHECK(aclrtMemcpyAsync(
      dst, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST, (aclrtStream)(stream)));
  return C_SUCCESS;
}

C_Status Allocate(const C_Device device, void **ptr, size_t size) {
  SetDevice(device);
  void *data;
  aclrtMalloc(&data, size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (data) {
    *ptr = data;
    return C_SUCCESS;
  } else {
    *ptr = nullptr;
  }
  return C_FAILED;
}

C_Status HostAllocate(const C_Device device, void **ptr, size_t size) {
  void *data = nullptr;
  ACL_CHECK(aclrtMallocHost(&data, size));
  if (data) {
    *ptr = data;
    return C_SUCCESS;
  } else {
    *ptr = nullptr;
  }
  return C_FAILED;
}

C_Status Deallocate(const C_Device device, void *ptr, size_t size) {
  SetDevice(device);
  ACL_CHECK(aclrtFree(ptr));
  return C_SUCCESS;
}

C_Status HostDeallocate(const C_Device device, void *ptr, size_t size) {
  ACL_CHECK(aclrtFreeHost(ptr));
  return C_SUCCESS;
}

C_Status CreateStream(const C_Device device, C_Stream *stream) {
  ACL_CHECK(aclrtCreateStream(reinterpret_cast<aclrtStream *>(stream)));
  SecondaryStream::Instance().Create(*reinterpret_cast<aclrtStream *>(stream));
  return C_SUCCESS;
}

C_Status DestroyStream(const C_Device device, C_Stream stream) {
  ACL_CHECK(aclrtDestroyStream(reinterpret_cast<aclrtStream>(stream)));
  SecondaryStream::Instance().Destroy(reinterpret_cast<aclrtStream>(stream));
  return C_SUCCESS;
}

C_Status CreateEvent(const C_Device device, C_Event *event) {
  ACL_CHECK(aclrtCreateEvent(reinterpret_cast<aclrtEvent *>(event)));
  return C_SUCCESS;
}

C_Status RecordEvent(const C_Device device, C_Stream stream, C_Event event) {
  ACL_CHECK(aclrtRecordEvent(reinterpret_cast<aclrtEvent *>(event),
                             reinterpret_cast<aclrtStream>(stream)));
  return C_SUCCESS;
}

C_Status DestroyEvent(const C_Device device, C_Event event) {
  ACL_CHECK(aclrtDestroyEvent(reinterpret_cast<aclrtEvent>(event)));
  return C_SUCCESS;
}

C_Status SyncDevice(const C_Device device) {
  ACL_CHECK(aclrtSynchronizeDevice());
  return C_SUCCESS;
}

C_Status SyncStream(const C_Device device, C_Stream stream) {
  ACL_CHECK(aclrtSynchronizeStream(reinterpret_cast<aclrtStream>(stream)));
  return C_SUCCESS;
}

C_Status SyncEvent(const C_Device device, C_Event event) {
  ACL_CHECK(aclrtSynchronizeEvent(reinterpret_cast<aclrtEvent>(event)));
  return C_SUCCESS;
}

C_Status StreamWaitEvent(const C_Device device,
                         C_Stream stream,
                         C_Event event) {
  ACL_CHECK(aclrtStreamWaitEvent(reinterpret_cast<aclrtStream>(stream),
                                 reinterpret_cast<aclrtEvent>(event)));
  return C_SUCCESS;
}

C_Status AddCallback(const C_Device device,
                     C_Stream stream,
                     C_Callback callback,
                     void *user_data) {
  C_Status ret = C_SUCCESS;
  callback(device, stream, user_data, &ret);
  return ret;
}

C_Status DeviceMemStats(const C_Device device,
                        size_t *total_memory,
                        size_t *free_memory) {
  aclrtGetMemInfo(ACL_HBM_MEM, free_memory, total_memory);
  return C_SUCCESS;
}

C_Status DeviceMinChunkSize(const C_Device device, size_t *size) {
  *size = 512;
  return C_SUCCESS;
}

C_Status ExtraPaddingSize(const C_Device device, size_t *size) {
  *size = 32;
  return C_SUCCESS;
}

// CCL
HcclDataType PDDataTypeToHcclDataType(C_DataType dtype) {
  if (dtype == C_DataType::FLOAT64) {
    return HCCL_DATA_TYPE_FP64;
  } else if (dtype == C_DataType::FLOAT32) {
    return HCCL_DATA_TYPE_FP32;
  } else if (dtype == C_DataType::FLOAT16) {
    return HCCL_DATA_TYPE_FP16;
  } else if (dtype == C_DataType::INT64) {
    return HCCL_DATA_TYPE_INT64;
  } else if (dtype == C_DataType::INT32) {
    return HCCL_DATA_TYPE_INT32;
  } else if (dtype == C_DataType::INT8) {
    return HCCL_DATA_TYPE_INT8;
  } else if (dtype == C_DataType::UINT8) {
    return HCCL_DATA_TYPE_UINT8;
  } else {
    LOG(ERROR) << "Datatype " << dtype << " in hccl is not supported.";
  }
}

HcclReduceOp PDReduceOpToHcclReduceOp(C_CCLReduceOp op) {
  if (op == C_CCLReduceOp::MIN) {
    return HCCL_REDUCE_MIN;
  } else if (op == C_CCLReduceOp::MAX) {
    return HCCL_REDUCE_MAX;
  } else if (op == C_CCLReduceOp::SUM) {
    return HCCL_REDUCE_SUM;
  } else if (op == C_CCLReduceOp::PRODUCT) {
    return HCCL_REDUCE_PROD;
  } else {
    LOG(ERROR) << "Reduceop " << op << " in hccl is not supported.";
  }
}

C_Status XcclGetUniqueIdSize(size_t *size) {
  *size = sizeof(HcclRootInfo);
  return C_SUCCESS;
}

C_Status XcclGetUniqueId(C_CCLRootId *unique_id) {
  if (unique_id->sz != sizeof(HcclRootInfo)) {
    LOG(ERROR) << "unique_id->sz must be equal sizeof(HcclRootInfo)";
    return C_FAILED;
  }
  HCCL_CHECK(
      HcclGetRootInfo(reinterpret_cast<HcclRootInfo *>(unique_id->data)));
  return C_SUCCESS;
}

C_Status XcclCommInitRank(size_t nranks,
                          C_CCLRootId *unique_id,
                          size_t rank,
                          C_CCLComm *comm) {
  HCCL_CHECK(
      HcclCommInitRootInfo(nranks,
                           reinterpret_cast<HcclRootInfo *>(unique_id->data),
                           rank,
                           reinterpret_cast<HcclComm *>(comm)));
  return C_SUCCESS;
}

C_Status XcclDestroyComm(C_CCLComm comm) {
  HCCL_CHECK(HcclCommDestroy(reinterpret_cast<HcclComm>(comm)));
  return C_SUCCESS;
}

C_Status XcclAllReduce(void *send_buf,
                       void *recv_buf,
                       size_t count,
                       C_DataType data_type,
                       C_CCLReduceOp op,
                       C_CCLComm comm,
                       C_Stream stream) {
  HCCL_CHECK(HcclAllReduce(send_buf,
                           recv_buf,
                           count,
                           PDDataTypeToHcclDataType(data_type),
                           PDReduceOpToHcclReduceOp(op),
                           reinterpret_cast<HcclComm>(comm),
                           reinterpret_cast<aclrtStream>(stream)));
  return C_SUCCESS;
}

C_Status XcclBroadcast(void *buf,
                       size_t count,
                       C_DataType data_type,
                       size_t root,
                       C_CCLComm comm,
                       C_Stream stream) {
  HCCL_CHECK(HcclBroadcast(buf,
                           count,
                           PDDataTypeToHcclDataType(data_type),
                           static_cast<uint32_t>(root),
                           reinterpret_cast<HcclComm>(comm),
                           reinterpret_cast<aclrtStream>(stream)));
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
  HCCL_CHECK(HcclReduce(send_buf,
                        recv_buf,
                        count,
                        PDDataTypeToHcclDataType(data_type),
                        PDReduceOpToHcclReduceOp(op),
                        root,
                        comm,
                        stream));
  return C_SUCCESS;
}

C_Status XcclAllGather(void *send_buf,
                       void *recv_buf,
                       size_t count,
                       C_DataType data_type,
                       C_CCLComm comm,
                       C_Stream stream) {
  HCCL_CHECK(HcclAllGather(send_buf,
                           recv_buf,
                           count,
                           PDDataTypeToHcclDataType(data_type),
                           reinterpret_cast<HcclComm>(comm),
                           reinterpret_cast<aclrtStream>(stream)));
  return C_SUCCESS;
}

C_Status XcclReduceScatter(void *send_buf,
                           void *recv_buf,
                           size_t count,
                           C_DataType data_type,
                           C_CCLReduceOp op,
                           C_CCLComm comm,
                           C_Stream stream) {
  HCCL_CHECK(HcclReduceScatter(send_buf,
                               recv_buf,
                               count,
                               PDDataTypeToHcclDataType(data_type),
                               PDReduceOpToHcclReduceOp(op),
                               reinterpret_cast<HcclComm>(comm),
                               reinterpret_cast<aclrtStream>(stream)));
  return C_SUCCESS;
}

C_Status XcclGroupStart() {
  LOG(ERROR) << "xccl_group_start is not supported on ascend npu device.";
  return C_ERROR;
}

C_Status XcclGroupEnd() {
  LOG(ERROR) << "xccl_group_end is not supported on ascend npu device.";
  return C_ERROR;
}

C_Status XcclSend(void *send_buf,
                  size_t count,
                  C_DataType data_type,
                  size_t dest_rank,
                  C_CCLComm comm,
                  C_Stream stream) {
  HCCL_CHECK(HcclSend(send_buf,
                      count,
                      PDDataTypeToHcclDataType(data_type),
                      static_cast<uint32_t>(dest_rank),
                      reinterpret_cast<HcclComm>(comm),
                      reinterpret_cast<aclrtStream>(stream)));
  return C_SUCCESS;
}

C_Status XcclRecv(void *recv_buf,
                  size_t count,
                  C_DataType data_type,
                  size_t src_rank,
                  C_CCLComm comm,
                  C_Stream stream) {
  HCCL_CHECK(HcclRecv(recv_buf,
                      count,
                      PDDataTypeToHcclDataType(data_type),
                      static_cast<uint32_t>(src_rank),
                      reinterpret_cast<HcclComm>(comm),
                      reinterpret_cast<aclrtStream>(stream)));
  return C_SUCCESS;
}

C_Status ProfilerInitialize(C_Profiler prof, void **user_data) {
  // NOTE(wangran16):
  // https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/60RC1alpha001/infacldevg/aclcppdevg/aclcppdevg_03_0784.html
  VLOG(1) << "Init NPU Profiling, FLAGS_npu_profiling_dir: "
          << FLAGS_npu_profiling_dir
          << ", FLAGS_npu_profiling_dtypes: " << FLAGS_npu_profiling_dtypes
          << ", FLAGS_npu_profiling_metrics: " << FLAGS_npu_profiling_metrics;
  std::vector<uint32_t> device_ids(
      {static_cast<uint32_t>(get_current_device_id())});
  AscendProfiler::Instance().update_config(
      device_ids,
      static_cast<aclprofAicoreMetrics>(FLAGS_npu_profiling_metrics),
      nullptr,
      FLAGS_npu_profiling_dtypes);
  ACL_CHECK(aclprofInit(FLAGS_npu_profiling_dir.c_str(),
                        FLAGS_npu_profiling_dir.size()));
  LOG(INFO) << "ascend profiling data will be saved in "
            << FLAGS_npu_profiling_dir;
  return C_SUCCESS;
}

C_Status ProfilerFinalize(C_Profiler prof, void *user_data) {
  AscendProfiler::Instance().stop();
  AscendProfiler::Instance().destroy_config();
  // ACL_CHECK(aclprofFinalize());
  return C_SUCCESS;
}

C_Status ProfilerPrepare(C_Profiler prof, void *user_data) { return C_SUCCESS; }

C_Status ProfilerStart(C_Profiler prof, void *user_data) {
  AscendProfiler::Instance().start();
  return C_SUCCESS;
}

C_Status ProfilerStop(C_Profiler prof, void *user_data) {
  AscendProfiler::Instance().stop();
  return C_SUCCESS;
}

C_Status ProfilerCollectData(C_Profiler prof,
                             uint64_t tracing_start_ns_,
                             void *user_data) {
  return C_SUCCESS;
}

void InitPlugin(CustomRuntimeParams *params) {
  PADDLE_CUSTOM_RUNTIME_CHECK_VERSION(params);
  memset(reinterpret_cast<void *>(params->interface),
         0,
         sizeof(C_DeviceInterface));

  params->device_type = "npu";
  params->sub_device_type = "Ascend910";

  params->interface->initialize = Init;
  params->interface->finalize = Finalize;

  params->interface->init_device = InitDevice;
  params->interface->set_device = SetDevice;
  params->interface->get_device = GetDevice;
  params->interface->deinit_device = ReleaseDevice;

  params->interface->create_stream = CreateStream;
  params->interface->destroy_stream = DestroyStream;

  params->interface->create_event = CreateEvent;
  params->interface->destroy_event = DestroyEvent;
  params->interface->record_event = RecordEvent;

  params->interface->synchronize_device = SyncDevice;
  params->interface->synchronize_stream = SyncStream;
  params->interface->synchronize_event = SyncEvent;
  params->interface->stream_wait_event = StreamWaitEvent;
  params->interface->stream_add_callback = AddCallback;

  params->interface->memory_copy_h2d = MemCpyH2D;
  params->interface->memory_copy_d2d = MemCpyD2D;
  params->interface->memory_copy_d2h = MemCpyD2H;
  params->interface->memory_copy_p2p = nullptr;
  params->interface->async_memory_copy_h2d = AsyncMemCpyH2D;
  params->interface->async_memory_copy_d2d = AsyncMemCpyD2D;
  params->interface->async_memory_copy_d2h = AsyncMemCpyD2H;
  params->interface->async_memory_copy_p2p = nullptr;
  params->interface->device_memory_allocate = Allocate;
  params->interface->host_memory_allocate = HostAllocate;
  params->interface->device_memory_deallocate = Deallocate;
  params->interface->host_memory_deallocate = HostDeallocate;

  params->interface->get_device_count = GetDevicesCount;
  params->interface->get_device_list = GetDevicesList;
  params->interface->device_memory_stats = DeviceMemStats;

  params->interface->device_min_chunk_size = DeviceMinChunkSize;
  params->interface->device_extra_padding_size = ExtraPaddingSize;

  // xccl
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

  // profiler
  params->interface->profiler_collect_trace_data = ProfilerCollectData;
  params->interface->profiler_initialize = ProfilerInitialize;
  params->interface->profiler_finalize = ProfilerFinalize;
  params->interface->profiler_start_tracing = ProfilerStart;
  params->interface->profiler_stop_tracing = ProfilerStop;
  params->interface->profiler_prepare_tracing = ProfilerPrepare;
}
