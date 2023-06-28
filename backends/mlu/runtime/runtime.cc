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
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "glog/logging.h"

namespace {

inline size_t get_devices_count() {
  uint32_t count = 0;
  PADDLE_ENFORCE_MLU_SUCCESS(cnrtGetDeviceCount(&count));
  return static_cast<size_t>(count);
}

}  // namespace

// Device
C_Status Init() {
  size_t dev_cnt = get_devices_count();
  return C_SUCCESS;
}

C_Status SetDevice(const C_Device device) {
  PADDLE_ENFORCE_MLU_SUCCESS(cnrtSetDevice(device->id));
  return C_SUCCESS;
}

C_Status GetDevice(const C_Device device) {
  PADDLE_ENFORCE_MLU_SUCCESS(cnrtGetDevice(&(device->id)));
  return C_SUCCESS;
}

C_Status SyncDevice(const C_Device device) {
  PADDLE_ENFORCE_MLU_SUCCESS(cnrtSyncDevice());
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

// Memory
C_Status MemCpyH2D(const C_Device device,
                   void *dst,
                   const void *src,
                   size_t size) {
  if (dst == nullptr && size == 0) return C_SUCCESS;
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnrtMemcpy(dst, const_cast<void *>(src), size, cnrtMemcpyHostToDev));
  return C_SUCCESS;
}

C_Status MemCpyD2D(const C_Device device,
                   void *dst,
                   const void *src,
                   size_t size) {
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnrtMemcpy(dst, const_cast<void *>(src), size, cnrtMemcpyDevToDev));
  return C_SUCCESS;
}

C_Status MemCpyD2H(const C_Device device,
                   void *dst,
                   const void *src,
                   size_t size) {
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnrtMemcpy(dst, const_cast<void *>(src), size, cnrtMemcpyDevToHost));
  return C_SUCCESS;
}

C_Status AsyncMemCpyH2D(const C_Device device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size) {
  PADDLE_ENFORCE_MLU_SUCCESS(cnrtMemcpyAsync(dst,
                                             const_cast<void *>(src),
                                             size,
                                             GetQueue(stream),
                                             cnrtMemcpyHostToDev));
  return C_SUCCESS;
}

C_Status AsyncMemCpyD2D(const C_Device device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size) {
  VLOG(3) << "AsyncMemCpyD2D: from " << src << " to " << dst << " size " << size
          << " stream " << stream;
  PADDLE_ENFORCE_MLU_SUCCESS(cnrtMemcpyAsync(dst,
                                             const_cast<void *>(src),
                                             size,
                                             GetQueue(stream),
                                             cnrtMemcpyDevToDev));
  return C_SUCCESS;
}

C_Status AsyncMemCpyD2H(const C_Device device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size) {
  PADDLE_ENFORCE_MLU_SUCCESS(cnrtMemcpyAsync(dst,
                                             const_cast<void *>(src),
                                             size,
                                             GetQueue(stream),
                                             cnrtMemcpyDevToHost));
  return C_SUCCESS;
}

C_Status Allocate(const C_Device device, void **ptr, size_t size) {
  PADDLE_ENFORCE_MLU_SUCCESS(cnrtSetDevice(device->id));
  PADDLE_ENFORCE_MLU_SUCCESS(cnrtMalloc(ptr, size));
  return C_SUCCESS;
}

C_Status Deallocate(const C_Device device, void *ptr, size_t size) {
  PADDLE_ENFORCE_MLU_SUCCESS(cnrtSetDevice(device->id));
  PADDLE_ENFORCE_MLU_SUCCESS(cnrtFree(ptr));
  return C_SUCCESS;
}

C_Status HostAllocate(const C_Device device, void **ptr, size_t size) {
  PADDLE_ENFORCE_MLU_SUCCESS(cnrtSetDevice(device->id));
  PADDLE_ENFORCE_MLU_SUCCESS(cnrtHostMalloc(ptr, size));
  return C_SUCCESS;
}

C_Status HostDeallocate(const C_Device device, void *ptr, size_t size) {
  PADDLE_ENFORCE_MLU_SUCCESS(cnrtFreeHost(ptr));
  return C_SUCCESS;
}

C_Status DeviceMemStats(const C_Device device,
                        size_t *total_memory,
                        size_t *free_memory) {
  PADDLE_ENFORCE_MLU_SUCCESS(cnrtMemGetInfo(free_memory, total_memory));
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

// Stream
C_Status CreateStream(const C_Device device, C_Stream *stream) {
  mluStream_t mlu_stream = new mluStream();

  cnrtQueue_t queue;
  PADDLE_ENFORCE_MLU_SUCCESS(cnrtQueueCreate(&queue));

  cnnlHandle_t handle;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlCreate(&handle));
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetQueue(handle, queue));

  mluOpHandle_t op_handle;
  PADDLE_ENFORCE_MLU_SUCCESS(mluOpCreate(&op_handle));
  PADDLE_ENFORCE_MLU_SUCCESS(mluOpSetQueue(op_handle, queue));

  mlu_stream->queue = queue;
  mlu_stream->handle = handle;
  mlu_stream->op_handle = op_handle;

  *stream = reinterpret_cast<C_Stream>(mlu_stream);

  return C_SUCCESS;
}

C_Status DestroyStream(const C_Device device, C_Stream stream) {
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlDestroy(GetHandle(stream)));
  PADDLE_ENFORCE_MLU_SUCCESS(mluOpDestroy(GetOpHandle(stream)));
  PADDLE_ENFORCE_MLU_SUCCESS(cnrtQueueDestroy(GetQueue(stream)));

  mluStream_t mlu_stream = reinterpret_cast<mluStream_t>(stream);
  delete[] mlu_stream;

  return C_SUCCESS;
}

C_Status SyncStream(const C_Device device, C_Stream stream) {
  PADDLE_ENFORCE_MLU_SUCCESS(cnrtQueueSync(GetQueue(stream)));
  return C_SUCCESS;
}

C_Status AddCallback(const C_Device device,
                     C_Stream stream,
                     C_Callback callback,
                     void *user_data) {
  PADDLE_ENFORCE_MLU_SUCCESS(cnrtInvokeHostFunc(
      GetQueue(stream), reinterpret_cast<cnrtHostFn_t>(callback), user_data));
  return C_SUCCESS;
}

C_Status StreamWaitEvent(const C_Device device,
                         C_Stream stream,
                         C_Event event) {
  PADDLE_ENFORCE_MLU_SUCCESS(cnrtQueueWaitNotifier(
      reinterpret_cast<cnrtNotifier_t>(event), GetQueue(stream), 0));
  return C_SUCCESS;
}

// Event
C_Status CreateEvent(const C_Device device, C_Event *event) {
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnrtNotifierCreate(reinterpret_cast<cnrtNotifier_t *>(event)));
  return C_SUCCESS;
}

C_Status DestroyEvent(const C_Device device, C_Event event) {
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnrtNotifierDestroy(reinterpret_cast<cnrtNotifier_t>(event)));
  return C_SUCCESS;
}

C_Status RecordEvent(const C_Device device, C_Stream stream, C_Event event) {
  PADDLE_ENFORCE_MLU_SUCCESS(cnrtPlaceNotifier(
      reinterpret_cast<cnrtNotifier_t>(event), GetQueue(stream)));
  return C_SUCCESS;
}

C_Status SyncEvent(const C_Device device, C_Event event) {
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnrtWaitNotifier(reinterpret_cast<cnrtNotifier_t>(event)));
  return C_SUCCESS;
}

// CNCL
namespace {

inline cnclDataType_t PDDataTypeToCnclDataType(C_DataType type) {
  if (type == C_DataType::FLOAT32) {
    return cnclFloat32;
  } else if (type == C_DataType::FLOAT16) {
    return cnclFloat16;
  } else if (type == C_DataType::INT32) {
    return cnclInt32;
  } else if (type == C_DataType::INT16) {
    return cnclInt16;
  } else if (type == C_DataType::INT8) {
    return cnclInt8;
  } else if (type == C_DataType::UINT8) {
    return cnclUint8;
  } else if (type == C_DataType::INT64) {
    return cnclInt32;
  } else {
    LOG(ERROR) << "Datatype " << type << " in cncl is not supported.";
  }
}

cnclReduceOp_t PDReduceOpToCnclReduceOp(C_CCLReduceOp op) {
  if (op == C_CCLReduceOp::MIN) {
    return cnclMin;
  } else if (op == C_CCLReduceOp::MAX) {
    return cnclMax;
  } else if (op == C_CCLReduceOp::SUM) {
    return cnclSum;
  } else if (op == C_CCLReduceOp::PRODUCT) {
    return cnclProd;
  } else {
    LOG(ERROR) << "Reduceop " << op << " in hccl is not supported.";
  }
}

}  // namespace

C_Status XcclGetUniqueIdSize(size_t *size) {
  *size = sizeof(cnclCliqueId);
  return C_SUCCESS;
}

C_Status XcclGetUniqueId(C_CCLRootId *unique_id) {
  if (unique_id->sz != sizeof(cnclCliqueId)) {
    LOG(ERROR) << "unique_id->sz must be equal sizeof(cnclCliqueId)";
    return C_FAILED;
  }
  VLOG(4) << "[CNCL] create clique.";
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnclGetCliqueId(reinterpret_cast<cnclCliqueId *>(unique_id->data)));
  VLOG(4) << "[CNCL] clique created: "
          << reinterpret_cast<cnclCliqueId *>(unique_id->data)->hash;
  return C_SUCCESS;
}

C_Status XcclCommInitRank(size_t nranks,
                          C_CCLRootId *unique_id,
                          size_t rank,
                          C_CCLComm *comm) {
  int dev_id;
  PADDLE_ENFORCE_MLU_SUCCESS(cnrtGetDevice(&dev_id));
  int dev_list[] = {dev_id};
  int rank_list[] = {rank};
  VLOG(4) << "[CNCL] create comm with rank: " << rank << " clique: "
          << reinterpret_cast<cnclCliqueId *>(unique_id->data)->hash;
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnclInitComms(reinterpret_cast<cnclComm_t *>(comm),
                    1,
                    dev_list,
                    rank_list,
                    nranks,
                    reinterpret_cast<cnclCliqueId *>(unique_id->data)));
  VLOG(4) << "[CNCL] comm inited: " << reinterpret_cast<cnclComm_t>(*comm);
  return C_SUCCESS;
}

C_Status XcclDestroyComm(C_CCLComm comm) {
  PADDLE_ENFORCE_MLU_SUCCESS(cnclFreeComm(reinterpret_cast<cnclComm_t>(comm)));
  return C_SUCCESS;
}

C_Status XcclAllReduce(void *send_buf,
                       void *recv_buf,
                       size_t count,
                       C_DataType data_type,
                       C_CCLReduceOp op,
                       C_CCLComm comm,
                       C_Stream stream) {
  PADDLE_ENFORCE_NE(data_type,
                    C_DataType::INT64,
                    phi::errors::InvalidArgument(
                        "The dtype of cncl reduce shouldn't be int64."));
  lastCommStream::Instance().Update(GetQueue(stream));
  PADDLE_ENFORCE_MLU_SUCCESS(cnclAllReduce(send_buf,
                                           recv_buf,
                                           count,
                                           PDDataTypeToCnclDataType(data_type),
                                           PDReduceOpToCnclReduceOp(op),
                                           reinterpret_cast<cnclComm_t>(comm),
                                           lastCommStream::Instance().get()));
  return C_SUCCESS;
}

C_Status XcclBroadcast(void *buf,
                       size_t count,
                       C_DataType data_type,
                       size_t root,
                       C_CCLComm comm,
                       C_Stream stream) {
  if (data_type == C_DataType::INT64) {
    // cncl does not support int64 dtype for now, use int32 as dtype and doulbe
    // THE count.
    count = count * 2;
  }
  lastCommStream::Instance().Update(GetQueue(stream));
  PADDLE_ENFORCE_MLU_SUCCESS(cnclBroadcast(buf,
                                           buf,
                                           count,
                                           PDDataTypeToCnclDataType(data_type),
                                           root,
                                           reinterpret_cast<cnclComm_t>(comm),
                                           lastCommStream::Instance().get()));
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
  VLOG(4) << "[CNCL] reduce.";
  lastCommStream::Instance().Update(GetQueue(stream));
  PADDLE_ENFORCE_MLU_SUCCESS(cnclReduce(send_buf,
                                        recv_buf,
                                        count,
                                        PDDataTypeToCnclDataType(data_type),
                                        PDReduceOpToCnclReduceOp(op),
                                        root,
                                        reinterpret_cast<cnclComm_t>(comm),
                                        lastCommStream::Instance().get()));
  return C_SUCCESS;
}

C_Status XcclAllGather(void *send_buf,
                       void *recv_buf,
                       size_t count,
                       C_DataType data_type,
                       C_CCLComm comm,
                       C_Stream stream) {
  if (data_type == C_DataType::INT64) {
    // cncl does not support int64 dtype for now, use int32 as dtype and doulbe
    // THE count.
    count = count * 2;
  }
  lastCommStream::Instance().Update(GetQueue(stream));
  PADDLE_ENFORCE_MLU_SUCCESS(cnclAllGather(send_buf,
                                           recv_buf,
                                           count,
                                           PDDataTypeToCnclDataType(data_type),
                                           reinterpret_cast<cnclComm_t>(comm),
                                           lastCommStream::Instance().get()));
  return C_SUCCESS;
}

C_Status XcclReduceScatter(void *send_buf,
                           void *recv_buf,
                           size_t count,
                           C_DataType data_type,
                           C_CCLReduceOp op,
                           C_CCLComm comm,
                           C_Stream stream) {
  PADDLE_ENFORCE_NE(data_type,
                    C_DataType::INT64,
                    phi::errors::InvalidArgument(
                        "The dtype of cncl reduce shouldn't be int64."));
  lastCommStream::Instance().Update(GetQueue(stream));
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnclReduceScatter(send_buf,
                        recv_buf,
                        count,
                        PDDataTypeToCnclDataType(data_type),
                        PDReduceOpToCnclReduceOp(op),
                        reinterpret_cast<cnclComm_t>(comm),
                        lastCommStream::Instance().get()));
  return C_SUCCESS;
}

C_Status XcclGroupStart() {
  LOG(ERROR) << "xccl_group_start is not supported on mlu device.";
  return C_ERROR;
}

C_Status XcclGroupEnd() {
  LOG(ERROR) << "xccl_group_end is not supported on mlu device.";
  return C_ERROR;
}

C_Status XcclSend(void *send_buf,
                  size_t count,
                  C_DataType data_type,
                  size_t dest_rank,
                  C_CCLComm comm,
                  C_Stream stream) {
  if (data_type == C_DataType::INT64) {
    // cncl does not support int64 dtype for now, use int32 as dtype and doulbe
    // THE count.
    count = count * 2;
  }
  lastCommStream::Instance().Update(GetQueue(stream));
  PADDLE_ENFORCE_MLU_SUCCESS(cnclSend(send_buf,
                                      count,
                                      PDDataTypeToCnclDataType(data_type),
                                      dest_rank,
                                      reinterpret_cast<cnclComm_t>(comm),
                                      lastCommStream::Instance().get()));
  return C_SUCCESS;
}

C_Status XcclRecv(void *recv_buf,
                  size_t count,
                  C_DataType data_type,
                  size_t src_rank,
                  C_CCLComm comm,
                  C_Stream stream) {
  if (data_type == C_DataType::INT64) {
    // cncl does not support int64 dtype for now, use int32 as dtype and doulbe
    // THE count.
    count = count * 2;
  }
  lastCommStream::Instance().Update(GetQueue(stream));
  PADDLE_ENFORCE_MLU_SUCCESS(cnclRecv(recv_buf,
                                      count,
                                      PDDataTypeToCnclDataType(data_type),
                                      src_rank,
                                      reinterpret_cast<cnclComm_t>(comm),
                                      lastCommStream::Instance().get()));
  return C_SUCCESS;
}

#ifdef WITH_PROFILE
void BufferRequestedCallback(uint64_t **buffer,
                             size_t *size,
                             size_t *max_num_records) {
  Tracer::Instance().AllocateBuffer(buffer, size);
  *max_num_records = 0;
}

void BufferCompletedCallback(uint64_t *buffer, size_t size, size_t valid_size) {
  Tracer::Instance().ProduceBuffer(buffer, valid_size);
}

std::unordered_map<uint32_t, uint64_t> CreateThreadIdMapping() {
  std::unordered_map<uint32_t, uint64_t> mapping;
  std::unordered_map<uint64_t, phi::ThreadId> ids = phi::GetAllThreadIds();
  for (const auto &id : ids) {
    mapping[id.second.cupti_tid] = id.second.sys_tid;
  }
  return mapping;
}

int ProcessCnpapiActivity(C_Profiler prof, uint64_t tracing_start_ns_) {
  int record_cnt = 0;
  CNPAPI_CALL(cnpapiActivityFlushAll());
  auto mapping = CreateThreadIdMapping();
  std::vector<ActivityBuffer> buffers = Tracer::Instance().ConsumeBuffers();
  for (auto &buffer : buffers) {
    if (buffer.addr == nullptr || buffer.valid_size == 0) {
      continue;
    }

    cnpapiActivity *record = nullptr;
    while (true) {
      cnpapiResult status =
          cnpapiActivityGetNextRecord(buffer.addr, buffer.valid_size, &record);
      if (status == CNPAPI_SUCCESS) {
        ProcessCnpapiActivityRecord(record, tracing_start_ns_, mapping, prof);
        ++record_cnt;
      } else if (status == CNPAPI_ERROR_MAX_LIMIT_REACHED) {
        break;
      } else {
        CNPAPI_CALL(status);
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
  CNPAPI_CALL(cnpapiRelease());
  return C_SUCCESS;
}

C_Status ProfilerPrepare(C_Profiler prof, void *user_data) {
  CNPAPI_CALL(cnpapiInit());
  CNPAPI_CALL(cnpapiActivityRegisterCallbacks(BufferRequestedCallback,
                                              BufferCompletedCallback));
  CNPAPI_CALL(cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_KERNEL));
  CNPAPI_CALL(cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_MEMCPY));
  CNPAPI_CALL(cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_MEMCPY_PTOP));
  CNPAPI_CALL(cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_MEMSET));
  CNPAPI_CALL(cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_CNDRV_API));
  VLOG(3) << "enable cnpapi activity";
  return C_SUCCESS;
}

C_Status ProfilerStart(C_Profiler prof, void *user_data) {
  Tracer::Instance().ConsumeBuffers();
  return C_SUCCESS;
}

C_Status ProfilerStop(C_Profiler prof, void *user_data) {
  CNPAPI_CALL(cnpapiActivityFlushAll());
  CNPAPI_CALL(cnpapiActivityDisable(CNPAPI_ACTIVITY_TYPE_KERNEL));
  CNPAPI_CALL(cnpapiActivityDisable(CNPAPI_ACTIVITY_TYPE_MEMCPY));
  CNPAPI_CALL(cnpapiActivityDisable(CNPAPI_ACTIVITY_TYPE_MEMCPY_PTOP));
  CNPAPI_CALL(cnpapiActivityDisable(CNPAPI_ACTIVITY_TYPE_MEMSET));
  CNPAPI_CALL(cnpapiActivityDisable(CNPAPI_ACTIVITY_TYPE_CNDRV_API));
  VLOG(3) << "disable cnpapi activity";
  return C_SUCCESS;
}

C_Status ProfilerCollectData(C_Profiler prof,
                             uint64_t tracing_start_ns_,
                             void *user_data) {
  ProcessCnpapiActivity(prof, tracing_start_ns_);
  return C_SUCCESS;
}
#endif

void InitPlugin(CustomRuntimeParams *params) {
  PADDLE_CUSTOM_RUNTIME_CHECK_VERSION(params);

  params->device_type = "mlu";
  params->sub_device_type = "none";

  memset(reinterpret_cast<void *>(params->interface),
         0,
         sizeof(C_DeviceInterface));

  // device
  params->interface->initialize = Init;
  params->interface->set_device = SetDevice;
  params->interface->get_device = GetDevice;
  params->interface->synchronize_device = SyncDevice;
  params->interface->get_device_count = GetDevicesCount;
  params->interface->get_device_list = GetDevicesList;

  // memory
  params->interface->memory_copy_h2d = MemCpyH2D;
  params->interface->memory_copy_d2d = MemCpyD2D;
  params->interface->memory_copy_d2h = MemCpyD2H;
  params->interface->memory_copy_p2p = nullptr;
  params->interface->async_memory_copy_h2d = AsyncMemCpyH2D;
  params->interface->async_memory_copy_d2d = AsyncMemCpyD2D;
  params->interface->async_memory_copy_d2h = AsyncMemCpyD2H;
  params->interface->async_memory_copy_p2p = nullptr;
  params->interface->device_memory_allocate = Allocate;
  params->interface->device_memory_deallocate = Deallocate;
  params->interface->host_memory_allocate = HostAllocate;
  params->interface->host_memory_deallocate = HostDeallocate;
  params->interface->device_memory_stats = DeviceMemStats;
  params->interface->device_min_chunk_size = DeviceMinChunkSize;
  params->interface->device_extra_padding_size = ExtraPaddingSize;

  // stream
  params->interface->create_stream = CreateStream;
  params->interface->destroy_stream = DestroyStream;
  params->interface->synchronize_stream = SyncStream;
  params->interface->stream_add_callback = AddCallback;
  params->interface->stream_wait_event = StreamWaitEvent;

  // event
  params->interface->create_event = CreateEvent;
  params->interface->destroy_event = DestroyEvent;
  params->interface->record_event = RecordEvent;
  params->interface->synchronize_event = SyncEvent;

  // cl
  params->interface->xccl_get_unique_id_size = XcclGetUniqueIdSize;
  params->interface->xccl_get_unique_id = XcclGetUniqueId;
  params->interface->xccl_comm_init_rank = XcclCommInitRank;
  params->interface->xccl_destroy_comm = XcclDestroyComm;
  params->interface->xccl_all_reduce = XcclAllReduce;
  params->interface->xccl_broadcast = XcclBroadcast;
  params->interface->xccl_reduce = XcclReduce;
  params->interface->xccl_all_gather = XcclAllGather;
  params->interface->xccl_reduce_scatter = XcclReduceScatter;
  params->interface->xccl_group_start = XcclGroupStart;
  params->interface->xccl_group_end = XcclGroupEnd;
  params->interface->xccl_send = XcclSend;
  params->interface->xccl_recv = XcclRecv;

  // profiler
#ifdef WITH_PROFILE
  params->interface->profiler_collect_trace_data = ProfilerCollectData;
  params->interface->profiler_initialize = ProfilerInitialize;
  params->interface->profiler_finalize = ProfilerFinalize;
  params->interface->profiler_start_tracing = ProfilerStart;
  params->interface->profiler_stop_tracing = ProfilerStop;
  params->interface->profiler_prepare_tracing = ProfilerPrepare;
#endif
}
