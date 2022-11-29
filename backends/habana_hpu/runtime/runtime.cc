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

#include <errno.h>
#include <fcntl.h>
#include <semaphore.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <iostream>
#include <chrono>
#include <string>

#include "paddle/phi/backends/device_ext.h"

#include "utils/utils.h"
#include "utils/hpu_helper.h"

#define MEMORY_FRACTION 0.5f

static int global_current_device = 0;
static C_Stream g_stream = nullptr;

C_Status Init() {
  FUNCALL_LOG
  std::cout << "habana_hpu plugin compiled with ";
#ifdef __clang__
  std::cout << "clang\n";
#else
  std::cout << "gcc\n";
#endif

  synStatus status = synFail;
  status = synInitialize();
  assert(status == synSuccess && "synapse initialization failed");
  return C_SUCCESS;
}

C_Status InitDevice(const C_Device device) {
  FUNCALL_LOG
  uint32_t deviceId = 0;
  synDeviceType deviceType = synDeviceGaudi;
  bool deviceAcquired = waitForIdleDevice(&deviceId, /* INOUT */ deviceType, 60);

  LOG(INFO) << "requested device id " << device->id << ", real device id " << deviceId;
  global_current_device = deviceId;
  device->id = deviceId;

  return C_SUCCESS;
}

C_Status SetDevice(const C_Device device) {
  FUNCALL_LOG
  //nothing to do
  global_current_device = device->id;
  return C_SUCCESS;
}

C_Status GetDevice(const C_Device device) {
  FUNCALL_LOG
  //nothing to do
  device->id = global_current_device;
  return C_SUCCESS;
}

C_Status DestroyDevice(const C_Device device) {
  FUNCALL_LOG

  LOG(INFO) << device->id;

  synStatus status = synDeviceRelease(device->id);
  assert(status == synSuccess && "synDeviceRelease failed!");
  return C_SUCCESS; 
}

C_Status Finalize() {
  FUNCALL_LOG

  // synDestroy();
  return C_SUCCESS; 
}

C_Status GetDevicesCount(size_t *count) {
  FUNCALL_LOG
  uint32_t pCount = 0;
    LOG(INFO) << "get real device count " << pCount;

  synStatus status = synDeviceGetCount(&pCount);
  LOG(INFO) << "get real device count " << pCount;
  assert(status == synSuccess && "synDeviceGetCount failed!");
  //currently only expose 1 device 
  *count = 1;
  return C_SUCCESS;
}

C_Status GetDevicesList(size_t *devices) {
  FUNCALL_LOG
  //TODO: suse HABANA_VISIBLE_DEVICES to get available device
  devices[0] = 0;
  // devices[1] = 1;
  return C_SUCCESS;
}

C_Status h2dMemCpy(const C_Device device,
                void *dst,
                const void *src,
                size_t size) {
  FUNCALL_LOG
  LOG(INFO) << dst << " " << src << " " << size;
  synStatus status = HostMap(device->id, size, src);
  assert(status == synSuccess && "synHostMap failed!");
  status = synMemCopyAsync(g_stream->memcpyStreamHostToDev, reinterpret_cast<uint64_t>(src), size, reinterpret_cast<uint64_t>(dst), HOST_TO_DRAM);
  assert(status == synSuccess && "synMemCopyAsync failed!");
  status = synStreamSynchronize(g_stream->memcpyStreamHostToDev);
  assert(status == synSuccess && "synStreamSynchronize failed!");
  status = HostUnmap(device->id, src);
  assert(status == synSuccess && "synHostUnmap failed!");
  return C_SUCCESS;
}

C_Status d2hMemCpy(const C_Device device,
                void *dst,
                const void *src,
                size_t size) {
  FUNCALL_LOG

  LOG(INFO) << dst << " " << src << " " << size;
  synStatus status = HostMap(device->id, size, dst);
  assert(status == synSuccess && "synHostMap failed!");
  status = synMemCopyAsync(g_stream->memcpyStreamDevToHost, reinterpret_cast<uint64_t>(src), size, reinterpret_cast<uint64_t>(dst), DRAM_TO_HOST);
  assert(status == synSuccess && "synMemCopyAsync failed!");
  status = synStreamSynchronize(g_stream->memcpyStreamDevToHost);
  assert(status == synSuccess && "synStreamSynchronize failed!");
  status = HostUnmap(device->id, dst);
  assert(status == synSuccess && "synHostUnmap failed!");
  
  return C_SUCCESS;
}

C_Status d2dMemCpy(const C_Device device,
                void *dst,
                const void *src,
                size_t size) {
  FUNCALL_LOG
  // memcpy(dst, src, size);
  // synMemCopyAsync(streamhandle, src, size, dst, DRAM_TO_DRAM);
  return C_SUCCESS;
}

C_Status h2dAsyncMemCpy(const C_Device device,
                     C_Stream stream,
                     void *dst,
                     const void *src,
                     size_t size) {
  FUNCALL_LOG
  memcpy(dst, src, size);
  return C_SUCCESS;
}

C_Status d2hAsyncMemCpy(const C_Device device,
                     C_Stream stream,
                     void *dst,
                     const void *src,
                     size_t size) {
  FUNCALL_LOG
  memcpy(dst, src, size);
  return C_SUCCESS;
}

C_Status d2dAsyncMemCpy(const C_Device device,
                     C_Stream stream,
                     void *dst,
                     const void *src,
                     size_t size) {
  FUNCALL_LOG
  memcpy(dst, src, size);
  return C_SUCCESS;
}

C_Status MemCpyP2P(const C_Device dst_device,
                   const C_Device src_device,
                   void *dst,
                   const void *src,
                   size_t size) {
  FUNCALL_LOG
  memcpy(dst, src, size);
  return C_SUCCESS;
}

C_Status AsyncMemCpyP2P(const C_Device dst_device,
                        const C_Device src_device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size) {
  FUNCALL_LOG
  memcpy(dst, src, size);
  return C_SUCCESS;
}

C_Status deviceAllocate(const C_Device device, void **ptr, size_t size) {
  FUNCALL_LOG

  // auto data = malloc(size);
  // if (data) {
  //   *ptr = data;
  //   return C_SUCCESS;
  // } else {
  //   *ptr = nullptr;
  // }

  //TODO: how to bring into tensor name
  LOG(INFO) << "deviceAllocate device id=" << device->id;
  static int i = 0;
  uint64_t input_dram;
  std::string name = "Tensor_" + std::to_string(device->id) + "_" + std::to_string(i);
  i ++;
  synStatus status = hbmAlloc(device->id, size, &input_dram, name);
  assert(status == synSuccess && "hbmAlloc failed!");
  *ptr = reinterpret_cast<void*>(input_dram);
  LOG(INFO) << "deviceAllocate ptr=" << *ptr << " input_dram=" << input_dram;

  return C_SUCCESS;
}

C_Status hostAllocate(const C_Device device, void **ptr, size_t size) {
  FUNCALL_LOG

  // auto data = malloc(size);
  // if (data) {
  //   *ptr = data;
  //   return C_SUCCESS;
  // } else {
  //   *ptr = nullptr;
  // }

  //TODO: how to bring into tensor name

  return C_FAILED;
}

C_Status deviceDeallocate(const C_Device device, void *ptr, size_t size) {
  FUNCALL_LOG

  // free(ptr);
  hbmFree(device->id, *reinterpret_cast<uint64_t*>(ptr), "");
  return C_SUCCESS;
}

C_Status hostDeallocate(const C_Device device, void *ptr, size_t size) {
  FUNCALL_LOG

  // free(ptr);
  return C_SUCCESS;
}

C_Status CreateStream(const C_Device device, C_Stream *stream) {
  FUNCALL_LOG
  // stream = nullptr;

  createStream(device->id, stream);
  g_stream = *stream;
  return C_SUCCESS;
}

C_Status DestroyStream(const C_Device device, C_Stream stream) {
  FUNCALL_LOG
  destroyStream(device->id, stream);
  return C_SUCCESS;
}

C_Status CreateEvent(const C_Device device, C_Event *event) {
  FUNCALL_LOG
  createEvent(device->id, event);
  return C_SUCCESS;
}

C_Status RecordEvent(const C_Device device, C_Stream stream, C_Event event) {
  FUNCALL_LOG
  recordEvent(device->id, stream, event);
  return C_SUCCESS;
}

C_Status DestroyEvent(const C_Device device, C_Event event) {
  FUNCALL_LOG
  destroyEvent(device->id, event);
  return C_SUCCESS;
}

C_Status SyncDevice(const C_Device device) {
  FUNCALL_LOG
  synDeviceSynchronize(device->id);
  return C_SUCCESS; 
}

C_Status SyncStream(const C_Device device, C_Stream stream) {
  FUNCALL_LOG
  //TODO: decide which stream to sync
  synStreamSynchronize(stream->computeStream);
  return C_SUCCESS;
}

C_Status SyncEvent(const C_Device device, C_Event event) {
  FUNCALL_LOG
  synEventSynchronize(event->eventHandle);
  return C_SUCCESS; 
}

C_Status StreamWaitEvent(const C_Device device,
                         C_Stream stream,
                         C_Event event) {
  FUNCALL_LOG
  //TODO: decide how which stream to wait
  synStreamWaitEvent(stream->computeStream, event->eventHandle, 0);                          
  return C_SUCCESS;
}

C_Status DeviceMemStats(const C_Device device,
                        size_t *total_memory,
                        size_t *free_memory) {
  FUNCALL_LOG

  synDeviceGetMemoryInfo(device->id, free_memory, total_memory);
  LOG(INFO) << "meminfo:" << *free_memory << "/" << *total_memory;

  return C_SUCCESS;
}

C_Status DeviceMinChunkSize(const C_Device device, size_t *size) {
  FUNCALL_LOG
  *size = 512;
  return C_SUCCESS;
}

// for unittest
C_Status XcclGetUniqueIdSize(size_t *sz) {
  FUNCALL_LOG

  return C_SUCCESS;
}

C_Status XcclGetUniqueId(C_CCLRootId *unique_id) {
  FUNCALL_LOG

  return C_SUCCESS;
}

C_Status XcclCommInitRank(size_t ranks,
                          C_CCLRootId *unique_id,
                          size_t rank,
                          C_CCLComm *comm) {
  FUNCALL_LOG

  return C_SUCCESS;
}

C_Status XcclDestroyComm(C_CCLComm comm) {
  FUNCALL_LOG

  return C_SUCCESS;
}

C_Status XcclAllReduce(void *send_buf,
                       void *recv_buf,
                       size_t count,
                       C_DataType data_type,
                       C_CCLReduceOp op,
                       C_CCLComm comm,
                       C_Stream stream) {
  FUNCALL_LOG

  return C_SUCCESS;
}

C_Status XcclBroadcast(void *buf,
                       size_t count,
                       C_DataType data_type,
                       size_t root,
                       C_CCLComm comm,
                       C_Stream stream) {
  FUNCALL_LOG

  return C_SUCCESS;
}

C_Status ProfilerInitialize(C_Profiler prof, void **user_data) {
  FUNCALL_LOG
  return C_SUCCESS;
}

C_Status ProfilerFinalize(C_Profiler prof, void *user_data) {
  FUNCALL_LOG
  return C_SUCCESS;
}

C_Status ProfilerPrepare(C_Profiler prof, void *user_data) {
  FUNCALL_LOG
  return C_SUCCESS; 
}

C_Status ProfilerStart(C_Profiler prof, void *user_data) {
  FUNCALL_LOG
  return C_SUCCESS; 
}

C_Status ProfilerStop(C_Profiler prof, void *user_data) {
  FUNCALL_LOG
  return C_SUCCESS; 
}

C_Status ProfilerCollectData(C_Profiler prof,
                             uint64_t start_ns,
                             void *user_data) {
  FUNCALL_LOG
  return C_SUCCESS;
}

void InitPlugin(CustomRuntimeParams *params) {
  FUNCALL_LOG
  PADDLE_CUSTOM_RUNTIME_CHECK_VERSION(params);
  params->device_type = "habana_hpu";
  params->sub_device_type = "gaudi";

  memset(reinterpret_cast<void *>(params->interface),
         0,
         sizeof(C_DeviceInterface));

  params->interface->initialize = Init;
  params->interface->finalize = Finalize;

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

  params->interface->memory_copy_h2d = h2dMemCpy;
  params->interface->memory_copy_d2d = d2dMemCpy;
  params->interface->memory_copy_d2h = d2hMemCpy;
  params->interface->memory_copy_p2p = MemCpyP2P;
  params->interface->async_memory_copy_h2d = h2dAsyncMemCpy;
  params->interface->async_memory_copy_d2d = d2dAsyncMemCpy;
  params->interface->async_memory_copy_d2h = d2hAsyncMemCpy;
  params->interface->async_memory_copy_p2p = AsyncMemCpyP2P;
  params->interface->device_memory_allocate = deviceAllocate;
  params->interface->host_memory_allocate = hostAllocate;
  // params->interface-> = Allocate;
  params->interface->device_memory_deallocate = deviceDeallocate;
  params->interface->host_memory_deallocate = hostDeallocate;
  // params->interface->unified_memory_deallocate = Deallocate;

  params->interface->get_device_count = GetDevicesCount;
  params->interface->get_device_list = GetDevicesList;
  params->interface->device_memory_stats = DeviceMemStats;
  params->interface->device_min_chunk_size = DeviceMinChunkSize;

  params->interface->xccl_get_unique_id_size = XcclGetUniqueIdSize;
  params->interface->xccl_get_unique_id = XcclGetUniqueId;
  params->interface->xccl_comm_init_rank = XcclCommInitRank;
  params->interface->xccl_destroy_comm = XcclDestroyComm;
  params->interface->xccl_all_reduce = XcclAllReduce;
  params->interface->xccl_broadcast = XcclBroadcast;

  params->interface->profiler_collect_trace_data = ProfilerCollectData;
  params->interface->profiler_initialize = ProfilerInitialize;
  params->interface->profiler_finalize = ProfilerFinalize;
  params->interface->profiler_start_tracing = ProfilerStart;
  params->interface->profiler_stop_tracing = ProfilerStop;
  params->interface->profiler_prepare_tracing = ProfilerPrepare;
}
