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

#include "utils/hpu_utils.h"
#include "utils/hpu_helper.h"

static int global_current_device = 0;
static C_Stream g_stream = nullptr;

C_Status Init() {
  FUNCALL_S

  synStatus status = synFail;
  status = synInitialize();
  assert(status == synSuccess && "synapse initialization failed");

  FUNCALL_E
  return C_SUCCESS;
}

C_Status InitDevice(const C_Device device) {
  FUNCALL_S
  LOG(INFO) << "device id=" << device->id;

  uint32_t deviceId = 0;
  synDeviceType deviceType = synDeviceGaudi2;
  bool deviceAcquired = waitForIdleDevice(&deviceId, /* INOUT */ deviceType, 1);

  LOG(INFO) << "requested device id=" << device->id << ", real device id=" << deviceId;
  global_current_device = deviceId;
  device->id = deviceId;

  FUNCALL_E
  return C_SUCCESS;
}

C_Status SetDevice(const C_Device device) {
  FUNCALL_S
  LOG(INFO) << "device id=" << device->id;

  //nothing to do
  global_current_device = device->id;
  FUNCALL_E
  return C_SUCCESS;
}

C_Status GetDevice(const C_Device device) {
  FUNCALL_S

  //nothing to do
  device->id = global_current_device;
  LOG(INFO) << "device id=" << device->id;
  FUNCALL_E
  return C_SUCCESS;
}

C_Status DestroyDevice(const C_Device device) {
  FUNCALL_S

  LOG(INFO) << "device id=" << device->id;

  synStatus status = synDeviceRelease(device->id);
  CHKSTATUS("synDeviceRelease failed!");
  FUNCALL_E
  return C_SUCCESS; 
}

C_Status Finalize() {
  FUNCALL_S

  synStatus status = synDestroy();
  CHKSTATUS("synDestroy failed");
  FUNCALL_E
  return C_SUCCESS; 
}

C_Status GetDevicesCount(size_t *count) {
  FUNCALL_S
  uint32_t pCount = 0;

  synStatus status = synDeviceGetCount(&pCount);
  CHKSTATUS("synDeviceGetCount failed!");
  //currently only expose 1 device 
  *count = 1;
  LOG(INFO) << "get real device count=" << pCount << " actual return " << *count;
  FUNCALL_E
  return C_SUCCESS;
}

C_Status GetDevicesList(size_t *devices) {
  FUNCALL_S

  //TODO: suse HABANA_VISIBLE_DEVICES to get available device
  devices[0] = 0;
  // devices[1] = 1;
  FUNCALL_E
  return C_SUCCESS;
}

C_Status MemCpy_h2d(const C_Device device,
                void *dst,
                const void *src,
                size_t size) {
  FUNCALL_S
  LOG(INFO) << "device id=" << device->id << " dst=" << dst << " src=" << src << " size=" << size;

  synStatus status = HostMap(device->id, size, src);
  CHKSTATUS("synHostMap failed!");
  status = synMemCopyAsync(g_stream->memcpyStreamHostToDev, reinterpret_cast<uint64_t>(src), size, reinterpret_cast<uint64_t>(dst), HOST_TO_DRAM);
  CHKSTATUS("synMemCopyAsync HOST_TO_DRAM failed!");
  status = synStreamSynchronize(g_stream->memcpyStreamHostToDev);
  CHKSTATUS("synStreamSynchronize failed!");
  status = HostUnmap(device->id, src);
  CHKSTATUS("synHostUnmap failed!");

  FUNCALL_E
  return C_SUCCESS;
}

C_Status MemCpy_d2h(const C_Device device,
                void *dst,
                const void *src,
                size_t size) {
  FUNCALL_S
  LOG(INFO) << "device id=" << device->id << " dst=" << dst << " src=" << src << " size=" << size;

  synStatus status = HostMap(device->id, size, dst);
  CHKSTATUS("synHostMap failed!");
  status = synMemCopyAsync(g_stream->memcpyStreamDevToHost, reinterpret_cast<uint64_t>(src), size, reinterpret_cast<uint64_t>(dst), DRAM_TO_HOST);
  CHKSTATUS("synMemCopyAsync DRAM_TO_HOST failed!");
  status = synStreamSynchronize(g_stream->memcpyStreamDevToHost);
  CHKSTATUS("synStreamSynchronize failed!");
  status = HostUnmap(device->id, dst);
  CHKSTATUS("synHostUnmap failed!");
  
  FUNCALL_E
  return C_SUCCESS;
}

C_Status MemCpy_d2d(const C_Device device,
                void *dst,
                const void *src,
                size_t size) {
  FUNCALL_S
  // memcpy(dst, src, size);
  // synMemCopyAsync(streamhandle, src, size, dst, DRAM_TO_DRAM);

  FUNCALL_E
  return C_SUCCESS;
}

C_Status AsyncMemCpy_h2d(const C_Device device,
                     C_Stream stream,
                     void *dst,
                     const void *src,
                     size_t size) {
  FUNCALL_S
  LOG(INFO) << "device id=" << device->id << " stream=" << stream << " dst=" << dst << " src=" << src << " size=" << size;

  synStatus status = HostMap(device->id, size, src);
  CHKSTATUS("synHostMap failed!");
  status = synMemCopyAsync(g_stream->memcpyStreamHostToDev, reinterpret_cast<uint64_t>(src), size, reinterpret_cast<uint64_t>(dst), HOST_TO_DRAM);
  CHKSTATUS("synMemCopyAsync HOST_TO_DRAM failed!");
  //TODO: when to unmap in async
  status = HostUnmap(device->id, src);
  CHKSTATUS("synHostUnmap failed!");
  
  FUNCALL_E
  return C_SUCCESS;
}

C_Status AsyncMemCpy_d2h(const C_Device device,
                     C_Stream stream,
                     void *dst,
                     const void *src,
                     size_t size) {
  FUNCALL_S

  LOG(INFO) << "device id=" << device->id << " stream=" << stream << " dst=" << dst << " src=" << src << " size=" << size;

  synStatus status = HostMap(device->id, size, dst);
  CHKSTATUS("synHostMap failed!");
  status = synMemCopyAsync(g_stream->memcpyStreamDevToHost, reinterpret_cast<uint64_t>(src), size, reinterpret_cast<uint64_t>(dst), DRAM_TO_HOST);
  CHKSTATUS("synMemCopyAsync DRAM_TO_HOST failed!");
  //TODO: when to unmap in async
  status = HostUnmap(device->id, dst);
  CHKSTATUS("synHostUnmap failed!");

  FUNCALL_E
  return C_SUCCESS;
}

C_Status AsyncMemCpy_d2d(const C_Device device,
                     C_Stream stream,
                     void *dst,
                     const void *src,
                     size_t size) {
  FUNCALL_S
  memcpy(dst, src, size);

  FUNCALL_E
  return C_SUCCESS;
}

C_Status Allocate_device(const C_Device device, void **ptr, size_t size) {
  FUNCALL_S

  // auto data = malloc(size);
  // if (data) {
  //   *ptr = data;
  //   return C_SUCCESS;
  // } else {
  //   *ptr = nullptr;
  // }

  //TODO: how to bring into tensor name
  static int i = 0;
  uint64_t input_dram;
  std::string name = "Tensor_" + std::to_string(device->id) + "_" + std::to_string(i);
  i ++;
  synStatus status = hbmAlloc(device->id, size, &input_dram, name);
  CHKSTATUS("hbmAlloc failed!");
  *ptr = reinterpret_cast<void*>(input_dram);
  LOG(INFO) << "device id=" << device->id << " name=" << name << " ptr=" << *ptr << " size=" << size;

  FUNCALL_E
  return C_SUCCESS;
}

C_Status Deallocate_device(const C_Device device, void *ptr, size_t size) {
  FUNCALL_S
  LOG(INFO) << "device id=" << device->id << " ptr=" << ptr;

  // free(ptr);
  synStatus status = hbmFree(device->id, *reinterpret_cast<uint64_t*>(ptr), "");
  CHKSTATUS("hbmFree failed!");

  FUNCALL_E
  return C_SUCCESS;
}

C_Status Allocate_host(const C_Device device, void **ptr, size_t size) {
  FUNCALL_S

  // auto data = malloc(size);
  // if (data) {
  //   *ptr = data;
  //   return C_SUCCESS;
  // } else {
  //   *ptr = nullptr;
  // }

  //TODO: how to bring into tensor name

  FUNCALL_E
  return C_FAILED;
}

C_Status Deallocate_host(const C_Device device, void *ptr, size_t size) {
  FUNCALL_S

  // free(ptr);
  FUNCALL_E
  return C_SUCCESS;
}

C_Status CreateStream(const C_Device device, C_Stream *stream) {
  FUNCALL_S
  // stream = nullptr;

  synStatus status = createStream(device->id, stream);
  CHKSTATUS("createStream failed!");
  LOG(INFO) << "device id=" << device->id << " stream=" << *stream;

  g_stream = *stream;

  FUNCALL_E
  return C_SUCCESS;
}

C_Status DestroyStream(const C_Device device, C_Stream stream) {
  FUNCALL_S
  LOG(INFO) << "device id=" << device->id << " stream=" << stream;

  synStatus status = destroyStream(device->id, stream);
  CHKSTATUS("destroyStream failed!");

  FUNCALL_E
  return C_SUCCESS;
}

C_Status CreateEvent(const C_Device device, C_Event *event) {
  FUNCALL_S
  synStatus status = createEvent(device->id, event);
  CHKSTATUS("createEvent failed!");
  LOG(INFO) << "device id=" << device->id << " event=" << *event;

  FUNCALL_E
  return C_SUCCESS;
}

C_Status RecordEvent(const C_Device device, C_Stream stream, C_Event event) {
  FUNCALL_S
  LOG(INFO) << "device id=" << device->id << " stream=" << stream << " event=" << event;

  synStatus status = recordEvent(device->id, stream, event);
  CHKSTATUS("recordEvent failed!");

  FUNCALL_E
  return C_SUCCESS;
}

C_Status DestroyEvent(const C_Device device, C_Event event) {
  FUNCALL_S
  LOG(INFO) << "device id=" << device->id << " event=" << event;
  synStatus status = destroyEvent(device->id, event);
  CHKSTATUS("destroyEvent failed!");

  FUNCALL_E
  return C_SUCCESS;
}

C_Status SyncDevice(const C_Device device) {
  FUNCALL_S
  LOG(INFO) << "device id=" << device->id;
  synStatus status = synDeviceSynchronize(device->id);
  CHKSTATUS("synDeviceSynchronize failed!");

  FUNCALL_E
  return C_SUCCESS; 
}

C_Status SyncStream(const C_Device device, C_Stream stream) {
  FUNCALL_S
  LOG(INFO) << "device id=" << device->id << " stream=" << stream;

  //TODO: decide which stream to sync
  synStatus status = synStreamSynchronize(stream->computeStream);
  CHKSTATUS("synStreamSynchronize failed!");

  FUNCALL_E
  return C_SUCCESS;
}

C_Status SyncEvent(const C_Device device, C_Event event) {
  FUNCALL_S
  LOG(INFO) << "device id=" << device->id << " event=" << event;

  synStatus status = synEventSynchronize(event->eventHandle);
  CHKSTATUS("synEventSynchronize failed!");

  FUNCALL_E
  return C_SUCCESS; 
}

C_Status StreamWaitEvent(const C_Device device,
                         C_Stream stream,
                         C_Event event) {
  FUNCALL_S
  LOG(INFO) << "device id=" << device->id << " stream=" << stream << " event=" << event;

  //TODO: decide how which stream to wait
  synStatus status = synStreamWaitEvent(stream->computeStream, event->eventHandle, 0);
  CHKSTATUS("synStreamWaitEvent failed!");

  FUNCALL_E                       
  return C_SUCCESS;
}

C_Status DeviceMemStats(const C_Device device,
                        size_t *total_memory,
                        size_t *free_memory) {
  FUNCALL_S
  LOG(INFO) << "device id=" << device->id;

  synStatus status = synDeviceGetMemoryInfo(device->id, free_memory, total_memory);
  CHKSTATUS("synDeviceGetMemoryInfo failed!"); 
  LOG(INFO) << "meminfo:" << *free_memory << "/" << *total_memory;

  FUNCALL_E
  return C_SUCCESS;
}

C_Status DeviceMinChunkSize(const C_Device device, size_t *size) {
  FUNCALL_S
  *size = 512;
  LOG(INFO) << "min chunksize=" << *size;

  FUNCALL_E
  return C_SUCCESS;
}

// for unittest
C_Status XcclGetUniqueIdSize(size_t *sz) {
  FUNCALL_S

  FUNCALL_E
  return C_SUCCESS;
}

C_Status XcclGetUniqueId(C_CCLRootId *unique_id) {
  FUNCALL_S

  FUNCALL_E
  return C_SUCCESS;
}

C_Status XcclCommInitRank(size_t ranks,
                          C_CCLRootId *unique_id,
                          size_t rank,
                          C_CCLComm *comm) {
  FUNCALL_S

  FUNCALL_E
  return C_SUCCESS;
}

C_Status XcclDestroyComm(C_CCLComm comm) {
  FUNCALL_S

  FUNCALL_E
  return C_SUCCESS;
}

C_Status XcclAllReduce(void *send_buf,
                       void *recv_buf,
                       size_t count,
                       C_DataType data_type,
                       C_CCLReduceOp op,
                       C_CCLComm comm,
                       C_Stream stream) {
  FUNCALL_S

  FUNCALL_E
  return C_SUCCESS;
}

C_Status XcclBroadcast(void *buf,
                       size_t count,
                       C_DataType data_type,
                       size_t root,
                       C_CCLComm comm,
                       C_Stream stream) {
  FUNCALL_S

  FUNCALL_E
  return C_SUCCESS;
}

C_Status ProfilerInitialize(C_Profiler prof, void **user_data) {
  FUNCALL_S

  FUNCALL_E
  return C_SUCCESS;
}

C_Status ProfilerFinalize(C_Profiler prof, void *user_data) {
  FUNCALL_S

  FUNCALL_E
  return C_SUCCESS;
}

C_Status ProfilerPrepare(C_Profiler prof, void *user_data) {
  FUNCALL_S

  FUNCALL_E
  return C_SUCCESS; 
}

C_Status ProfilerStart(C_Profiler prof, void *user_data) {
  FUNCALL_S

  FUNCALL_E
  return C_SUCCESS; 
}

C_Status ProfilerStop(C_Profiler prof, void *user_data) {
  FUNCALL_S

  FUNCALL_E
  return C_SUCCESS; 
}

C_Status ProfilerCollectData(C_Profiler prof,
                             uint64_t start_ns,
                             void *user_data) {
  FUNCALL_S

  FUNCALL_E
  return C_SUCCESS;
}

void InitPlugin(CustomRuntimeParams *params) {
  FUNCALL_S
  PADDLE_CUSTOM_RUNTIME_CHECK_VERSION(params);
  params->device_type = "intel_hpu";
  params->sub_device_type = "gaudi2";

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

  params->interface->memory_copy_h2d = MemCpy_h2d;
  params->interface->memory_copy_d2d = MemCpy_d2d;
  params->interface->memory_copy_d2h = MemCpy_d2h;
  // params->interface->memory_copy_p2p = MemCpyP2P;
  params->interface->async_memory_copy_h2d = AsyncMemCpy_h2d;
  params->interface->async_memory_copy_d2d = AsyncMemCpy_d2d;
  params->interface->async_memory_copy_d2h = AsyncMemCpy_d2h;
  // params->interface->async_memory_copy_p2p = AsyncMemCpyP2P;
  params->interface->device_memory_allocate = Allocate_device;
  params->interface->host_memory_allocate = Allocate_host;
  // params->interface->unified_memory_allocate = Allocate;
  params->interface->device_memory_deallocate = Deallocate_device;
  params->interface->host_memory_deallocate = Deallocate_host;
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

  FUNCALL_E
}
