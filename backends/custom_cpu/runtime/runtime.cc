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
#include <iostream>

#include "paddle/phi/backends/device_ext.h"

#define MEMORY_FRACTION 0.5f

static int global_current_device = 0;

C_Status Init() {
  std::cout << "custom_cpu plugin compiled with ";
#ifdef __clang__
  std::cout << "clang\n";
#else
  std::cout << "gcc\n";
#endif
  return C_SUCCESS;
}

C_Status InitDevice(const C_Device device) {
  global_current_device = device->id;
  return C_SUCCESS;
}

C_Status SetDevice(const C_Device device) {
  global_current_device = device->id;
  return C_SUCCESS;
}

C_Status GetDevice(const C_Device device) {
  device->id = global_current_device;
  return C_SUCCESS;
}

C_Status DestroyDevice(const C_Device device) { return C_SUCCESS; }

C_Status Finalize() { return C_SUCCESS; }

C_Status GetDevicesCount(size_t *count) {
  *count = 2;
  return C_SUCCESS;
}

C_Status GetDevicesList(size_t *devices) {
  devices[0] = 0;
  devices[1] = 1;
  return C_SUCCESS;
}

C_Status MemCpy(const C_Device device,
                void *dst,
                const void *src,
                size_t size) {
  memcpy(dst, src, size);
  return C_SUCCESS;
}

C_Status AsyncMemCpy(const C_Device device,
                     C_Stream stream,
                     void *dst,
                     const void *src,
                     size_t size) {
  memcpy(dst, src, size);
  return C_SUCCESS;
}

C_Status MemCpyP2P(const C_Device dst_device,
                   const C_Device src_device,
                   void *dst,
                   const void *src,
                   size_t size) {
  memcpy(dst, src, size);
  return C_SUCCESS;
}

C_Status AsyncMemCpyP2P(const C_Device dst_device,
                        const C_Device src_device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size) {
  memcpy(dst, src, size);
  return C_SUCCESS;
}

C_Status Allocate(const C_Device device, void **ptr, size_t size) {
  auto data = malloc(size);
  if (data) {
    *ptr = data;
    return C_SUCCESS;
  } else {
    *ptr = nullptr;
  }
  return C_FAILED;
}

C_Status Deallocate(const C_Device device, void *ptr, size_t size) {
  free(ptr);
  return C_SUCCESS;
}

C_Status CreateStream(const C_Device device, C_Stream *stream) {
  stream = nullptr;
  return C_SUCCESS;
}

C_Status DestroyStream(const C_Device device, C_Stream stream) {
  return C_SUCCESS;
}

C_Status CreateEvent(const C_Device device, C_Event *event) {
  return C_SUCCESS;
}

C_Status RecordEvent(const C_Device device, C_Stream stream, C_Event event) {
  return C_SUCCESS;
}

C_Status DestroyEvent(const C_Device device, C_Event event) {
  return C_SUCCESS;
}

C_Status SyncDevice(const C_Device device) { return C_SUCCESS; }

C_Status SyncStream(const C_Device device, C_Stream stream) {
  return C_SUCCESS;
}

C_Status SyncEvent(const C_Device device, C_Event event) { return C_SUCCESS; }

C_Status StreamWaitEvent(const C_Device device,
                         C_Stream stream,
                         C_Event event) {
  return C_SUCCESS;
}

C_Status VisibleDevices(size_t *devices) { return C_SUCCESS; }

C_Status DeviceMemStats(const C_Device device,
                        size_t *total_memory,
                        size_t *free_memory) {
  float memusage;
  FILE *fp;
  char buffer[1024];
  size_t byte_read;
  char *pos;

  fp = fopen("/proc/meminfo", "r");
  byte_read = fread(buffer, 1, sizeof(buffer), fp);
  fclose(fp);
  buffer[byte_read] = '\0';
  pos = strstr(buffer, "MemTotal:");
  sscanf(pos, "MemTotal: %lu kB", total_memory);
  pos = strstr(pos, "MemFree:");
  sscanf(pos, "MemFree: %lu kB", free_memory);
  *total_memory = *total_memory * 1024;
  *free_memory = *free_memory * 1024;
  *free_memory = *free_memory * MEMORY_FRACTION;

  return C_SUCCESS;
}

C_Status DeviceMinChunkSize(const C_Device device, size_t *size) {
  *size = 512;
  return C_SUCCESS;
}

struct C_CCLComm_st {
  size_t rank;
  size_t nranks;
  sem_t *sig;
  sem_t *sig_2;
  std::string sig_name;
  std::string sig_2_name;
};

// for unittest
C_Status XcclGetUniqueIdSize(size_t *sz) {
  *sz = sizeof(size_t);
  return C_SUCCESS;
}

C_Status XcclGetUniqueId(C_CCLRootId *unique_id) {
  auto ptr = reinterpret_cast<int8_t *>(unique_id->data);
  for (auto i = 0; i < unique_id->sz - 1; ++i) {
    ptr[i] = static_cast<int8_t>(std::rand() % ('z' - 'a') + 'a');
  }
  ptr[unique_id->sz - 1] = '\0';
  return C_SUCCESS;
}

C_Status XcclCommInitRank(size_t ranks,
                          C_CCLRootId *unique_id,
                          size_t rank,
                          C_CCLComm *comm) {
  auto sig = sem_open(static_cast<char *>(unique_id->data),
                      O_CREAT,
                      0644,
                      0);
  auto sig_2 = sem_open(static_cast<char *>(unique_id->data) + 1,
                        O_CREAT,
                        0644,
                        0);
  *comm =
      new C_CCLComm_st({rank,
                        ranks,
                        sig,
                        sig_2,
                        std::string(static_cast<char *>(unique_id->data)),
                        std::string(static_cast<char *>(unique_id->data) + 1)});
  return C_SUCCESS;
}

C_Status XcclDestroyComm(C_CCLComm comm) {
  if (comm) {
    sem_unlink(comm->sig_name.c_str());
    sem_unlink(comm->sig_2_name.c_str());
    delete comm;
  }
  return C_SUCCESS;
}

C_Status XcclAllReduce(void *send_buf,
                       void *recv_buf,
                       size_t count,
                       C_DataType data_type,
                       C_CCLReduceOp op,
                       C_CCLComm comm,
                       C_Stream stream) {
  sem_post(comm->sig);

  if (comm->rank == 0) {
    for (auto i = 0; i < comm->nranks; ++i) {
      sem_wait(comm->sig);
    }

    for (auto i = 0; i < comm->nranks; ++i) {
      sem_post(comm->sig_2);
    }
  }

  sem_wait(comm->sig_2);
  return C_SUCCESS;
}

C_Status XcclBroadcast(void *buf,
                       size_t count,
                       C_DataType data_type,
                       size_t root,
                       C_CCLComm comm,
                       C_Stream stream) {
  sem_post(comm->sig);

  if (comm->rank == 0) {
    for (auto i = 0; i < comm->nranks; ++i) {
      sem_wait(comm->sig);
    }

    for (auto i = 0; i < comm->nranks; ++i) {
      sem_post(comm->sig_2);
    }
  }

  sem_wait(comm->sig_2);
  return C_SUCCESS;
}

void InitPlugin(CustomRuntimeParams *params) {
  PADDLE_CUSTOM_RUNTIME_CHECK_VERSION(params);
  params->device_type = "custom_cpu";
  params->sub_device_type = "v0.1";

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

  params->interface->memory_copy_h2d = MemCpy;
  params->interface->memory_copy_d2d = MemCpy;
  params->interface->memory_copy_d2h = MemCpy;
  params->interface->memory_copy_p2p = MemCpyP2P;
  params->interface->async_memory_copy_h2d = AsyncMemCpy;
  params->interface->async_memory_copy_d2d = AsyncMemCpy;
  params->interface->async_memory_copy_d2h = AsyncMemCpy;
  params->interface->async_memory_copy_p2p = AsyncMemCpyP2P;
  params->interface->device_memory_allocate = Allocate;
  params->interface->host_memory_allocate = Allocate;
  params->interface->unified_memory_allocate = Allocate;
  params->interface->device_memory_deallocate = Deallocate;
  params->interface->host_memory_deallocate = Deallocate;
  params->interface->unified_memory_deallocate = Deallocate;

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
}
