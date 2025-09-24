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
//
// Modifications:
// Copyright (c) 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All
// rights reserved.

// - [add musa runtime api]
// - [add ccl and others api of Moorethread]

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
#include <mutex>

// #include "glog/logging.h"
#include "musa_device_handler.h"  // NOLINT
#include "paddle/common/exception.h"
#include "paddle/phi/backends/device_ext.h"
#include "runtime/mccl_handler.h"
#include "runtime/utils.h"

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

C_Status DestroyDevice(const C_Device device) { return C_SUCCESS; }

C_Status Finalize() { return C_SUCCESS; }

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
  auto sig = sem_open(static_cast<char *>(unique_id->data), O_CREAT, 0644, 0);
  auto sig_2 =
      sem_open(static_cast<char *>(unique_id->data) + 1, O_CREAT, 0644, 0);
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

C_Status ProfilerInitialize(C_Profiler prof, void **user_data) {
  return C_SUCCESS;
}

C_Status ProfilerFinalize(C_Profiler prof, void *user_data) {
  return C_SUCCESS;
}

C_Status ProfilerPrepare(C_Profiler prof, void *user_data) { return C_SUCCESS; }

C_Status ProfilerStart(C_Profiler prof, void *user_data) { return C_SUCCESS; }

C_Status ProfilerStop(C_Profiler prof, void *user_data) { return C_SUCCESS; }

C_Status ProfilerCollectData(C_Profiler prof,
                             uint64_t start_ns,
                             void *user_data) {
  return C_SUCCESS;
}

void InitPlugin(CustomRuntimeParams *params) {
  PADDLE_CUSTOM_RUNTIME_CHECK_VERSION(params);
  params->device_type = "musa";
  params->sub_device_type = "1.0.0";

  memset(reinterpret_cast<void *>(params->interface),
         0,
         sizeof(C_DeviceInterface));

  params->interface->initialize = Init;
  params->interface->finalize = Finalize;

  params->interface->init_device = musa::device::InitMusaDevice;
  params->interface->set_device = musa::device::SetMusaDevice;
  params->interface->get_device = musa::device::GetMusaDevice;
  params->interface->deinit_device = DestroyDevice;
  params->interface->get_device_count = musa::device::GetMusaDeviceCount;
  params->interface->get_device_list = musa::device::GetMusaDevicesList;

  params->interface->create_stream = musa::device::CreateMusaStream;
  params->interface->destroy_stream = musa::device::DestroyMusaStream;

  params->interface->create_event = musa::device::CreateMusaEvent;
  params->interface->destroy_event = musa::device::DestroyMusaEvent;
  params->interface->record_event = musa::device::RecordMusaEvent;

  params->interface->synchronize_device = musa::device::SyncMusaDevice;
  params->interface->synchronize_stream = musa::device::SyncMusaStream;
  params->interface->synchronize_event = musa::device::SyncMusaEvent;
  params->interface->stream_wait_event = musa::device::MusaStreamWaitEvent;

  params->interface->memory_copy_h2d = musa::device::MusaMemCpyH2D;
  params->interface->memory_copy_d2d = musa::device::MusaMemCpyD2D;
  params->interface->memory_copy_d2h = musa::device::MusaMemCpyD2H;
  params->interface->memory_copy_p2p = musa::device::MusaMemCpyP2P;
  params->interface->async_memory_copy_h2d = musa::device::MusaMemCpyH2DAsync;
  params->interface->async_memory_copy_d2d = musa::device::MusaMemCpyD2DAsync;
  params->interface->async_memory_copy_d2h = musa::device::MusaMemCpyD2HAsync;
  params->interface->async_memory_copy_p2p = musa::device::MusaMemCpyP2PAsync;
  params->interface->device_memory_allocate = musa::device::MusaMemAllocate;
  params->interface->host_memory_allocate = musa::device::MusaMemAllocateHost;
  params->interface->device_memory_deallocate = musa::device::MusaMemDeallocate;
  params->interface->host_memory_deallocate =
      musa::device::MusaMemDeallocateHost;

  params->interface->unified_memory_allocate =
      nullptr;  // TODO(jihong.zhong): fill it
  params->interface->unified_memory_deallocate =
      nullptr;  // TODO(jihong.zhong): fill it

  // params->interface->device_memory_stats = DeviceMemStats;
  // params->interface->device_min_chunk_size = DeviceMinChunkSize;

  params->interface->get_compute_capability =
      musa::device::GetMusaComputeCapability;
  params->interface->get_device_properties =
      musa::device::GetMusaDeviceProperties;
  params->interface->get_runtime_version = musa::device::GetMusaRuntimeVersion;
  params->interface->get_driver_version = musa::device::GetMusaDriverVersion;
  params->interface->get_multi_process =
      musa::device::GetMusaMultiProcessorCount;
  params->interface->get_max_threads_per_mp =
      musa::device::GetMusaMaxThreadsPerMP;
  params->interface->get_max_threads_per_block =
      musa::device::GetMusaMaxThreadsPerBlock;
  params->interface->get_max_grid_dim_size =
      musa::device::GetMusaMaxGridDimSize;

  params->interface->init_eigen_device = musa::device::MusaInitEigenDevice;
  params->interface->destroy_eigen_device =
      musa::device::MusaDestroyEigenDevice;

  auto is_fp16_func = [](const C_Device device, bool *supported) -> C_Status {
    *supported = true;
    return C_SUCCESS;
  };
  auto is_bf16_func = is_fp16_func;
  // params->interface->is_float16_supported = is_fp16_func;
  // params->interface->is_bfloat16_supported = is_bf16_func;
  // //TODO(jihong.zhong): fix it

  params->interface->xccl_group_end = musa::mccl::McclGroupEnd;
  params->interface->xccl_group_start = musa::mccl::McclGroupStart;
  params->interface->xccl_get_unique_id_size = musa::mccl::McclGetUniqueIdSize;
  params->interface->xccl_get_unique_id = musa::mccl::McclGetUniqueId;
  params->interface->xccl_comm_init_rank = musa::mccl::McclCommInitRank;
  params->interface->xccl_destroy_comm = musa::mccl::McclDestroyComm;
  params->interface->xccl_all_reduce = musa::mccl::McclAllReduce;
  params->interface->xccl_all_gather = musa::mccl::McclAllGather;
  params->interface->xccl_broadcast = musa::mccl::McclBroadcast;
  params->interface->xccl_all_to_all = musa::mccl::McclAll2All;
  params->interface->xccl_reduce_scatter = musa::mccl::McclReduceScatter;
  params->interface->xccl_send = musa::mccl::McclSend;
  params->interface->xccl_recv = musa::mccl::McclRecv;
  params->interface->xccl_reduce = musa::mccl::McclReduce;
  params->interface->xccl_get_comm_name = [](C_CCLComm comm,
                                             char *comm_name) -> C_Status {
    static std::string name("PaddleWithMccl_" + musa::GetMcclVer());
    memcpy(comm_name, name.c_str(), name.size());
    return C_SUCCESS;
  };
  // params->interface->profiler_collect_trace_data = ProfilerCollectData;
  // params->interface->profiler_initialize = ProfilerInitialize;
  // params->interface->profiler_finalize = ProfilerFinalize;
  // params->interface->profiler_start_tracing = ProfilerStart;
  // params->interface->profiler_stop_tracing = ProfilerStop;
  // params->interface->profiler_prepare_tracing = ProfilerPrepare;
}
