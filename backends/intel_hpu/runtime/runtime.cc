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

#include "runtime.h"

#include <errno.h>
#include <fcntl.h>
#include <semaphore.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

#include "hccl.h"
#include "hccl_types.h"

FLAGS_DEFINE_bool(intel_hpu_runtime_debug, false, "runtime debug log");

inline hcclDataType_t PDDataTypeToHcclDataType(C_DataType type) {
  if (type == C_DataType::FLOAT32) {
    return hcclFloat32;
  } else if (type == C_DataType::FLOAT16) {
    return hcclBfloat16;
  } else if (type == C_DataType::INT32) {
    return hcclInt32;
  } else if (type == C_DataType::INT8) {
    return hcclInt8;
  } else if (type == C_DataType::UINT8) {
    return hcclUint8;
  } else if (type == C_DataType::INT64) {
    return hcclInt64;
  } else {
    LOG(ERROR) << "Datatype " << type << " in hccl is not supported.";
  }
}

hcclRedOp_t PDReduceOpToHcclReduceOp(C_CCLReduceOp op) {
  if (op == C_CCLReduceOp::MIN) {
    return hcclMin;
  } else if (op == C_CCLReduceOp::MAX) {
    return hcclMax;
  } else if (op == C_CCLReduceOp::SUM) {
    return hcclSum;
  } else if (op == C_CCLReduceOp::PRODUCT) {
    return hcclProd;
  } else {
    LOG(ERROR) << "Reduceop " << op << " in hccl is not supported.";
  }
}

class RuntimeManager {
 public:
  RuntimeManager() {}
  ~RuntimeManager() {}

  void SetDevice(const C_Device device) {
    synStatus status = synFail;
    if (static_cast<synModuleId>(device->id) == moduleID) {
      if (Status == 0) {
        // status = synDeviceAcquireByModuleId(&deviceId, moduleID);
        status = synDeviceAcquire(&deviceId, nullptr);
        LOG_IF(ERROR, status != synSuccess)
            << "[RUNTIME] synDeviceAcquireByModuleId() failed: [" << status
            << "]";
      }
    } else {
      // release the old one and acquire a new one
      if (Status == 1) {
        status = synDeviceRelease(deviceId);
        LOG_IF(ERROR, status != synSuccess)
            << "[RUNTIME] synDeviceRelease() failed: [" << status << "]";
      }
      // status = synDeviceAcquireByModuleId(&deviceId, moduleID);
      status = synDeviceAcquire(&deviceId, nullptr);
      LOG_IF(ERROR, status != synSuccess)
          << "[RUNTIME] synDeviceAcquireByModuleId() failed: [" << status
          << "]";
    }
    Status = 1;

    LOG_IF(INFO, FLAGS_intel_hpu_runtime_debug)
        << "set device id to " << device->id;
    moduleID = device->id;
  }

  void Release(const C_Device device) {
    LOG_IF(ERROR, static_cast<synModuleId>(device->id) != moduleID)
        << "[RUNTIME] moduleID mismatch : moduleID = " << moduleID
        << ", current = " << moduleID;

    if (Status == 1) {
      synStatus status = synDeviceRelease(deviceId);
      LOG_IF(ERROR, status != synSuccess)
          << "[RUNTIME] synDeviceRelease() failed: [" << status << "]";
    }

    LOG_IF(INFO, FLAGS_intel_hpu_runtime_debug)
        << "moduleID =  " << moduleID << " released";
  }

  int GetDevice() { return deviceId; }

  void GetMemoryStats(const C_Device device,
                      size_t *total_memory,
                      size_t *free_memory) {
    LOG_IF(ERROR, static_cast<synModuleId>(device->id) != moduleID)
        << "[RUNTIME] moduleID mismatch : moduleID = " << moduleID
        << ", current = " << moduleID;

    LOG_IF(ERROR, Status != 1)
        << "[RUNTIME] device not acquired, status = " << Status;

    uint64_t free = 0;
    uint64_t total = 0;

    synStatus status = synDeviceGetMemoryInfo(deviceId, &free, &total);
    LOG_IF(ERROR, status != synSuccess)
        << "[RUNTIME] synDeviceGetMemoryInfo() failed = " << status;

    LOG_IF(INFO, FLAGS_intel_hpu_runtime_debug) << "free = " << free;
    LOG_IF(INFO, FLAGS_intel_hpu_runtime_debug) << "total = " << total;

    *total_memory = static_cast<size_t>(total);
    *free_memory = static_cast<size_t>(free);
  }

  void GetNumDevices(size_t *count) {
    uint32_t num_devices = 0;
    synStatus status = synDeviceGetCount(&num_devices);
    LOG_IF(ERROR, status != synSuccess)
        << "[RUNTIME] synDeviceGetCount() failed = " << status;
    if (num_devices >= 1) {
      LOG_IF(INFO, FLAGS_intel_hpu_runtime_debug)
          << "total device num =" << num_devices
          << " actual return 1, only support 1 thread acquire 1 device";
      num_devices = 1;
    }

    *count = static_cast<size_t>(num_devices);
    LOG_IF(INFO, FLAGS_intel_hpu_runtime_debug)
        << " number device = " << *count;
  }

  C_Status CreateStream(const C_Device device, C_Stream *stream) {
    LOG_IF(ERROR, static_cast<synModuleId>(device->id) != moduleID)
        << "[RUNTIME] moduleID mismatch : moduleID = " << moduleID
        << ", current = " << moduleID;

    auto it = streams.find(reinterpret_cast<synStreamHandle>(*stream));
    if (it == streams.end()) {
      synStreamHandle h = nullptr;
      synStatus status = synStreamCreateGeneric(&h, device->id, 0);
      LOG_IF(ERROR, status != synSuccess)
          << "[RUNTIME] synStreamCreateGeneric() failed = " << status;

      streams.insert(h);
      *stream = reinterpret_cast<C_Stream>(h);
      LOG_IF(INFO, FLAGS_intel_hpu_runtime_debug)
          << "create stream stream = " << h;
    }

    return C_SUCCESS;
  }

  C_Status DestroyStream(const C_Device device, C_Stream stream) {
    LOG_IF(ERROR, static_cast<synModuleId>(device->id) != moduleID)
        << "[RUNTIME] moduleID mismatch : moduleID = " << moduleID
        << ", current = " << moduleID;
    auto it = streams.find(reinterpret_cast<synStreamHandle>(stream));
    if (it != streams.end()) {
      synStatus status = synStreamDestroy(*it);
      LOG_IF(ERROR, status != synSuccess)
          << "[RUNTIME] synStreamDestroy() failed = " << status;
      streams.erase(*it);
    }
    return C_SUCCESS;
  }

  C_Status Copy(const C_Device device,
                void *dst,
                const void *src,
                size_t size,
                size_t flag = 0 /*0 = h2d, 1 = d2h, 2=d2d*/) {
    // TODO: cache mapped host addr
    LOG_IF(INFO, FLAGS_intel_hpu_runtime_debug)
        << "copy: flag = " << flag << ", size = " << size << ", src = " << src
        << ", dst = " << dst;
    synStatus status = synFail;
    if (flag == 0) {
      if (stream_h2d == nullptr) {
        status = synStreamCreateGeneric(
            reinterpret_cast<synStreamHandle *>(&stream_h2d), device->id, 0);
        LOG_IF(ERROR, status != synSuccess)
            << "[RUNTIME] synStreamCreateGeneric() failed = " << status;
        LOG_IF(INFO, FLAGS_intel_hpu_runtime_debug)
            << "create builtin stream h2d" << stream_h2d;
      }

      status = synHostMap(device->id, size, src);
      LOG_IF(ERROR, status != synSuccess)
          << "[RUNTIME] synHostMap() failed = " << status;

      status = synMemCopyAsync(stream_h2d,
                               reinterpret_cast<uint64_t>(src),
                               size,
                               reinterpret_cast<uint64_t>(dst),
                               HOST_TO_DRAM);
      LOG_IF(ERROR, status != synSuccess)
          << "[RUNTIME] synMemCopyAsync(HOST_TO_DRAM) failed = " << status;

      status = synStreamSynchronize(stream_h2d);
      LOG_IF(ERROR, status != synSuccess)
          << "[RUNTIME] synStreamSynchronize(stream_h2d) failed = " << status;

      status = synHostUnmap(device->id, src);
      LOG_IF(ERROR, status != synSuccess)
          << "[RUNTIME] synHostUnmap() failed = " << status;
    } else if (flag == 1) {
      if (stream_d2h == nullptr) {
        status = synStreamCreateGeneric(
            reinterpret_cast<synStreamHandle *>(&stream_d2h), device->id, 0);
        LOG_IF(ERROR, status != synSuccess)
            << "[RUNTIME] synStreamCreateGeneric() failed = " << status;
      }

      status = synHostMap(device->id, size, dst);
      LOG_IF(ERROR, status != synSuccess)
          << "[RUNTIME] synHostMap() failed = " << status;
      status = synMemCopyAsync(stream_d2h,
                               reinterpret_cast<uint64_t>(src),
                               size,
                               reinterpret_cast<uint64_t>(dst),
                               DRAM_TO_HOST);
      LOG_IF(ERROR, status != synSuccess)
          << "[RUNTIME] synMemCopyAsync() failed = " << status;
      status = synStreamSynchronize(stream_d2h);
      LOG_IF(ERROR, status != synSuccess)
          << "[RUNTIME] synStreamSynchronize() failed = " << status;
      status = synHostUnmap(device->id, dst);
      LOG_IF(ERROR, status != synSuccess)
          << "[RUNTIME] synHostUnmap() failed = " << status;
    } else if (flag == 2) {
      if (stream_d2d == nullptr) {
        status = synStreamCreateGeneric(
            reinterpret_cast<synStreamHandle *>(&stream_d2d), device->id, 0);
        LOG_IF(ERROR, status != synSuccess)
            << "[RUNTIME] synStreamCreateGeneric() failed = " << status;
      }
      status = synMemCopyAsync(stream_d2d,
                               reinterpret_cast<uint64_t>(src),
                               size,
                               reinterpret_cast<uint64_t>(dst),
                               DRAM_TO_DRAM);
      LOG_IF(ERROR, status != synSuccess)
          << "[RUNTIME] synMemCopyAsync() failed = " << status;
      status = synStreamSynchronize(stream_d2d);
      LOG_IF(ERROR, status != synSuccess)
          << "[RUNTIME] synStreamSynchronize() failed = " << status;
    }
    return C_SUCCESS;
  }

  C_Status AsyncCopy(const C_Device device,
                     C_Stream stream,
                     void *dst,
                     const void *src,
                     size_t size,
                     size_t flag = 0 /*0 = h2d, 1 = d2h, 2=d2d*/) {
    // TODO: cache mapped host addr
    LOG_IF(INFO, FLAGS_intel_hpu_runtime_debug)
        << "AsyncCopy: flag = " << flag << ", size = " << size
        << ", stream = " << stream << ", src = " << src << ", dst = " << dst;
    synStatus status = synFail;
    if (flag == 0) {
      status = synHostMap(device->id, size, src);
      LOG_IF(ERROR, status != synSuccess)
          << "[RUNTIME] synHostMap() failed = " << status;
      status = synMemCopyAsync(reinterpret_cast<synStreamHandle>(stream),
                               reinterpret_cast<uint64_t>(src),
                               size,
                               reinterpret_cast<uint64_t>(dst),
                               HOST_TO_DRAM);
      LOG_IF(ERROR, status != synSuccess)
          << "[RUNTIME] synMemCopyAsync() failed = " << status;
    } else if (flag == 1) {
      status = synHostMap(device->id, size, dst);
      LOG_IF(ERROR, status != synSuccess)
          << "[RUNTIME] synHostMap() failed = " << status;
      status = synMemCopyAsync(reinterpret_cast<synStreamHandle>(stream),
                               reinterpret_cast<uint64_t>(src),
                               size,
                               reinterpret_cast<uint64_t>(dst),
                               DRAM_TO_HOST);
      LOG_IF(ERROR, status != synSuccess)
          << "[RUNTIME] synMemCopyAsync() failed = " << status;
    } else if (flag == 2) {
      status = synMemCopyAsync(reinterpret_cast<synStreamHandle>(stream),
                               reinterpret_cast<uint64_t>(src),
                               size,
                               reinterpret_cast<uint64_t>(dst),
                               DRAM_TO_DRAM);
      LOG_IF(ERROR, status != synSuccess)
          << "[RUNTIME] synMemCopyAsync() failed = " << status;
    }
    return C_SUCCESS;
  }

  C_Status CreateEvent(const C_Device device, C_Event *event) {
    DEBUG_LOG
    LOG_IF(ERROR, static_cast<synModuleId>(device->id) != moduleID)
        << "[RUNTIME] moduleID mismatch : moduleID = " << moduleID
        << ", current = " << moduleID;

    auto it = events.find(reinterpret_cast<synEventHandle>(*event));
    if (it == events.end()) {
      synEventHandle e = nullptr;
      synStatus status = synEventCreate(&e, deviceId, 0);
      LOG_IF(ERROR, status != synSuccess)
          << "[RUNTIME] synEventCreate() failed = " << status;
      events.insert(e);
      *event = reinterpret_cast<C_Event>(e);
    }

    LOG_IF(INFO, FLAGS_intel_hpu_runtime_debug)
        << "device id=" << device->id << " create event = " << *event;

    DEBUG_LOG

    return C_SUCCESS;
  }

  C_Status DestroyEvent(const C_Device device, C_Event event) {
    DEBUG_LOG
    LOG_IF(ERROR, static_cast<synModuleId>(device->id) != moduleID)
        << "[RUNTIME] moduleID mismatch : moduleID = " << moduleID
        << ", current = " << moduleID;
    auto it = events.find(reinterpret_cast<synEventHandle>(event));
    if (it != events.end()) {
      synStatus status = synEventDestroy(*it);
      LOG_IF(ERROR, status != synSuccess)
          << "[RUNTIME] destroyEvent() failed = " << status;
      events.erase(*it);
    }
    LOG_IF(INFO, FLAGS_intel_hpu_runtime_debug)
        << "device id=" << device->id << " remove event = " << event;

    DEBUG_LOG
    return C_SUCCESS;
  }

  inline synModuleId GetModuleID() { return moduleID; }
  inline synDeviceId GetDeviceID() { return deviceId; }

 private:
  synModuleId moduleID = 0;
  std::string busID = "";
  synDeviceId deviceId = 0;
  uint32_t Status = 0;  // 1 acquire, 0 not acquire
  uint32_t count = 0;

  // hccl
  hcclUniqueId uid;

  // user streams
  std::set<synStreamHandle> streams;

  // builtin stream
  synStreamHandle stream_h2d = nullptr;
  synStreamHandle stream_d2h = nullptr;
  synStreamHandle stream_d2d = nullptr;

  // user events
  std::set<synEventHandle> events;

  // cache
  std::unordered_map<void *, size_t> hostMappedAddress;
};

static RuntimeManager runtimeManager;

C_Status Init() {
  DEBUG_LOG
  synStatus status = synInitialize();
  LOG_IF(ERROR, status != synSuccess)
      << "[RUNTIME] synInitialize() failed: [" << status << "]";

  DEBUG_LOG
  return C_SUCCESS;
}

C_Status InitDevice(const C_Device device) {
  DEBUG_LOG
  runtimeManager.SetDevice(device);
  DEBUG_LOG
  return C_SUCCESS;
}

C_Status SetDevice(const C_Device device) {
  DEBUG_LOG
  runtimeManager.SetDevice(device);
  DEBUG_LOG
  return C_SUCCESS;
}

C_Status GetDevice(const C_Device device) {
  DEBUG_LOG
  device->id = runtimeManager.GetDevice();
  DEBUG_LOG
  return C_SUCCESS;
}

C_Status DestroyDevice(const C_Device device) {
  DEBUG_LOG
  runtimeManager.Release(device);
  DEBUG_LOG
  return C_SUCCESS;
}

C_Status Finalize() {
  DEBUG_LOG

  synStatus status = synDestroy();
  LOG_IF(ERROR, status != synSuccess)
      << "[RUNTIME] synDestroy() failed: [" << status << "]";
  DEBUG_LOG
  return C_SUCCESS;
}

C_Status GetDevicesCount(size_t *count) {
  DEBUG_LOG
  runtimeManager.GetNumDevices(count);
  DEBUG_LOG
  return C_SUCCESS;
}

C_Status GetDevicesList(size_t *devices) {
  DEBUG_LOG

  // TODO: suse HABANA_VISIBLE_DEVICES to get available device
  devices[0] = 0;
  // devices[1] = 1;
  DEBUG_LOG
  return C_SUCCESS;
}

C_Status MemCpyH2D(const C_Device device,
                   void *dst,
                   const void *src,
                   size_t size) {
  DEBUG_LOG
  runtimeManager.Copy(device, dst, src, size, 0);
  DEBUG_LOG
  return C_SUCCESS;
}

C_Status MemCpyD2H(const C_Device device,
                   void *dst,
                   const void *src,
                   size_t size) {
  DEBUG_LOG
  runtimeManager.Copy(device, dst, src, size, 1);
  DEBUG_LOG
  return C_SUCCESS;
}

C_Status MemCpyD2D(const C_Device device,
                   void *dst,
                   const void *src,
                   size_t size) {
  DEBUG_LOG
  runtimeManager.Copy(device, dst, src, size, 2);
  DEBUG_LOG
  return C_SUCCESS;
}

C_Status AsyncMemCpyH2D(const C_Device device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size) {
  DEBUG_LOG
  runtimeManager.AsyncCopy(device, stream, dst, src, size, 0);
  DEBUG_LOG
  return C_SUCCESS;
}

C_Status AsyncMemCpyD2H(const C_Device device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size) {
  DEBUG_LOG
  runtimeManager.AsyncCopy(device, stream, dst, src, size, 1);
  DEBUG_LOG
  return C_SUCCESS;
}

C_Status AsyncMemCpyD2D(const C_Device device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size) {
  DEBUG_LOG
  runtimeManager.AsyncCopy(device, stream, dst, src, size, 2);
  DEBUG_LOG
  return C_SUCCESS;
}

C_Status Allocate_device(const C_Device device, void **ptr, size_t size) {
  DEBUG_LOG
  uint64_t p;
  synStatus status =
      synDeviceMalloc(runtimeManager.GetDeviceID(), size, 0, 0, &p);
  LOG_IF(ERROR, status != synSuccess)
      << "[RUNTIME] hbmAlloc() failed = " << status;
  *ptr = reinterpret_cast<void *>(p);
  LOG_IF(INFO, FLAGS_intel_hpu_runtime_debug)
      << "device id = " << runtimeManager.GetDeviceID()
      << " malloc ptr=" << *ptr << " size=" << size;

  DEBUG_LOG
  return C_SUCCESS;
}

C_Status Deallocate_device(const C_Device device, void *ptr, size_t size) {
  DEBUG_LOG
  LOG_IF(INFO, FLAGS_intel_hpu_runtime_debug)
      << "device id=" << runtimeManager.GetDeviceID() << " free ptr = " << ptr
      << " size=" << size;

  synStatus status = synDeviceFree(
      runtimeManager.GetDeviceID(), *reinterpret_cast<uint64_t *>(ptr), 0);
  LOG_IF(ERROR, status != synSuccess)
      << "[RUNTIME] hbmFree() failed = " << status;

  DEBUG_LOG
  return C_SUCCESS;
}

C_Status Allocate_host(const C_Device device, void **ptr, size_t size) {
  DEBUG_LOG
  synStatus status = synHostMalloc(runtimeManager.GetDeviceID(), size, 0, ptr);
  LOG_IF(ERROR, status != synSuccess)
      << "[RUNTIME] synHostMalloc() failed = " << status;

  DEBUG_LOG
  return C_FAILED;
}

C_Status Deallocate_host(const C_Device device, void *ptr, size_t size) {
  DEBUG_LOG
  synStatus status = synHostFree(runtimeManager.GetDeviceID(), ptr, 0);
  LOG_IF(ERROR, status != synSuccess)
      << "[RUNTIME] synHostFree() failed = " << status;
  DEBUG_LOG
  return C_SUCCESS;
}

C_Status CreateStream(const C_Device device, C_Stream *stream) {
  DEBUG_LOG
  runtimeManager.CreateStream(device, stream);
  DEBUG_LOG
  return C_SUCCESS;
}

C_Status DestroyStream(const C_Device device, C_Stream stream) {
  DEBUG_LOG
  runtimeManager.DestroyStream(device, stream);
  LOG_IF(INFO, FLAGS_intel_hpu_runtime_debug)
      << "device id=" << device->id << " stream=" << stream;
  DEBUG_LOG
  return C_SUCCESS;
}

C_Status CreateEvent(const C_Device device, C_Event *event) {
  DEBUG_LOG
  runtimeManager.CreateEvent(device, event);
  DEBUG_LOG
  return C_SUCCESS;
}

C_Status RecordEvent(const C_Device device, C_Stream stream, C_Event event) {
  DEBUG_LOG
  LOG_IF(ERROR,
         static_cast<synModuleId>(device->id) != runtimeManager.GetModuleID())
      << "[RUNTIME] moduleID mismatch : moduleID = "
      << runtimeManager.GetModuleID() << ", current = " << device->id;
  synStatus status =
      synEventRecord(reinterpret_cast<synEventHandle>(event),
                     reinterpret_cast<const synStreamHandle>(stream));
  LOG_IF(ERROR, status != synSuccess)
      << "[RUNTIME] synEventRecord() failed = " << status;

  DEBUG_LOG
  return C_SUCCESS;
}

C_Status DestroyEvent(const C_Device device, C_Event event) {
  DEBUG_LOG
  runtimeManager.DestroyEvent(device, event);
  DEBUG_LOG
  return C_SUCCESS;
}

C_Status SyncDevice(const C_Device device) {
  DEBUG_LOG
  LOG_IF(ERROR,
         static_cast<synModuleId>(device->id) != runtimeManager.GetModuleID())
      << "[RUNTIME] moduleID mismatch : moduleID = "
      << runtimeManager.GetModuleID() << ", current = " << device->id;

  synStatus status = synDeviceSynchronize(runtimeManager.GetDevice());
  LOG_IF(ERROR, status != synSuccess)
      << "[RUNTIME] synDeviceSynchronize() failed = " << status;

  DEBUG_LOG
  return C_SUCCESS;
}

C_Status SyncStream(const C_Device device, C_Stream stream) {
  DEBUG_LOG
  LOG_IF(INFO, FLAGS_intel_hpu_runtime_debug)
      << "SyncStream: " << static_cast<synModuleId>(device->id) << ", "
      << reinterpret_cast<const synStreamHandle>(stream);
  LOG_IF(ERROR,
         static_cast<synModuleId>(device->id) != runtimeManager.GetModuleID())
      << "[RUNTIME] moduleID mismatch : moduleID = "
      << runtimeManager.GetModuleID() << ", current = " << device->id;

  synStatus status =
      synStreamSynchronize(reinterpret_cast<const synStreamHandle>(stream));
  LOG_IF(ERROR, status != synSuccess)
      << "[RUNTIME] synStreamSynchronize() failed: [" << status << "]";

  DEBUG_LOG
  return C_SUCCESS;
}

C_Status SyncEvent(const C_Device device, C_Event event) {
  DEBUG_LOG
  LOG_IF(ERROR,
         static_cast<synModuleId>(device->id) != runtimeManager.GetModuleID())
      << "[RUNTIME] moduleID mismatch : moduleID = "
      << runtimeManager.GetModuleID() << ", current = " << device->id;

  synStatus status =
      synEventSynchronize(reinterpret_cast<const synEventHandle>(event));
  LOG_IF(ERROR, status != synSuccess)
      << "[RUNTIME] synEventSynchronize() failed: [" << status << "]";

  DEBUG_LOG
  return C_SUCCESS;
}

C_Status StreamWaitEvent(const C_Device device,
                         C_Stream stream,
                         C_Event event) {
  DEBUG_LOG
  LOG_IF(ERROR,
         static_cast<synModuleId>(device->id) != runtimeManager.GetModuleID())
      << "[RUNTIME] moduleID mismatch : moduleID = "
      << runtimeManager.GetModuleID() << ", current = " << device->id;

  synStatus status =
      synStreamWaitEvent(reinterpret_cast<const synStreamHandle>(stream),
                         reinterpret_cast<synEventHandle>(event),
                         0);
  LOG_IF(ERROR, status != synSuccess)
      << "[RUNTIME] synStreamWaitEvent() failed: [" << status << "]";

  DEBUG_LOG
  return C_SUCCESS;
}

C_Status DeviceMemStats(const C_Device device,
                        size_t *total_memory,
                        size_t *free_memory) {
  DEBUG_LOG
  runtimeManager.GetMemoryStats(device, total_memory, free_memory);
  DEBUG_LOG
  return C_SUCCESS;
}

C_Status DeviceMinChunkSize(const C_Device device, size_t *size) {
  DEBUG_LOG
  *size = 1;
  LOG_IF(INFO, FLAGS_intel_hpu_runtime_debug) << "min chunksize=" << *size;

  DEBUG_LOG
  return C_SUCCESS;
}

C_Status XcclGetUniqueIdSize(size_t *sz) {
  DEBUG_LOG

  DEBUG_LOG
  return C_SUCCESS;
}

C_Status XcclGetUniqueId(C_CCLRootId *unique_id) {
  DEBUG_LOG
  CHECK_HCCL_STATUS(
      hcclGetUniqueId(reinterpret_cast<hcclUniqueId *>(unique_id)));
  DEBUG_LOG
  return C_SUCCESS;
}

C_Status XcclCommInitRank(size_t ranks,
                          C_CCLRootId *unique_id,
                          size_t rank,
                          C_CCLComm *comm) {
  DEBUG_LOG
  hcclUniqueId uniqueId{};
  CHECK_HCCL_STATUS(hcclCommInitRank(
      reinterpret_cast<hcclComm_t *>(comm), ranks, uniqueId, rank));
  DEBUG_LOG
  return C_SUCCESS;
}

C_Status XcclDestroyComm(C_CCLComm comm) {
  DEBUG_LOG
  CHECK_HCCL_STATUS(hcclCommDestroy(reinterpret_cast<hcclComm_t>(comm)));
  DEBUG_LOG
  return C_SUCCESS;
}

C_Status XcclAllReduce(void *send_buf,
                       void *recv_buf,
                       size_t count,
                       C_DataType data_type,
                       C_CCLReduceOp op,
                       C_CCLComm comm,
                       C_Stream stream) {
  CHECK_HCCL_STATUS(hcclAllReduce(static_cast<const void *>(send_buf),
                                  recv_buf,
                                  count,
                                  PDDataTypeToHcclDataType(data_type),
                                  PDReduceOpToHcclReduceOp(op),
                                  reinterpret_cast<hcclComm_t>(comm),
                                  reinterpret_cast<synStreamHandle>(stream)));
  return C_SUCCESS;
}

C_Status XcclBroadcast(void *buf,
                       size_t count,
                       C_DataType data_type,
                       size_t root,
                       C_CCLComm comm,
                       C_Stream stream) {
  CHECK_HCCL_STATUS(hcclBroadcast(static_cast<const void *>(buf),
                                  buf,
                                  static_cast<uint64_t>(count),
                                  PDDataTypeToHcclDataType(data_type),
                                  static_cast<int>(root),
                                  reinterpret_cast<hcclComm_t>(comm),
                                  reinterpret_cast<synStreamHandle>(stream)));
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
  CHECK_HCCL_STATUS(hcclReduce(static_cast<const void *>(send_buf),
                               recv_buf,
                               count,
                               PDDataTypeToHcclDataType(data_type),
                               PDReduceOpToHcclReduceOp(op),
                               root,
                               reinterpret_cast<hcclComm_t>(comm),
                               reinterpret_cast<synStreamHandle>(stream)));
  return C_SUCCESS;
}

C_Status XcclAllGather(void *send_buf,
                       void *recv_buf,
                       size_t count,
                       C_DataType data_type,
                       C_CCLComm comm,
                       C_Stream stream) {
  CHECK_HCCL_STATUS(hcclAllGather(static_cast<const void *>(send_buf),
                                  recv_buf,
                                  count,
                                  PDDataTypeToHcclDataType(data_type),
                                  reinterpret_cast<hcclComm_t>(comm),
                                  reinterpret_cast<synStreamHandle>(stream)));
  return C_SUCCESS;
}

C_Status XcclReduceScatter(void *send_buf,
                           void *recv_buf,
                           size_t count,
                           C_DataType data_type,
                           C_CCLReduceOp op,
                           C_CCLComm comm,
                           C_Stream stream) {
  CHECK_HCCL_STATUS(
      hcclReduceScatter(static_cast<const void *>(send_buf),
                        recv_buf,
                        count,
                        PDDataTypeToHcclDataType(data_type),
                        PDReduceOpToHcclReduceOp(op),
                        reinterpret_cast<hcclComm_t>(comm),
                        reinterpret_cast<synStreamHandle>(stream)));
  return C_SUCCESS;
}

C_Status XcclGroupStart() {
  CHECK_HCCL_STATUS(hcclGroupStart());
  return C_SUCCESS;
}

C_Status XcclGroupEnd() {
  CHECK_HCCL_STATUS(hcclGroupEnd());
  return C_SUCCESS;
}

C_Status XcclSend(void *send_buf,
                  size_t count,
                  C_DataType data_type,
                  size_t dest_rank,
                  C_CCLComm comm,
                  C_Stream stream) {
  CHECK_HCCL_STATUS(hcclSend(static_cast<const void *>(send_buf),
                             count,
                             PDDataTypeToHcclDataType(data_type),
                             static_cast<uint32_t>(dest_rank),
                             reinterpret_cast<hcclComm_t>(comm),
                             reinterpret_cast<synStreamHandle>(stream)));
  return C_SUCCESS;
}

C_Status XcclRecv(void *recv_buf,
                  size_t count,
                  C_DataType data_type,
                  size_t src_rank,
                  C_CCLComm comm,
                  C_Stream stream) {
  CHECK_HCCL_STATUS(hcclRecv(recv_buf,
                             count,
                             PDDataTypeToHcclDataType(data_type),
                             static_cast<uint32_t>(src_rank),
                             reinterpret_cast<hcclComm_t>(comm),
                             reinterpret_cast<synStreamHandle>(stream)));
  return C_SUCCESS;
}

C_Status ProfilerInitialize(C_Profiler prof, void **user_data) {
  DEBUG_LOG

  DEBUG_LOG
  return C_SUCCESS;
}

C_Status ProfilerFinalize(C_Profiler prof, void *user_data) {
  DEBUG_LOG

  DEBUG_LOG
  return C_SUCCESS;
}

C_Status ProfilerPrepare(C_Profiler prof, void *user_data) {
  DEBUG_LOG

  DEBUG_LOG
  return C_SUCCESS;
}

C_Status ProfilerStart(C_Profiler prof, void *user_data) {
  DEBUG_LOG

  DEBUG_LOG
  return C_SUCCESS;
}

C_Status ProfilerStop(C_Profiler prof, void *user_data) {
  DEBUG_LOG

  DEBUG_LOG
  return C_SUCCESS;
}

C_Status ProfilerCollectData(C_Profiler prof,
                             uint64_t start_ns,
                             void *user_data) {
  DEBUG_LOG

  DEBUG_LOG
  return C_SUCCESS;
}

void InitPlugin(CustomRuntimeParams *params) {
  DEBUG_LOG
  PADDLE_CUSTOM_RUNTIME_CHECK_VERSION(params);
  params->version.major = 1;
  params->version.minor = 16;
  params->version.patch = 0;
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

  params->interface->memory_copy_h2d = MemCpyH2D;
  params->interface->memory_copy_d2d = MemCpyD2D;
  params->interface->memory_copy_d2h = MemCpyD2H;
  // params->interface->memory_copy_p2p = MemCpyP2P;
  params->interface->async_memory_copy_h2d = AsyncMemCpyH2D;
  params->interface->async_memory_copy_d2d = AsyncMemCpyD2D;
  params->interface->async_memory_copy_d2h = AsyncMemCpyD2H;
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

  params->interface->profiler_collect_trace_data = ProfilerCollectData;
  params->interface->profiler_initialize = ProfilerInitialize;
  params->interface->profiler_finalize = ProfilerFinalize;
  params->interface->profiler_start_tracing = ProfilerStart;
  params->interface->profiler_stop_tracing = ProfilerStop;
  params->interface->profiler_prepare_tracing = ProfilerPrepare;

  DEBUG_LOG
}
