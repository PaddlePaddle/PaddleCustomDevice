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

#include "runtime/runtime.h"

#include <eccl/eccl.h>

#include <cstring>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>  // NOLINT
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "backend/executor/cast_runner.h"
#include "dtu/hlir/dispatch.h"
#include "glog/logging.h"
#include "paddle/phi/capi/include/type_utils.h"
#include "runtime/flags.h"
#include "runtime/gcu_memory.h"

namespace {
const char *const kDeviceType = "gcu";
const char *const kSubDeviceType = "none";

static std::set<void *> scatter_memorys;
std::mutex scatter_mem_mutex_;

template <typename T>
static std::string VectorToString(std::vector<T> vec) {
  std::ostringstream os;
  os << "[";
  for (auto tmp : vec) {
    os << std::fixed << tmp << "; ";
  }
  os << "]";
  return os.str();
}

inline size_t get_devices_count() {
  int count;
  RT_CHECK(topsGetDeviceCount(&count));
  return static_cast<size_t>(count);
}

inline int get_current_device_id() {
  int device = 0;
  RT_CHECK(topsGetDevice(&device));
  return device;
}
}  // namespace

thread_local int g_current_device_id(-1);

static std::unordered_map<int32_t, topsResource_t> res_bundles_;
std::mutex mutex_;

// Device
C_Status Init() {
  size_t dev_cnt = get_devices_count();
  VLOG(0) << "Backend GCU Init, get GCU count:" << dev_cnt;
  return C_SUCCESS;
}

C_Status Finalize() {
  VLOG(0) << "Backend GCU Finalize";
  return C_SUCCESS;
}

C_Status InitDevice(const C_Device device) {
  RT_CHECK(topsSetDevice(device->id));
  if (UseScatterMemory()) {
    InitResource(device->id);
  }
  VLOG(0) << "Backend GCU init device:" << device->id;
  g_current_device_id = device->id;
  return C_SUCCESS;
}

C_Status SetDevice(const C_Device device) {
  RT_CHECK(topsSetDevice(device->id));
  g_current_device_id = device->id;
  return C_SUCCESS;
}

C_Status GetDevice(const C_Device device) {
  RT_CHECK(topsGetDevice(&(device->id)));
  g_current_device_id = device->id;
  return C_SUCCESS;
}

C_Status DeInitDevice(const C_Device device) {
  if (UseScatterMemory()) {
    FinalizeResource();
  }
  return C_SUCCESS;
}

// Stream
C_Status CreateStream(const C_Device device, C_Stream *stream) {
  GcuDeviceGuard guard(device->id);
  RT_CHECK(topsStreamCreate(reinterpret_cast<topsStream_t *>(stream)));
  return C_SUCCESS;
}

C_Status DestroyStream(const C_Device device, C_Stream stream) {
  GcuDeviceGuard guard(device->id);
  RT_CHECK(topsStreamDestroy(reinterpret_cast<topsStream_t>(stream)));
  return C_SUCCESS;
}

static void StreamCallbackFunc(topsStream_t stream,
                               topsError_t status,
                               void *user_data) {
  std::unique_ptr<std::function<void()>> func(
      reinterpret_cast<std::function<void()> *>(user_data));
  (*func)();
}

// user_data is phi::stream::Stream::Callback, which is std::function<void()>
// defined at paddle/phi/backends/stream.h
// For more information about this interface, please refer to
// CustomDevice::AddCallback at paddle/phi/backends/custom/custom_device.cc
C_Status AddCallback(const C_Device device,
                     C_Stream stream,
                     C_Callback callback,
                     void *user_data) {
  RT_CHECK(topsStreamAddCallback(reinterpret_cast<topsStream_t>(stream),
                                 StreamCallbackFunc,
                                 user_data,
                                 topsStreamDefault));
  return C_SUCCESS;
}

// Event
C_Status CreateEvent(const C_Device device, C_Event *event) {
  GcuDeviceGuard guard(device->id);
  RT_CHECK(topsEventCreate(reinterpret_cast<topsEvent_t *>(event)));
  return C_SUCCESS;
}

C_Status RecordEvent(const C_Device device, C_Stream stream, C_Event event) {
  GcuDeviceGuard guard(device->id);
  RT_CHECK(topsEventRecord(reinterpret_cast<topsEvent_t>(event),
                           reinterpret_cast<topsStream_t>(stream)));
  return C_SUCCESS;
}

C_Status DestroyEvent(const C_Device device, C_Event event) {
  GcuDeviceGuard guard(device->id);
  RT_CHECK(topsEventDestroy(reinterpret_cast<topsEvent_t>(event)));
  return C_SUCCESS;
}

// Synchronize
C_Status SyncDevice(const C_Device device) {
  GcuDeviceGuard guard(device->id);
  RT_CHECK(topsDeviceSynchronize());
  return C_SUCCESS;
}

C_Status SyncStream(const C_Device device, C_Stream stream) {
  GcuDeviceGuard guard(device->id);
  RT_CHECK(topsStreamSynchronize(reinterpret_cast<topsStream_t>(stream)));
  return C_SUCCESS;
}

C_Status SyncEvent(const C_Device device, C_Event event) {
  GcuDeviceGuard guard(device->id);
  RT_CHECK(topsEventSynchronize(reinterpret_cast<topsEvent_t>(event)));
  return C_SUCCESS;
}

C_Status StreamWaitEvent(const C_Device device,
                         C_Stream stream,
                         C_Event event) {
  GcuDeviceGuard guard(device->id);
  RT_CHECK(topsStreamWaitEvent(reinterpret_cast<topsStream_t>(stream),
                               reinterpret_cast<topsEvent_t>(event),
                               0));
  return C_SUCCESS;
}

// Memory
C_Status Allocate(const C_Device device, void **ptr, size_t size) {
  GcuDeviceGuard guard(device->id);
  void *tmp_ptr;

  if (UseScatterMemory()) {
    auto ret = topsMallocScatter(&tmp_ptr, size);
    VLOG(6) << "alloc scatter memory ptr: " << tmp_ptr;
    if (ret == topsSuccess) {
      auto *gcu_mem = new GcuMemory(tmp_ptr, unset_flag_dims, size);
      *ptr = gcu_mem;
      VLOG(3) << "[AllocFromGCU] Alloc gcu hbm size:" << size
              << " ptr: " << *ptr << " mem_ptr: " << tmp_ptr;
      std::lock_guard<std::mutex> lock(scatter_mem_mutex_);
      scatter_memorys.insert(*ptr);
      return C_SUCCESS;
    }
  } else {
    auto ret = topsMalloc(&tmp_ptr, size);
    if (ret == topsSuccess) {
      *ptr = tmp_ptr;
      VLOG(3) << "[AllocFromGCU] Alloc gcu hbm size:" << size;
      return C_SUCCESS;
    }
  }

  *ptr = nullptr;
  return C_FAILED;
}

C_Status Deallocate(const C_Device device, void *ptr, size_t size) {
  GcuDeviceGuard guard(device->id);
  if (UseScatterMemory()) {
    if (ptr != nullptr) {
      if (UseScatterMemory()) {
        std::lock_guard<std::mutex> lock(scatter_mem_mutex_);
        scatter_memorys.erase(ptr);
      }
      delete static_cast<GcuMemory *>(ptr);
      ptr = nullptr;
    }
  } else {
    RT_CHECK(topsFree(ptr));
  }

  VLOG(3) << "[FreeToGCU] Free gcu hbm size:" << size;
  return C_SUCCESS;
}

C_Status HostAllocate(const C_Device device, void **ptr, size_t size) {
  GcuDeviceGuard guard(device->id);
  void *gcu_host_mem;
  // NOTE: If size is 0, no memory is allocated, *ptr returns nullptr, and
  // topsSuccess is returned.
  auto ret = topsHostMalloc(&gcu_host_mem, size, topsHostMallocDefault);
  if (ret == topsSuccess) {
    *ptr = gcu_host_mem;
    VLOG(3) << "[AllocFromGCU] Alloc gcu host memory size:" << size;
    return C_SUCCESS;
  }
  *ptr = nullptr;
  return C_FAILED;
}

C_Status HostDeallocate(const C_Device device, void *ptr, size_t size) {
  GcuDeviceGuard guard(device->id);
  RT_CHECK(topsHostFree(ptr));
  VLOG(3) << "[FreeToGCU] Free gcu host memory size:" << size;
  return C_SUCCESS;
}

C_Status MemCpyH2D(const C_Device device,
                   void *dst,
                   const void *src,
                   size_t size) {
  GcuDeviceGuard guard(device->id);
  if (UseScatterMemory()) {
    auto *dst_ptr = static_cast<GcuMemory *>(dst)->mem_ptr;
    RT_CHECK(topsMemcpy(dst_ptr, src, size, topsMemcpyHostToDevice));
  } else {
    RT_CHECK(topsMemcpy(dst, src, size, topsMemcpyHostToDevice));
  }
  return C_SUCCESS;
}

C_Status MemCpyD2H(const C_Device device,
                   void *dst,
                   const void *src,
                   size_t size) {
  GcuDeviceGuard guard(device->id);
  if (UseScatterMemory()) {
    auto *src_ptr = static_cast<GcuMemory *>(const_cast<void *>(src))->mem_ptr;
    RT_CHECK(topsMemcpy(dst, src_ptr, size, topsMemcpyDeviceToHost));
  } else {
    RT_CHECK(topsMemcpy(dst, src, size, topsMemcpyDeviceToHost));
  }
  return C_SUCCESS;
}

C_Status MemCpyD2D(const C_Device device,
                   void *dst,
                   const void *src,
                   size_t size) {
  GcuDeviceGuard guard(device->id);
  if (UseScatterMemory()) {
    auto *src_ptr = static_cast<GcuMemory *>(const_cast<void *>(src))->mem_ptr;
    auto *dst_ptr = static_cast<GcuMemory *>(dst)->mem_ptr;
    RT_CHECK(topsMemcpy(dst_ptr, src_ptr, size, topsMemcpyDeviceToDevice));
  } else {
    RT_CHECK(topsMemcpy(dst, src, size, topsMemcpyDeviceToDevice));
  }
  return C_SUCCESS;
}

C_Status AsyncMemCpyH2D(const C_Device device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size) {
  GcuDeviceGuard guard(device->id);
  if (UseScatterMemory()) {
    auto *dst_ptr = static_cast<GcuMemory *>(dst)->mem_ptr;
    RT_CHECK(topsMemcpyAsync(dst_ptr,
                             src,
                             size,
                             topsMemcpyHostToDevice,
                             reinterpret_cast<topsStream_t>(stream)));
  } else {
    RT_CHECK(topsMemcpyAsync(dst,
                             src,
                             size,
                             topsMemcpyHostToDevice,
                             reinterpret_cast<topsStream_t>(stream)));
  }
  return C_SUCCESS;
}

C_Status AsyncMemCpyD2H(const C_Device device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size) {
  GcuDeviceGuard guard(device->id);
  if (UseScatterMemory()) {
    auto *src_ptr = static_cast<GcuMemory *>(const_cast<void *>(src))->mem_ptr;
    RT_CHECK(topsMemcpyAsync(dst,
                             src_ptr,
                             size,
                             topsMemcpyDeviceToHost,
                             reinterpret_cast<topsStream_t>(stream)));
  } else {
    RT_CHECK(topsMemcpyAsync(dst,
                             src,
                             size,
                             topsMemcpyDeviceToHost,
                             reinterpret_cast<topsStream_t>(stream)));
  }
  return C_SUCCESS;
}

C_Status AsyncMemCpyD2D(const C_Device device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size) {
  GcuDeviceGuard guard(device->id);
  if (UseScatterMemory()) {
    void *src_ptr = const_cast<void *>(src);
    void *dst_ptr = dst;
    if (scatter_memorys.count(src_ptr) > 0) {  // for dist concat & split
      src_ptr = static_cast<GcuMemory *>(const_cast<void *>(src))->mem_ptr;
    }
    if (scatter_memorys.count(dst_ptr) > 0) {
      dst_ptr = static_cast<GcuMemory *>(dst)->mem_ptr;
    }
    RT_CHECK(topsMemcpyAsync(dst_ptr,
                             src_ptr,
                             size,
                             topsMemcpyDeviceToDevice,
                             reinterpret_cast<topsStream_t>(stream)));
  } else {
    RT_CHECK(topsMemcpyAsync(dst,
                             src,
                             size,
                             topsMemcpyDeviceToDevice,
                             reinterpret_cast<topsStream_t>(stream)));
  }
  return C_SUCCESS;
}

// Get visible device info
C_Status GetDevicesCount(size_t *count) {
  *count = get_devices_count();
  return C_SUCCESS;
}

C_Status GetDevicesList(size_t *device) {
  size_t count = get_devices_count();
  for (size_t dev_id = 0; dev_id < count; ++dev_id) {
    device[dev_id] = dev_id;
  }
  return C_SUCCESS;
}

C_Status DeviceMinChunkSize(const C_Device device, size_t *size) {
  *size = 256;
  return C_SUCCESS;
}

C_Status DeviceMaxChunkSize(const C_Device device, size_t *size) {
  *size = 0;
  return C_SUCCESS;
}

C_Status ExtraPaddingSize(const C_Device device, size_t *size) {
  *size = 32;
  return C_SUCCESS;
}

// CCL
namespace {
const std::unordered_map<C_CCLReduceOp, ecclRedOp_t> kEcclOpMap = {
    {C_CCLReduceOp::SUM, ecclSum},
    {C_CCLReduceOp::MAX, ecclMax},
    {C_CCLReduceOp::MIN, ecclMin},
    {C_CCLReduceOp::PRODUCT, ecclProd},
};

ecclRedOp_t ConvertEcclReduceOp(const C_CCLReduceOp &reduce_op) {
  try {
    return kEcclOpMap.at(reduce_op);
  } catch (const std::out_of_range &e) {
    throw std::runtime_error(
        "Failed to ConvertEcclReduceOp, unsupport reduce_op");
  }
}

const std::unordered_map<C_DataType, ecclDataType_t> kEcclDtypeMap = {
    {C_DataType::UINT8, ecclUint8},
    {C_DataType::UINT32, ecclUint32},
    {C_DataType::UINT64, ecclUint64},
    {C_DataType::INT8, ecclInt8},
    {C_DataType::INT32, ecclInt32},
    {C_DataType::INT64, ecclInt64},
    {C_DataType::FLOAT16, ecclFloat16},
    {C_DataType::FLOAT32, ecclFloat32},
    {C_DataType::FLOAT64, ecclFloat64},
    {C_DataType::BFLOAT16, ecclBFloat16},
};

ecclDataType_t ConvertEcclDataType(const C_DataType &dtype) {
  try {
    return kEcclDtypeMap.at(dtype);
  } catch (const std::out_of_range &e) {
    throw std::runtime_error("Failed to ConvertEcclDataType, unsupport dtype");
  }
}

const std::unordered_map<C_DataType, std::string> kDtypeToStr = {
    {C_DataType::UNDEFINED, "UNDEFINED"},
    {C_DataType::BOOL, "BOOL"},
    {C_DataType::UINT8, "UINT8"},
    {C_DataType::UINT16, "UINT16"},
    {C_DataType::UINT32, "UINT32"},
    {C_DataType::UINT64, "UINT64"},
    {C_DataType::INT8, "INT8"},
    {C_DataType::INT16, "INT16"},
    {C_DataType::INT32, "INT32"},
    {C_DataType::INT64, "INT64"},
    {C_DataType::FLOAT16, "FLOAT16"},
    {C_DataType::FLOAT32, "FLOAT32"},
    {C_DataType::FLOAT64, "FLOAT64"},
    {C_DataType::BFLOAT16, "BFLOAT16"},
    {C_DataType::COMPLEX64, "COMPLEX64"},
    {C_DataType::COMPLEX128, "COMPLEX128"},
};

std::string DataTypeToStr(const C_DataType &dtype) {
  try {
    return kDtypeToStr.at(dtype);
  } catch (const std::out_of_range &e) {
    throw std::runtime_error("DataTypeToStr failed, unsupport dtype");
  }
}

std::string SerializeUniqueId(C_CCLRootId *unique_id) {
  const uint8_t *bytes = reinterpret_cast<const uint8_t *>(unique_id->data);
  std::ostringstream oss;
  for (size_t i = 0; i < unique_id->sz; ++i) {
    oss << std::hex << static_cast<int>(bytes[i]);
  }
  return oss.str();
}
}  // namespace

C_Status XcclGetUniqueIdSize(size_t *size) {
  *size = sizeof(ecclUniqueId);
  return C_SUCCESS;
}

C_Status XcclGetUniqueId(C_CCLRootId *unique_id) {
  if (unique_id->sz != sizeof(ecclUniqueId)) {
    VLOG(0) << "unique_id->sz must be equal sizeof(ecclUniqueId)";
    return C_FAILED;
  }
  ECCL_CHECK(
      ecclGetUniqueId(reinterpret_cast<ecclUniqueId *>(unique_id->data)));
  VLOG(0) << "Backend GCU GetUniqueId, UniqueId size:" << unique_id->sz
          << ", UniqueId:" << SerializeUniqueId(unique_id);
  return C_SUCCESS;
}

C_Status XcclCommInitRank(size_t nranks,
                          C_CCLRootId *unique_id,
                          size_t rank,
                          C_CCLComm *comm) {
  ECCL_CHECK(
      ecclCommInitRank(reinterpret_cast<ecclComm_t *>(comm),
                       nranks,
                       *(reinterpret_cast<ecclUniqueId *>(unique_id->data)),
                       rank));
  VLOG(0) << "Backend GCU CommInitRank successfully, world size:" << nranks
          << ", rank:" << rank << ", current device id:" << g_current_device_id;
  return C_SUCCESS;
}

C_Status XcclDestroyComm(C_CCLComm comm) {
  ECCL_CHECK(ecclCommDestroy(reinterpret_cast<ecclComm_t>(comm)));
  VLOG(0) << "Backend GCU CommDestroy successfully, current device id:"
          << g_current_device_id;
  return C_SUCCESS;
}

C_DataType CastRealDataType(C_DataType data_type) {
  switch (data_type) {
    case C_DataType::INT64:
      return C_DataType::INT32;
    case C_DataType::UINT64:
      return C_DataType::UINT32;
    case C_DataType::FLOAT64:
      return C_DataType::FLOAT32;

    default:
      return data_type;
  }
}

C_Status XcclAllReduce(void *send_buf,
                       void *recv_buf,
                       size_t count,
                       C_DataType data_type,
                       C_CCLReduceOp op,
                       C_CCLComm comm,
                       C_Stream stream) {
  void *real_send_buf = send_buf;
  void *real_recv_buf = recv_buf;
  if (UseScatterMemory()) {
    if (scatter_memorys.count(send_buf) > 0) {
      real_send_buf = static_cast<GcuMemory *>(send_buf)->mem_ptr;
    }
    if (scatter_memorys.count(recv_buf) > 0) {
      real_recv_buf = static_cast<GcuMemory *>(recv_buf)->mem_ptr;
    }
  }
  VLOG(6) << "real_send_buf " << real_send_buf << " send_buf " << send_buf;
  VLOG(6) << "real_recv_buf " << real_recv_buf << " recv_buf " << recv_buf;
  C_DataType real_data_type = CastRealDataType(data_type);

  if (data_type != real_data_type) {
    phi::DataType dtype = phi::capi::ToPhiDataType(data_type);
    phi::DataType real_dtype = phi::capi::ToPhiDataType(real_data_type);

    VLOG(6) << "XcclAllReduce data_type: " << phi::DataTypeToString(dtype)
            << ", real_data_type: " << phi::DataTypeToString(real_dtype);
    static std::unordered_map<std::string, void *> device_memorys;
    std::string key = phi::DataTypeToString(real_dtype) + " " +
                      std::to_string(count * phi::SizeOf(real_dtype));
    void *tmp_buf = nullptr;
    if (device_memorys.count(key) <= 0) {
      C_Device_st device;
      device.id = get_current_device_id();
      Allocate(&device, &tmp_buf, count * phi::SizeOf(real_dtype));
      device_memorys[key] = tmp_buf;
    } else {
      tmp_buf = device_memorys[key];
    }

    backend::CastRunner(reinterpret_cast<topsStream_t>(stream),
                        {static_cast<int64_t>(count)},
                        dtype,
                        real_dtype,
                        real_send_buf,
                        tmp_buf);

    ECCL_CHECK(ecclAllReduce(tmp_buf,
                             tmp_buf,
                             count,
                             ConvertEcclDataType(real_data_type),
                             ConvertEcclReduceOp(op),
                             reinterpret_cast<ecclComm_t>(comm),
                             reinterpret_cast<topsStream_t>(stream)));

    backend::CastRunner(reinterpret_cast<topsStream_t>(stream),
                        {static_cast<int64_t>(count)},
                        real_dtype,
                        dtype,
                        tmp_buf,
                        real_recv_buf);
  } else {
    ECCL_CHECK(ecclAllReduce(real_send_buf,
                             real_recv_buf,
                             count,
                             ConvertEcclDataType(data_type),
                             ConvertEcclReduceOp(op),
                             reinterpret_cast<ecclComm_t>(comm),
                             reinterpret_cast<topsStream_t>(stream)));
  }

  VLOG(6) << "Backend GCU XcclAllReduce successfully, data_type:"
          << DataTypeToStr(data_type) << ", count:" << count;
  return C_SUCCESS;
}

C_Status XcclBroadcast(void *buf,
                       size_t count,
                       C_DataType data_type,
                       size_t root,
                       C_CCLComm comm,
                       C_Stream stream) {
  void *real_buf = buf;
  if (UseScatterMemory()) {
    real_buf = static_cast<GcuMemory *>(buf)->mem_ptr;
  }
  ECCL_CHECK(ecclBroadcast(real_buf,
                           real_buf,
                           count,
                           ConvertEcclDataType(data_type),
                           static_cast<int>(root),
                           reinterpret_cast<ecclComm_t>(comm),
                           reinterpret_cast<topsStream_t>(stream)));
  VLOG(6) << "Backend GCU XcclBroadcast successfully, data_type:"
          << DataTypeToStr(data_type) << ", count:" << count;
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
  void *real_send_buf = send_buf;
  void *real_recv_buf = recv_buf;
  if (UseScatterMemory()) {
    real_send_buf = static_cast<GcuMemory *>(send_buf)->mem_ptr;
    real_recv_buf = static_cast<GcuMemory *>(recv_buf)->mem_ptr;
  }
  ECCL_CHECK(ecclReduce(real_send_buf,
                        real_recv_buf,
                        count,
                        ConvertEcclDataType(data_type),
                        ConvertEcclReduceOp(op),
                        static_cast<int>(root),
                        reinterpret_cast<ecclComm_t>(comm),
                        reinterpret_cast<topsStream_t>(stream)));
  return C_SUCCESS;
}

C_Status XcclAllGather(void *send_buf,
                       void *recv_buf,
                       size_t count,
                       C_DataType data_type,
                       C_CCLComm comm,
                       C_Stream stream) {
  void *real_send_buf = send_buf;
  void *real_recv_buf = recv_buf;
  if (UseScatterMemory()) {
    real_send_buf = static_cast<GcuMemory *>(send_buf)->mem_ptr;
    real_recv_buf = static_cast<GcuMemory *>(recv_buf)->mem_ptr;
  }
  ECCL_CHECK(ecclAllGather(real_send_buf,
                           real_recv_buf,
                           count,
                           ConvertEcclDataType(data_type),
                           reinterpret_cast<ecclComm_t>(comm),
                           reinterpret_cast<topsStream_t>(stream)));
  VLOG(6) << "Backend GCU XcclAllGather successfully, data_type:"
          << DataTypeToStr(data_type) << ", count:" << count;
  return C_SUCCESS;
}

C_Status XcclReduceScatter(void *send_buf,
                           void *recv_buf,
                           size_t count,
                           C_DataType data_type,
                           C_CCLReduceOp op,
                           C_CCLComm comm,
                           C_Stream stream) {
  void *real_send_buf = send_buf;
  void *real_recv_buf = recv_buf;
  if (UseScatterMemory()) {
    real_send_buf = static_cast<GcuMemory *>(send_buf)->mem_ptr;
    real_recv_buf = static_cast<GcuMemory *>(recv_buf)->mem_ptr;
  }
  ECCL_CHECK(ecclReduceScatter(real_send_buf,
                               real_recv_buf,
                               count,
                               ConvertEcclDataType(data_type),
                               ConvertEcclReduceOp(op),
                               reinterpret_cast<ecclComm_t>(comm),
                               reinterpret_cast<topsStream_t>(stream)));
  return C_SUCCESS;
}

C_Status XcclGroupStart() {
  ECCL_CHECK(ecclGroupStart());
  return C_SUCCESS;
}

C_Status XcclGroupEnd() {
  ECCL_CHECK(ecclGroupEnd());
  return C_SUCCESS;
}

C_Status XcclSend(void *send_buf,
                  size_t count,
                  C_DataType data_type,
                  size_t dest_rank,
                  C_CCLComm comm,
                  C_Stream stream) {
  void *real_send_buf = send_buf;
  if (UseScatterMemory()) {
    real_send_buf = static_cast<GcuMemory *>(send_buf)->mem_ptr;
  }
  ECCL_CHECK(ecclSend(real_send_buf,
                      count,
                      ConvertEcclDataType(data_type),
                      static_cast<int>(dest_rank),
                      reinterpret_cast<ecclComm_t>(comm),
                      reinterpret_cast<topsStream_t>(stream)));
  return C_SUCCESS;
}

C_Status XcclRecv(void *recv_buf,
                  size_t count,
                  C_DataType data_type,
                  size_t src_rank,
                  C_CCLComm comm,
                  C_Stream stream) {
  void *real_recv_buf = recv_buf;
  if (UseScatterMemory()) {
    real_recv_buf = static_cast<GcuMemory *>(recv_buf)->mem_ptr;
  }
  ECCL_CHECK(ecclRecv(real_recv_buf,
                      count,
                      ConvertEcclDataType(data_type),
                      static_cast<int>(src_rank),
                      reinterpret_cast<ecclComm_t>(comm),
                      reinterpret_cast<topsStream_t>(stream)));
  return C_SUCCESS;
}

C_Status ScatterMemorySetDims(const C_Device device,
                              void *ptr,
                              int64_t *dims_data,
                              uint32_t ndims) {
  GcuDeviceGuard guard(device->id);
  if (UseScatterMemory()) {
    auto *mem_ptr = static_cast<GcuMemory *>(ptr)->mem_ptr;
    auto dims = std::vector<int64_t>(dims_data, dims_data + ndims);
    VLOG(6) << "ScatterMemorySetDims: "
            << " ptr: " << ptr << " mem_ptr: " << mem_ptr
            << VectorToString(dims);
    static_cast<GcuMemory *>(ptr)->dims = dims;
    if (dims.empty()) {
      static_cast<GcuMemory *>(ptr)->dims = {1};
      int64_t one = 1;
      RT_CHECK(topsMemorySetDims(mem_ptr, &one, one));
    } else {
      RT_CHECK(topsMemorySetDims(mem_ptr, dims_data, ndims));
    }
  }
  return C_SUCCESS;
}

void InitPlugin(CustomRuntimeParams *params) {
  PADDLE_CUSTOM_RUNTIME_CHECK_VERSION(params);
  memset(reinterpret_cast<void *>(params->interface),
         0,
         sizeof(C_DeviceInterface));

  params->device_type = const_cast<char *>(kDeviceType);
  params->sub_device_type = const_cast<char *>(kSubDeviceType);

  // Device
  params->interface->initialize = Init;
  params->interface->finalize = Finalize;
  params->interface->init_device = InitDevice;
  params->interface->set_device = SetDevice;
  params->interface->get_device = GetDevice;
  params->interface->deinit_device = DeInitDevice;

  // Stream
  params->interface->create_stream = CreateStream;
  params->interface->destroy_stream = DestroyStream;
  params->interface->stream_add_callback = AddCallback;

  // Event
  params->interface->create_event = CreateEvent;
  params->interface->record_event = RecordEvent;
  params->interface->destroy_event = DestroyEvent;

  // Synchronize
  params->interface->synchronize_device = SyncDevice;
  params->interface->synchronize_stream = SyncStream;
  params->interface->synchronize_event = SyncEvent;
  params->interface->stream_wait_event = StreamWaitEvent;

  // Memory
  params->interface->device_memory_allocate = Allocate;
  params->interface->device_memory_deallocate = Deallocate;
  params->interface->host_memory_allocate = HostAllocate;
  params->interface->host_memory_deallocate = HostDeallocate;
  params->interface->memory_copy_h2d = MemCpyH2D;
  params->interface->memory_copy_d2h = MemCpyD2H;
  params->interface->memory_copy_d2d = MemCpyD2D;
  params->interface->memory_copy_p2p = nullptr;
  params->interface->async_memory_copy_h2d = AsyncMemCpyH2D;
  params->interface->async_memory_copy_d2h = AsyncMemCpyD2H;
  params->interface->async_memory_copy_d2d = AsyncMemCpyD2D;
  params->interface->async_memory_copy_p2p = nullptr;

  // Get visible device info
  params->interface->get_device_count = GetDevicesCount;
  params->interface->get_device_list = GetDevicesList;
  params->interface->device_min_chunk_size = DeviceMinChunkSize;
  params->interface->device_max_chunk_size = DeviceMaxChunkSize;
  params->interface->device_extra_padding_size = ExtraPaddingSize;

  // Xccl
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
  VLOG(0) << "InitPlugin for backend GCU successfully.";
}

void DeAllocScatter(void *ptr) {
  if (ptr) {
    uint32_t mem_type = 0;
    RT_CHECK(topsPointerGetAttribute(
        &mem_type, TOPS_POINTER_ATTRIBUTE_MEMORY_TYPE, ptr));
    if ((mem_type & topsMemoryTypeScatter) &&
        !(mem_type & topsMemoryTypeLazy)) {
      uint64_t sub_num = 0;
      RT_CHECK(topsScatterMemoryGetSubNum(ptr, &sub_num));
      for (uint64_t idx = 0; idx < sub_num; ++idx) {
        void *sub_mem = nullptr;
        RT_CHECK(topsScatterGetSubMem(ptr, idx, &sub_mem));
        RT_CHECK(topsFree(sub_mem));
      }
      RT_CHECK(topsScatterClearSubMemory(ptr));
    }
    RT_CHECK(topsFree(ptr));
  }
  return;
}

std::string DumpHbm(void *ptr) {
  std::stringstream ss;
  if (ptr) {
    size_t ndims = 16;
    std::vector<int64_t> dims(ndims, 0);
    ss << "  ptr address: " << ptr << "\n";
    RT_CHECK(topsMemoryGetDims(
        ptr, reinterpret_cast<int64_t *>(dims.data()), &ndims));
    ss << "  scatter dims: "
       << VectorToString(
              std::vector<int64_t>(dims.begin(), dims.begin() + ndims))
       << "\n";
    ss << "    {\n";
    uint64_t sub_num = 0;
    RT_CHECK(topsScatterMemoryGetSubNum(ptr, &sub_num));
    for (uint64_t idx = 0; idx < sub_num; ++idx) {
      void *sub_mem;
      RT_CHECK(topsScatterGetSubMem(ptr, idx, &sub_mem));
      RT_CHECK(topsMemoryGetDims(
          sub_mem, reinterpret_cast<int64_t *>(dims.data()), &ndims));
      ss << "    sub " << idx << " " << sub_mem << ": "
         << VectorToString(
                std::vector<int64_t>(dims.begin(), dims.begin() + ndims))
         << "\n";
    }
    ss << "    }\n";
  } else {
    ss << "  nullptr\n";
  }
  return ss.str();
}

C_Status InitResource(const int32_t device_id) {
  VLOG(1) << "start init resource";
  thread_local topsResource_t res_bundle = nullptr;

  if (res_bundle == nullptr) {
    VLOG(1) << " thread id: " << std::this_thread::get_id();
    GcuDeviceGuard guard(device_id);
    topsResourceRequest_t req;
    topsDeviceProp_t prop;
    int block_num = 0;
    int dev = -1;
    RT_CHECK(topsGetDevice(&dev));
    RT_CHECK(topsGetDeviceProperties(&prop, dev));
    block_num = prop.multiProcessorCount;

    if (res_bundles_.count(device_id) <= 0) {
      if (block_num == 4) {  // pavo 4c
        VLOG(1) << "create resource for pavo";
        memset(&req, 0x0, sizeof(req));
        req.cluster_count = block_num;
        req.need_alloc_cluster = true;
        topsCreateResource(&res_bundle, req);
      } else if (block_num == 2) {  // dorado 2c
        VLOG(1) << "create resource for dorado";
        memset(&req, 0x0, sizeof(req));
        topsResourceBundle_t *resource =
            reinterpret_cast<topsResourceBundle_t *>(&req.res_bundle_cfg[0]);

        req.cluster_count = 2;
        req.need_alloc_cluster = true;
        req.compute_res_claim = 3 * req.cluster_count;
        resource->mode = TOPS_RES_BUNDLE_MODE__VG;
        resource->hbm_mem_size = 0;
        resource->hbm_policy = TOPS_HBM_POLICY__BEST_FIT;
        resource->hcve_num = 1;
        resource->u.vg.sip_num = 12 * req.cluster_count;
        resource->u.vg.cdma_num = 4 * req.cluster_count;
        resource->u.vg.cdma_vc_num = 32;

        RT_CHECK(topsCreateResource(&res_bundle, req));
      }
      res_bundles_[device_id] = res_bundle;
    } else {
      res_bundle = res_bundles_.at(device_id);
    }

    if (block_num == 2) {
      PADDLE_ENFORCE_EQ(res_bundle,
                        res_bundles_.at(device_id),
                        phi::errors::InvalidArgument(
                            "can only create one resource in a thread "));

      VLOG(1) << "set resource for dorado";
      RT_CHECK(topsDeviceSetResource(res_bundle));
    }
    VLOG(1) << "start dispatch init";
    hlir::HlirDispatch::dispatchInit();
  }

  VLOG(1) << "success init resource";

  return C_SUCCESS;
}

void FinalizeResource() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto iter : res_bundles_) {
    RT_CHECK(topsDestroyResource(iter.second));
  }

  hlir::HlirDispatch::dispatchDeInit();
}
