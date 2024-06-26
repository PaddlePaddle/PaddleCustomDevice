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

#include <gflags/gflags.h>

#include <algorithm>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/flags.h"
#include "runtime/flags.h"
#include "runtime/sdaaEvent.h"
#include "version/query.h"

#define MEMORY_FRACTION 0.5f
#define PINNED_MEMORY_BLOCKSIZE (64 * 1024)
#define PINNED_MEMORY_BLOCKNUM 64
#define MIN_CHUNK_SIZE 0x2000

FLAGS_DEFINE_bool(sdaa_reuse_event, true, "enable event reuse.");
FLAGS_DEFINE_bool(sdaa_runtime_debug, false, "runtime debug log");
FLAGS_DEFINE_bool(sdaa_error_check, false, "enable error check for runtime");

PHI_DECLARE_double(fraction_of_gpu_memory_to_use);
PHI_DECLARE_uint64(initial_gpu_memory_in_mb);
PHI_DECLARE_uint64(reallocate_gpu_memory_in_mb);
constexpr static float fraction_reserve_sdaa_memory = 0.05f;

std::mutex g_mutex;

namespace phi {
std::unordered_map<uint64_t, ThreadId> GetAllThreadIds();
}  // namespace phi

thread_local int g_current_device_id(-1);

// NOTE: do not create instance of EventPool manually.
class EventPool {
 public:
  using Event = std::unique_ptr<sdaaEventWrapper,
                                std::function<void(sdaaEventWrapper *)>>;
  EventPool() : pools_(get_devices_count()) {}

  Event get(int device_id) {
    PADDLE_ENFORCE_GE(device_id,
                      0,
                      phi::errors::InvalidArgument(
                          "device_id should be greater than or equal to 0."));
    PADDLE_ENFORCE_LT(device_id,
                      pools_.size(),
                      phi::errors::InvalidArgument(
                          "device_id should be less than device count."));

    auto &pool = pools_[device_id];
    auto destructor = [&pool](sdaaEventWrapper *event) {
      std::lock_guard<std::mutex> lock(pool.mutex_);
      VLOG_IF(0, FLAGS_sdaa_runtime_debug)
          << "[Before]event pool size is: " << pool.event_pool_.size();
      pool.event_pool_.push_back(std::unique_ptr<sdaaEventWrapper>(event));
      VLOG_IF(0, FLAGS_sdaa_runtime_debug)
          << "[After]event pool size is: " << pool.event_pool_.size();
    };

    // Try to acquire an event from the per-device pool.
    {
      std::lock_guard<std::mutex> lock(pool.mutex_);
      if (!pool.event_pool_.empty()) {
        VLOG_IF(0, FLAGS_sdaa_runtime_debug)
            << "[EventPool::get]get event from event pool. pool id is: "
            << device_id << ", size is: " << pool.event_pool_.size();

        auto *event = pool.event_pool_.front().release();
        pool.event_pool_.pop_front();
        VLOG_IF(0, FLAGS_sdaa_runtime_debug)
            << "[EventPool::get]get event from event pool. pool id is: "
            << device_id << ", after size is: " << pool.event_pool_.size();
        return Event(event, destructor);
      }
    }
    // otherwise, allocate a new event that will be returned to the pool on
    // destruction.
    VLOG_IF(0, FLAGS_sdaa_runtime_debug) << "[EventPool::get]create new event.";
    return Event(std::make_unique<sdaaEventWrapper>(device_id).release(),
                 destructor);
  }

  void empty_cache(int dev_id) {
    std::lock_guard<std::mutex> lock(pools_[dev_id].mutex_);
    pools_[dev_id].event_pool_.clear();
  }

 private:
  struct PerDevicePool {
    alignas(64) std::mutex mutex_;
    std::deque<std::unique_ptr<sdaaEventWrapper>> event_pool_{};
  };
  std::vector<PerDevicePool> pools_{};
} g_event_pool;

static std::vector<std::vector<EventPool::Event>> hold_event_vecs(
    get_devices_count());

class AlignnedAllocator {
 public:
  void *pinned_mem_block[PINNED_MEMORY_BLOCKNUM] = {};
  sdaaEvent_t event[PINNED_MEMORY_BLOCKNUM] = {};
  int counter = 0;

  void Alloc(size_t size, size_t align) {
    std::lock_guard<std::mutex> lock(mtx_);
    void *p[PINNED_MEMORY_BLOCKNUM];
    for (int i = 0; i < PINNED_MEMORY_BLOCKNUM; i++) {
      checkSdaaErrors(sdaaMallocHost(&p[i], size + align));
      void *ret = reinterpret_cast<void *>(
          reinterpret_cast<size_t>(p[i]) + align -
          (reinterpret_cast<size_t>(p[i]) & (align - 1)));
      recorded_events_[ret] = {p[i], nullptr};
      pinned_mem_block[i] = ret;
    }
  }

  void Record() {
    std::lock_guard<std::mutex> lock(mtx_);
    int idx = counter % PINNED_MEMORY_BLOCKNUM;
    recorded_events_[pinned_mem_block[idx]].second = event[idx];
    counter++;
  }

  void ClearEvent() {
    std::lock_guard<std::mutex> lock(mtx_);
    for (auto it = recorded_events_.begin(); it != recorded_events_.end();) {
      sdaaEvent_t event_tmp = it->second.second;
      if (!event_tmp) {
        it++;
        continue;
      }
      checkSdaaErrors(sdaaEventSynchronize(event_tmp));
      void *ptr = it->second.first;
      checkSdaaErrors(sdaaFreeHost(ptr));
      checkSdaaErrors(sdaaEventDestroy(event_tmp));
      it = recorded_events_.erase(it);
    }
  }

 private:
  std::unordered_map<void *, std::pair<void *, sdaaEvent_t>> recorded_events_;
  std::mutex mtx_;
};

class AlignnedAllocatorList {
 public:
  explicit AlignnedAllocatorList(size_t device_count)
      : allocator_list(device_count, nullptr) {}

  void Init(size_t dev_id) {
    allocator_list[dev_id] = new AlignnedAllocator;
    allocator_list[dev_id]->Alloc(PINNED_MEMORY_BLOCKSIZE, 64);
    for (int i = 0; i < PINNED_MEMORY_BLOCKNUM; i++) {
      checkSdaaErrors(sdaaEventCreate(&(allocator_list[dev_id]->event[i])));
    }
  }

  bool Inited(size_t dev_id) { return allocator_list[dev_id] != nullptr; }

  void Deinit(size_t dev_id) {
    delete allocator_list[dev_id];
    allocator_list[dev_id] = nullptr;
  }

  AlignnedAllocator *GetAllocator(size_t dev_id) {
    return allocator_list[dev_id];
  }

  void *GetPinnedMemoryPtr(size_t dev_id) {
    int idx = allocator_list[dev_id]->counter % PINNED_MEMORY_BLOCKNUM;
    while (sdaaEventQuery(allocator_list[dev_id]->event[idx]) != sdaaSuccess) {
    }
    return allocator_list[dev_id]->pinned_mem_block[idx];
  }

  sdaaEvent_t GetEvent(size_t dev_id) {
    int idx = allocator_list[dev_id]->counter % PINNED_MEMORY_BLOCKNUM;
    return allocator_list[dev_id]->event[idx];
  }

  void Record(size_t dev_id) { allocator_list[dev_id]->Record(); }

 private:
  std::vector<AlignnedAllocator *> allocator_list;
};

static AlignnedAllocatorList *global_allocator_list = nullptr;

bool isEnvEnable(std::string env_) {
  static std::unordered_map<std::string, bool> envMap;
  if (!envMap.count(env_)) {
    const char *ret = std::getenv(env_.c_str());
    if (ret) {
      envMap[env_] = atoi(ret);
    }
  }
  return envMap[env_];
}

// -1 when env is not set / 0 when env is 0 or not number
int getEnvVal(std::string env_) {
  static std::unordered_map<std::string, int> valMap;
  if (!valMap.count(env_)) {
    const char *ret = std::getenv(env_.c_str());
    if (ret) {
      valMap[env_] = atoi(ret);
    } else {
      valMap[env_] = -1;
    }
  }
  return valMap[env_];
}

inline void check_uninitialized_thread(int device_id) {
  if (g_current_device_id == -1) {
    g_current_device_id = device_id;
    checkSdaaErrors(sdaaSetDevice(device_id));
  }
}

inline size_t get_current_device_id() {
  check_uninitialized_thread(0);
  checkSdaaErrors(sdaaGetDevice(&g_current_device_id));
  return g_current_device_id;
}

C_Status Init() {
  std::cout << "sdaa plugin compiled with ";
#ifdef __clang__
  std::cout << "clang\n";
#else
  std::cout << "gcc\n";
#endif
  auto paddle_commit = GetPaddlePaddleCommit().version;
  auto paddle_version = GetPaddlePaddleVersion().version;
  if (!paddle_commit.empty() && !paddle_version.empty()) {
    std::cout << "PaddlePaddle Compilation Commit: " << paddle_commit
              << std::endl;
    std::cout << "PaddlePaddle Compilation Version: " << paddle_version
              << std::endl;
  }
  auto version_table = GetVersionTable();
  std::cout << version_table << std::endl;

  PrintExtraInfo();

  size_t count = get_devices_count();
  if (count) {
    global_allocator_list = new AlignnedAllocatorList(count);
  }

  auto version_check = CheckVersions();

  switch (version_check) {
    case VersionCheckType::COMPATIBLE:
      LOG(WARNING) << "WARNING: The installed paddle-sdaa is compatible with "
                      "underlying software stack, but is not consistent. "
                      "Advise to reinstall paddle-sdaa or reload dynamic "
                      "libraries with consistent versions.";
      break;

    case VersionCheckType::INCOMPATIBLE:
      LOG(WARNING) << "WARNING: The installed paddle-sdaa is incompatible with "
                      "underlying software "
                      "stack, which will cause serious incompatible bug. "
                      "Please reinstall "
                      "paddle-sdaa or reload dynamic libraries with consistent "
                      "versions.";
      break;

    default:
      break;
  }

  return C_SUCCESS;
}

C_Status InitDevice(const C_Device device) {
  // checkSdaaErrors(sdaaSetDevice(device->id));
  return C_SUCCESS;
}

C_Status SetDevice(const C_Device device) {
  if (g_current_device_id != device->id) {
    checkSdaaErrors(sdaaSetDevice(device->id));
    g_current_device_id = device->id;
  }
  if (global_allocator_list && !global_allocator_list->Inited(device->id)) {
    global_allocator_list->Init(device->id);
  }
  return C_SUCCESS;
}

C_Status GetDevice(const C_Device device) {
  device->id = get_current_device_id();
  return C_SUCCESS;
}

C_Status DestroyDevice(const C_Device device) {
  // checkSdaaErrors(sdaaSetDevice(device->id));
  if (global_allocator_list) {
    if (!global_allocator_list->Inited(device->id)) {
      return C_SUCCESS;
    }
    checkSdaaErrors(sdaaSetDevice(device->id));
    global_allocator_list->GetAllocator(device->id)->ClearEvent();
    global_allocator_list->Deinit(device->id);
  }
  hold_event_vecs[device->id].clear();
  g_event_pool.empty_cache(device->id);
  checkSdaaErrors(sdaaDeviceReset());
  return C_SUCCESS;
}

C_Status Finalize() {
  for (auto iter = hold_event_vecs.begin(); iter != hold_event_vecs.end();
       ++iter) {
    iter->clear();
  }
  hold_event_vecs.clear();
  size_t device_count = get_devices_count();
  for (size_t i = 0; i < device_count; ++i) {
    g_event_pool.empty_cache(i);
  }
  return C_SUCCESS;
}

C_Status GetDevicesCount(size_t *count) {
  int device_cnt = 1;
  checkSdaaErrors(sdaaGetDeviceCount(&device_cnt));
  *count = device_cnt;
  return C_SUCCESS;
}

C_Status GetDevicesList(size_t *devices) {
  int count = 1;
  checkSdaaErrors(sdaaGetDeviceCount(&count));
  for (int dev_id = 0; dev_id < count; dev_id++) {
    devices[dev_id] = dev_id;
  }
  return C_SUCCESS;
}

C_Status MemCpyH2D(const C_Device device,
                   void *dst,
                   const void *src,
                   size_t size) {
  checkSdaaErrors(sdaaMemcpy(dst, src, size, sdaaMemcpyHostToDevice));

  return C_SUCCESS;
}

C_Status MemCpyD2H(const C_Device device,
                   void *dst,
                   const void *src,
                   size_t size) {
  checkSdaaErrors(sdaaMemcpy(dst, src, size, sdaaMemcpyDeviceToHost));
  return C_SUCCESS;
}

C_Status MemCpyD2D(const C_Device device,
                   void *dst,
                   const void *src,
                   size_t size) {
  checkSdaaErrors(sdaaMemcpy(dst, src, size, sdaaMemcpyDeviceToDevice));
  return C_SUCCESS;
}

C_Status AsyncMemCpyH2D(const C_Device device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size) {
  CustomSDAAStream_t sdaa_stream = reinterpret_cast<CustomSDAAStream_t>(stream);
  sdaaStream_t p_sdaa_stream = sdaa_stream->pStream;
  // CustomSDAAStream_t p_sdaa_stream = GetSecondaryStream(sdaa_stream);
  if (PINNED_MEMORY_BLOCKSIZE < size) {
    checkSdaaErrors(
        sdaaMemcpyAsync(dst, src, size, sdaaMemcpyHostToDevice, p_sdaa_stream));
  } else {
    auto device_id = get_current_device_id();
    void *tmp = global_allocator_list->GetPinnedMemoryPtr(device_id);
    memcpy(tmp, src, size);
    checkSdaaErrors(
        sdaaMemcpyAsync(dst, tmp, size, sdaaMemcpyHostToDevice, p_sdaa_stream));
    checkSdaaErrors(sdaaEventRecord(global_allocator_list->GetEvent(device_id),
                                    p_sdaa_stream));
    global_allocator_list->Record(device_id);
  }
  return C_SUCCESS;
}

C_Status AsyncMemCpyD2H(const C_Device device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size) {
  CustomSDAAStream_t sdaa_stream = reinterpret_cast<CustomSDAAStream_t>(stream);
  sdaaStream_t p_sdaa_stream = sdaa_stream->pStream;
  checkSdaaErrors(
      sdaaMemcpyAsync(dst, src, size, sdaaMemcpyDeviceToHost, p_sdaa_stream));
  return C_SUCCESS;
}

C_Status AsyncMemCpyD2D(const C_Device device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size) {
  CustomSDAAStream_t sdaa_stream = reinterpret_cast<CustomSDAAStream_t>(stream);
  sdaaStream_t p_sdaa_stream = sdaa_stream->pStream;
  checkSdaaErrors(
      sdaaMemcpyAsync(dst, src, size, sdaaMemcpyDeviceToDevice, p_sdaa_stream));
  return C_SUCCESS;
}

C_Status DeviceAllocate(const C_Device device, void **ptr, size_t size) {
  void *data;
  checkSdaaErrors(sdaaMalloc(&data, size));
  if (data) {
    *ptr = data;
    return C_SUCCESS;
  } else {
    *ptr = nullptr;
  }
  return C_FAILED;
}

// NOTE(liaotianju):pinned-memory is not invoked by CustomDevice for now,
// uncomment when it is
//
// C_Status HostAllocate(const C_Device device, void **ptr, size_t size) {
//   void *data;
//   checkSdaaErrors(sdaaMallocHost(&data, size));
//   if (data) {
//     *ptr = data;
//     return C_SUCCESS;
//   } else {
//     *ptr = nullptr;
//   }
//   return C_FAILED;
// }

C_Status DeviceDeallocate(const C_Device device, void *ptr, size_t size) {
  checkSdaaErrors(sdaaFree(ptr));
  return C_SUCCESS;
}

// NOTE(liaotianju):pinned-memory is not invoked by CustomDevice for now,
// uncomment when it is
//
// C_Status HostDeallocate(const C_Device device, void *ptr, size_t size) {
//   checkSdaaErrors(sdaaFreeHost(ptr));
//   return C_SUCCESS;
// }

C_Status CreateStream(const C_Device device, C_Stream *stream) {
  VLOG(4) << "start to create stream.";
  tecodnnHandle_t tecodnnHandle;
  TECODNN_CHECK(tecodnnCreate(&tecodnnHandle));
  tblasHandle_t tecoblasHandle;
  TBLAS_CHECK(tblasCreate(&tecoblasHandle));
  sdaaStream_t stream_;
  checkSdaaErrors(sdaaStreamCreateWithFlags(&stream_, sdaaStreamNonBlocking));
  TECODNN_CHECK(tecodnnSetStream(tecodnnHandle, stream_));
  TBLAS_CHECK(tecoblasSetStream(tecoblasHandle, stream_));
  CustomSDAAStream_t sdaa_stream = new CustomSDAAStream();
  sdaa_stream->dnnHandle = tecodnnHandle;
  sdaa_stream->tblasHandle = tecoblasHandle;
  sdaa_stream->pStream = stream_;
  *stream = reinterpret_cast<C_Stream>(sdaa_stream);
  return C_SUCCESS;
}

C_Status DestroyStream(const C_Device device, C_Stream stream) {
  VLOG(4) << "start to destory stream.";
  if (stream) {
    CustomSDAAStream_t sdaa_stream =
        reinterpret_cast<CustomSDAAStream_t>(stream);
    delete sdaa_stream;
  }
  return C_SUCCESS;
}

C_Status QueryStream(const C_Device device, C_Stream stream) {
  CustomSDAAStream_t sdaa_stream = reinterpret_cast<CustomSDAAStream_t>(stream);
  sdaaStream_t p_sdaa_stream = sdaa_stream->pStream;
  auto status = sdaaStreamQuery(p_sdaa_stream);
  return status == sdaaSuccess ? C_SUCCESS : C_FAILED;
}

C_Status CreateEvent(const C_Device device, C_Event *event) {
  SetDevice(device);
  if (FLAGS_sdaa_reuse_event) {
    std::lock_guard<std::mutex> g_lock(g_mutex);
    hold_event_vecs[device->id].push_back(
        std::move(g_event_pool.get(device->id)));
    *event =
        reinterpret_cast<C_Event>(hold_event_vecs[device->id].back()->event());
    VLOG_IF(0, FLAGS_sdaa_runtime_debug)
        << "[RUNTIME] CreateEvent: device=" << device->id
        << ", event=" << *event
        << ". Push_back hold_vec size: " << hold_event_vecs[device->id].size();
  } else {
    checkSdaaErrors(sdaaEventCreate(reinterpret_cast<sdaaEvent_t *>(event)));
    VLOG_IF(0, FLAGS_sdaa_runtime_debug)
        << "[RUNTIME] CreateEvent: device=" << device->id
        << ", event=" << *event;
  }
  return C_SUCCESS;
}

C_Status RecordEvent(const C_Device device, C_Stream stream, C_Event event) {
  SetDevice(device);
  VLOG_IF(0, FLAGS_sdaa_runtime_debug)
      << "[RUNTIME] RecordEvent: device=" << device->id << ", stream=" << stream
      << ", event=" << event;
  sdaaEvent_t sd_event = reinterpret_cast<sdaaEvent_t>(event);
  CustomSDAAStream_t sdaa_stream = reinterpret_cast<CustomSDAAStream_t>(stream);
  sdaaStream_t p_sdaa_stream = sdaa_stream->pStream;
  if (FLAGS_sdaa_reuse_event) {
    std::lock_guard<std::mutex> g_lock(g_mutex);
    for (auto iter = hold_event_vecs[device->id].begin();
         iter != hold_event_vecs[device->id].end();
         ++iter) {
      if ((*iter)->isCreated() && (*iter)->event() == sd_event) {
        (*iter)->record(device->id, p_sdaa_stream);
        return C_SUCCESS;
      }
    }
    LOG(ERROR) << "[RecordEvent]: can not find event " << event
               << " in hold_event_vecs!";
  } else {
    checkSdaaErrors(sdaaEventRecord(sd_event, p_sdaa_stream));
    return C_SUCCESS;
  }
  return C_FAILED;
}

C_Status QueryEvent(const C_Device device, C_Event event) {
  SetDevice(device);
  VLOG_IF(0, FLAGS_sdaa_runtime_debug)
      << "[RUNTIME] QueryEvent: device=" << device->id << ", event=" << event;
  sdaaEvent_t sd_event = reinterpret_cast<sdaaEvent_t>(event);
  if (FLAGS_sdaa_reuse_event) {
    std::lock_guard<std::mutex> g_lock(g_mutex);
    bool query_status = false;
    for (auto iter = hold_event_vecs[device->id].begin();
         iter != hold_event_vecs[device->id].end();
         ++iter) {
      if ((*iter)->isCreated() && (*iter)->wasRecorded() &&
          (*iter)->event() == sd_event) {
        query_status = (*iter)->query();
      }
    }
    return query_status ? C_SUCCESS : C_FAILED;
  } else {
    auto status = sdaaEventQuery(sd_event);
    return status == sdaaSuccess ? C_SUCCESS : C_FAILED;
  }
}

C_Status DestroyEvent(const C_Device device, C_Event event) {
  SetDevice(device);
  VLOG_IF(0, FLAGS_sdaa_runtime_debug)
      << "[RUNTIME] DestroyEvent: device=" << device->id << ", event=" << event;
  sdaaEvent_t sd_event = reinterpret_cast<sdaaEvent_t>(event);
  if (FLAGS_sdaa_reuse_event) {
    std::lock_guard<std::mutex> g_lock(g_mutex);
    for (auto iter = hold_event_vecs[device->id].begin();
         iter != hold_event_vecs[device->id].end();) {
      if ((*iter)->isCreated() && (*iter)->event() == sd_event) {
        iter = hold_event_vecs[device->id].erase(iter);
        return C_SUCCESS;
      } else {
        ++iter;
      }
    }
    LOG(ERROR) << "[DestroyEvent]: can not find event " << event
               << " in hold_event_vecs!";
  } else {
    checkSdaaErrors(sdaaEventDestroy(sd_event));
    return C_SUCCESS;
  }
  return C_FAILED;
}

C_Status SyncDevice(const C_Device device) {
  checkSdaaErrors(sdaaDeviceSynchronize());
  return C_SUCCESS;
}

C_Status SyncStream(const C_Device device, C_Stream stream) {
  CustomSDAAStream_t sdaa_stream = reinterpret_cast<CustomSDAAStream_t>(stream);
  sdaaStream_t p_sdaa_stream = sdaa_stream->pStream;
  checkSdaaErrors(sdaaStreamSynchronize(p_sdaa_stream));
  return C_SUCCESS;
}

static void CallBackWarpper(void *user_data) {
  std::unique_ptr<std::function<void()>> func(
      reinterpret_cast<std::function<void()> *>(user_data));
  (*func)();
}

C_Status AddCallback(const C_Device device,
                     C_Stream stream,
                     C_Callback callback,
                     void *user_data) {
  CustomSDAAStream_t sdaa_stream = reinterpret_cast<CustomSDAAStream_t>(stream);
  sdaaStream_t p_sdaa_stream = sdaa_stream->pStream;
  checkSdaaErrors(
      sdaaLaunchHostFunc(p_sdaa_stream, CallBackWarpper, user_data));
  return C_SUCCESS;
}

C_Status SyncEvent(const C_Device device, C_Event event) {
  SetDevice(device);
  VLOG_IF(0, FLAGS_sdaa_runtime_debug)
      << "[RUNTIME] SyncEvent: device=" << device->id << ", event=" << event;
  sdaaEvent_t sd_event = reinterpret_cast<sdaaEvent_t>(event);
  if (FLAGS_sdaa_reuse_event) {
    std::lock_guard<std::mutex> g_lock(g_mutex);
    for (auto iter = hold_event_vecs[device->id].begin();
         iter != hold_event_vecs[device->id].end();
         ++iter) {
      if ((*iter)->isCreated() && (*iter)->wasRecorded() &&
          (*iter)->event() == sd_event) {
        (*iter)->synchronize();
        return C_SUCCESS;
      }
    }
    LOG(ERROR) << "[SyncEvent]: can not find event " << event
               << " in hold_event_vecs!";
  } else {
    checkSdaaErrors(sdaaEventSynchronize(sd_event));
    return C_SUCCESS;
  }
  return C_FAILED;
}

C_Status StreamWaitEvent(const C_Device device,
                         C_Stream stream,
                         C_Event event) {
  SetDevice(device);
  VLOG_IF(0, FLAGS_sdaa_runtime_debug)
      << "[RUNTIME] StreamWaitEvent: device=" << device->id
      << ", stream=" << stream << ", event=" << event;
  sdaaEvent_t sd_event = reinterpret_cast<sdaaEvent_t>(event);
  CustomSDAAStream_t sdaa_stream = reinterpret_cast<CustomSDAAStream_t>(stream);
  sdaaStream_t p_sdaa_stream = sdaa_stream->pStream;
  if (FLAGS_sdaa_reuse_event) {
    std::lock_guard<std::mutex> g_lock(g_mutex);
    for (auto iter = hold_event_vecs[device->id].begin();
         iter != hold_event_vecs[device->id].end();
         ++iter) {
      if ((*iter)->isCreated() && (*iter)->wasRecorded() &&
          (*iter)->event() == sd_event) {
        (*iter)->block(p_sdaa_stream);
        return C_SUCCESS;
      }
    }
    LOG(ERROR) << "[StreamWaitEvent]: can not find event " << event
               << " in hold_event_vecs!";
  } else {
    checkSdaaErrors(sdaaStreamWaitEvent(p_sdaa_stream, sd_event, 0));
    return C_SUCCESS;
  }
  return C_FAILED;
}

C_Status VisibleDevices(size_t *devices) { return C_SUCCESS; }

C_Status DeviceMemStats(const C_Device device,
                        size_t *total_memory,
                        size_t *free_memory) {
  checkSdaaErrors(sdaaMemGetInfo(free_memory, total_memory));
  return C_SUCCESS;
}

C_Status DeviceMemSet(const C_Device device,
                      void *ptr,
                      unsigned char value,
                      size_t size) {
  VLOG(4) << "call sdaa memset with size: " << size << " value: " << value;
  checkSdaaErrors(sdaaMemset(ptr, value, size));
  return C_SUCCESS;
}

C_Status DeviceMinChunkSize(const C_Device device, size_t *size) {
  *size = MIN_CHUNK_SIZE;
  return C_SUCCESS;
}

size_t sdaaAvailableMemToAlloc() {
  size_t total = 0;
  size_t available = 0;
  checkSdaaErrors(sdaaMemGetInfo(&available, &total));
  size_t reserving =
      static_cast<size_t>(fraction_reserve_sdaa_memory * available);
  // If available size is less than minimum chunk size, no usable memory exists
  size_t available_to_alloc = available - reserving;
  size_t min_chunk_size = MIN_CHUNK_SIZE;
  if (available_to_alloc < min_chunk_size) {
    available_to_alloc = 0;
  }
  VLOG(10) << "SDAA usage " << (available >> 20) << "M/" << (total >> 20)
           << "M, " << (available_to_alloc >> 20) << "M available to allocate";
  return available_to_alloc;
}

static size_t sdaaAllocSize(bool realloc) {
  size_t available_to_alloc = sdaaAvailableMemToAlloc();
  PADDLE_ENFORCE_GT(
      available_to_alloc,
      0,
      phi::errors::ResourceExhausted("Not enough available SDAA memory."));
  // If FLAGS_initial_gpu_memory_in_mb is 0, then initial memory will be
  // allocated by fraction
  size_t flag_mb = realloc ? FLAGS_reallocate_gpu_memory_in_mb
                           : FLAGS_initial_gpu_memory_in_mb;
  size_t alloc_bytes =
      (flag_mb > 0ul
           ? flag_mb << 20
           : available_to_alloc * FLAGS_fraction_of_gpu_memory_to_use);
  PADDLE_ENFORCE_GE(
      available_to_alloc,
      alloc_bytes,
      phi::errors::ResourceExhausted("Not enough available SDAA memory."));
  VLOG(10) << "Alloc size is " << (alloc_bytes >> 20)
           << " MiB, is it Re-alloc: " << realloc;
  return alloc_bytes;
}

C_Status DeviceMaxChunkSize(const C_Device device, size_t *size) {
  size_t max_chunk_size = std::max(sdaaAllocSize(/* realloc = */ false),
                                   sdaaAllocSize(/* realloc = */ true));
  VLOG(10) << "Max chunk size " << (max_chunk_size >> 20) << "M";
  *size = max_chunk_size;
  return C_SUCCESS;
}

void SdptiBufferRequestedCallback(uint8_t **buffer,
                                  size_t *size,
                                  size_t *max_num_records) {
  RecordEvent::Instance().AllocateSdptiBuffer(buffer, size);
  *max_num_records = 0;
}

void BufferCompletedCallback(uint8_t *buffer, size_t size, size_t validSize) {
  SDptiResult status;
  SDpti_Activity *record = NULL;
  auto mapping = CreateThreadIdMapping();
  if (validSize > 0) {
    while (true) {
      status = custom_dynload::sdptiActivityGetNextRecord(
          buffer, validSize, &record);
      if (status == SDPTI_SUCCESS) {
        RecordEvent::Instance().ProcessActivityRecord(record, mapping);
      } else if (status == SDPTI_ERROR_MAX_LIMIT_REACHED) {
        break;
      }
    }
  }
  RecordEvent::Instance().ReleaseSdptiBuffer(buffer);
}

bool isSdptiEnabled() { return getEnvVal("ENABLE_SDPTI") != 0; }

C_Status ProfilerCollectData(C_Profiler prof,
                             uint64_t tracing_start_ns_,
                             void *user_data) {
  VLOG(4) << "Start to collect data for sdaa profiler!";
  if (isSdptiEnabled()) {
    // request from SDPTI to sync before flush
    checkSdaaErrors(sdaaDeviceSynchronize());
    CHECK_SDPTI(custom_dynload::sdptiActivityFlushAll(0));
    CHECK_SDPTI(custom_dynload::sdptiFinalize());
  }
  RecordEvent::Instance().BufferCompleted();
  return C_SUCCESS;
}

C_Status ProfilerInitialize(C_Profiler prof, void **user_data) {
  RecordEvent::Instance().SetSdptiMode(isSdptiEnabled());
  if (isEnvEnable("USE_ATTRIBUTE_INFO_DUMP")) {
    RecordEvent::Instance().AttributeDumpEnable();
  }
  return C_SUCCESS;
}

C_Status ProfilerFinalize(C_Profiler prof, void *user_data) {
  return C_SUCCESS;
}

static inline uint32_t GetTid() { return syscall(__NR_gettid); }
static inline uint32_t GetPid() { return syscall(__NR_getpid); }

C_Status ProfilerPrepare(C_Profiler prof, void *user_data) {
  VLOG(4) << "ProfilerPrepare sdaa profiler!";
  // SDPTI init
  if (isSdptiEnabled()) {
    uint64_t count = 1;
    uint64_t id = get_current_device_id();
    uint64_t value_size{0};
    uint64_t buffer_size = RecordEvent::Instance().getBufferSize();
    CHECK_SDPTI(custom_dynload::sdptiActivitySetAttribute(
        SDPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &value_size, &buffer_size));
    CHECK_SDPTI(custom_dynload::sdptiActivitySetAttribute(
        SDPTI_ACTIVITY_ATTR_DEVICE_ENABLED_ID, &count, &id));
    CHECK_SDPTI(custom_dynload::sdptiInit());
    CHECK_SDPTI(custom_dynload::sdptiActivityRegisterCallbacks(
        SdptiBufferRequestedCallback, BufferCompletedCallback));
  }
  // old profiler init
  RecordEvent::Instance().BufferRequested();
  return C_SUCCESS;
}

C_Status ProfilerStart(C_Profiler prof, void *user_data) {
  uint64_t start_ns = PosixInNsec();
  RecordEvent::Instance().SetProfPara(prof, start_ns);
  if (isSdptiEnabled()) {
    CHECK_SDPTI(
        custom_dynload::sdptiActivityEnable(SDPTI_ACTIVITY_KIND_RUNTIME));
    CHECK_SDPTI(
        custom_dynload::sdptiActivityEnable(SDPTI_ACTIVITY_KIND_KERNEL));
    CHECK_SDPTI(
        custom_dynload::sdptiActivityEnable(SDPTI_ACTIVITY_KIND_MEMCPY));
    CHECK_SDPTI(
        custom_dynload::sdptiActivityEnable(SDPTI_ACTIVITY_KIND_MEMSET));
    CHECK_SDPTI(
        custom_dynload::sdptiActivityEnable(SDPTI_ACTIVITY_KIND_MEMCPY_P2P));
    CHECK_SDPTI(
        custom_dynload::sdptiActivityEnable(SDPTI_ACTIVITY_KIND_MEMOPS));
  }
  RecordEvent::Instance().ActivityEnable();
  return C_SUCCESS;
}

C_Status ProfilerStop(C_Profiler prof, void *user_data) {
  VLOG(4) << "ProfilerStop sdaa profiler!";
  // SDPTI stop to record
  if (isSdptiEnabled()) {
    CHECK_SDPTI(
        custom_dynload::sdptiActivityDisable(SDPTI_ACTIVITY_KIND_RUNTIME));
    CHECK_SDPTI(
        custom_dynload::sdptiActivityDisable(SDPTI_ACTIVITY_KIND_KERNEL));
    CHECK_SDPTI(
        custom_dynload::sdptiActivityDisable(SDPTI_ACTIVITY_KIND_MEMCPY));
    CHECK_SDPTI(
        custom_dynload::sdptiActivityDisable(SDPTI_ACTIVITY_KIND_MEMSET));
    CHECK_SDPTI(
        custom_dynload::sdptiActivityDisable(SDPTI_ACTIVITY_KIND_MEMCPY_P2P));
    CHECK_SDPTI(
        custom_dynload::sdptiActivityDisable(SDPTI_ACTIVITY_KIND_MEMOPS));
  }
  // old profiler stop to record
  RecordEvent::Instance().ActivityDisable();
  return C_SUCCESS;
}

// TCCL
tcclDataType_t PDDataTypeToTcclDataType(C_DataType dtype) {
  if (dtype == C_DataType::FLOAT32) {
    return tcclFloat;
  } else if (dtype == C_DataType::FLOAT64) {
    return tcclDouble;
  } else if (dtype == C_DataType::UINT32) {
    return tcclUint;
  } else if (dtype == C_DataType::UINT8) {
    return tcclUint8;
  } else if (dtype == C_DataType::INT32) {
    return tcclInt;
  } else if (dtype == C_DataType::INT8) {
    return tcclInt8;
  } else if (dtype == C_DataType::INT64) {
    return tcclInt64;
  } else if (dtype == C_DataType::UINT64) {
    return tcclUint64;
  } else if (dtype == C_DataType::FLOAT16) {
    return tcclHalf;
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Datatype %s in tccl is not supported.", dtype));
  }
}

tcclRedOp_t PDReduceOpToTcclReduceOp(C_CCLReduceOp op) {
  if (op == C_CCLReduceOp::MIN) {
    return tcclMin;
  } else if (op == C_CCLReduceOp::MAX) {
    return tcclMax;
  } else if (op == C_CCLReduceOp::SUM) {
    return tcclSum;
  } else if (op == C_CCLReduceOp::PRODUCT) {
    return tcclProd;
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Reduceop %s in tccl is not supported.", op));
  }
}

C_Status XcclGetUniqueIdSize(size_t *size) {
  *size = sizeof(tcclUniqueId);
  return C_SUCCESS;
}

C_Status XcclGetUniqueId(C_CCLRootId *unique_id) {
  if (unique_id->sz != sizeof(tcclUniqueId)) {
    LOG(ERROR) << "unique_id->sz must be equal sizeof(tcclUniqueId)";
    return C_FAILED;
  }
  TCCL_CHECK(
      tcclGetUniqueId(reinterpret_cast<tcclUniqueId *>(unique_id->data)));
  return C_SUCCESS;
}

C_Status XcclCommInitRank(size_t nranks,
                          C_CCLRootId *unique_id,
                          size_t rank,
                          C_CCLComm *comm) {
  tcclUniqueId *commId = reinterpret_cast<tcclUniqueId *>(unique_id->data);
  TCCL_CHECK(tcclCommInitRank(reinterpret_cast<tcclComm_t *>(comm),
                              static_cast<int>(nranks),
                              *commId,
                              static_cast<int>(rank)));
  return C_SUCCESS;
}

C_Status XcclDestroyComm(C_CCLComm comm) {
  TCCL_CHECK(tcclCommDestroy(reinterpret_cast<tcclComm_t>(comm)));
  return C_SUCCESS;
}

C_Status XcclAllReduce(void *send_buf,
                       void *recv_buf,
                       size_t count,
                       C_DataType data_type,
                       C_CCLReduceOp op,
                       C_CCLComm comm,
                       C_Stream stream) {
  CustomSDAAStream_t sdaa_stream = reinterpret_cast<CustomSDAAStream_t>(stream);
  lastCommStream::Instance().update(
      reinterpret_cast<sdaaStream_t>(sdaa_stream->pStream));
  TCCL_CHECK(tcclAllReduce(send_buf,
                           recv_buf,
                           count,
                           PDDataTypeToTcclDataType(data_type),
                           PDReduceOpToTcclReduceOp(op),
                           reinterpret_cast<tcclComm_t>(comm),
                           lastCommStream::Instance().get()));
  return C_SUCCESS;
}

C_Status XcclBroadcast(void *buf,
                       size_t count,
                       C_DataType data_type,
                       size_t root,
                       C_CCLComm comm,
                       C_Stream stream) {
  CustomSDAAStream_t sdaa_stream = reinterpret_cast<CustomSDAAStream_t>(stream);
  lastCommStream::Instance().update(
      reinterpret_cast<sdaaStream_t>(sdaa_stream->pStream));
  TCCL_CHECK(tcclBroadcast(buf,
                           buf,
                           count,
                           PDDataTypeToTcclDataType(data_type),
                           static_cast<int>(root),
                           reinterpret_cast<tcclComm_t>(comm),
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
  CustomSDAAStream_t sdaa_stream = reinterpret_cast<CustomSDAAStream_t>(stream);
  lastCommStream::Instance().update(
      reinterpret_cast<sdaaStream_t>(sdaa_stream->pStream));
  TCCL_CHECK(tcclReduce(send_buf,
                        recv_buf,
                        count,
                        PDDataTypeToTcclDataType(data_type),
                        PDReduceOpToTcclReduceOp(op),
                        static_cast<int>(root),
                        reinterpret_cast<tcclComm_t>(comm),
                        lastCommStream::Instance().get()));
  return C_SUCCESS;
}

C_Status XcclAllGather(void *send_buf,
                       void *recv_buf,
                       size_t count,
                       C_DataType data_type,
                       C_CCLComm comm,
                       C_Stream stream) {
  CustomSDAAStream_t sdaa_stream = reinterpret_cast<CustomSDAAStream_t>(stream);
  lastCommStream::Instance().update(
      reinterpret_cast<sdaaStream_t>(sdaa_stream->pStream));
  TCCL_CHECK(tcclAllGather(send_buf,
                           recv_buf,
                           count,
                           PDDataTypeToTcclDataType(data_type),
                           reinterpret_cast<tcclComm_t>(comm),
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
  CustomSDAAStream_t sdaa_stream = reinterpret_cast<CustomSDAAStream_t>(stream);
  lastCommStream::Instance().update(
      reinterpret_cast<sdaaStream_t>(sdaa_stream->pStream));
  TCCL_CHECK(tcclReduceScatter(send_buf,
                               recv_buf,
                               count,
                               PDDataTypeToTcclDataType(data_type),
                               PDReduceOpToTcclReduceOp(op),
                               reinterpret_cast<tcclComm_t>(comm),
                               lastCommStream::Instance().get()));
  return C_SUCCESS;
}

C_Status XcclSend(void *send_buf,
                  size_t count,
                  C_DataType data_type,
                  size_t dest_rank,
                  C_CCLComm comm,
                  C_Stream stream) {
  CustomSDAAStream_t sdaa_stream = reinterpret_cast<CustomSDAAStream_t>(stream);
  lastCommStream::Instance().update(
      reinterpret_cast<sdaaStream_t>(sdaa_stream->pStream));
  TCCL_CHECK(tcclSend(send_buf,
                      count,
                      PDDataTypeToTcclDataType(data_type),
                      static_cast<int>(dest_rank),
                      reinterpret_cast<tcclComm_t>(comm),
                      lastCommStream::Instance().get()));
  return C_SUCCESS;
}

C_Status XcclRecv(void *recv_buf,
                  size_t count,
                  C_DataType data_type,
                  size_t src_rank,
                  C_CCLComm comm,
                  C_Stream stream) {
  CustomSDAAStream_t sdaa_stream = reinterpret_cast<CustomSDAAStream_t>(stream);
  lastCommStream::Instance().update(
      reinterpret_cast<sdaaStream_t>(sdaa_stream->pStream));
  TCCL_CHECK(tcclRecv(recv_buf,
                      count,
                      PDDataTypeToTcclDataType(data_type),
                      static_cast<int>(src_rank),
                      reinterpret_cast<tcclComm_t>(comm),
                      lastCommStream::Instance().get()));
  return C_SUCCESS;
}

void InitPlugin(CustomRuntimeParams *params) {
  PADDLE_CUSTOM_RUNTIME_CHECK_VERSION(params);
  params->device_type = const_cast<char *>("sdaa");
  params->sub_device_type = const_cast<char *>("v0.1");

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
  params->interface->query_stream = QueryStream;

  params->interface->create_event = CreateEvent;
  params->interface->destroy_event = DestroyEvent;
  params->interface->record_event = RecordEvent;
  params->interface->query_event = QueryEvent;

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
  params->interface->device_memory_allocate = DeviceAllocate;
  // params->interface->host_memory_allocate = HostAllocate;
  params->interface->unified_memory_allocate = nullptr;
  params->interface->device_memory_deallocate = DeviceDeallocate;
  // params->interface->host_memory_deallocate = HostDeallocate;
  params->interface->unified_memory_deallocate = nullptr;

  params->interface->get_device_count = GetDevicesCount;
  params->interface->get_device_list = GetDevicesList;
  params->interface->device_memory_stats = DeviceMemStats;
  params->interface->device_memory_set = DeviceMemSet;
  params->interface->device_min_chunk_size = DeviceMinChunkSize;
  params->interface->device_max_chunk_size = DeviceMaxChunkSize;

  params->interface->profiler_collect_trace_data = ProfilerCollectData;
  params->interface->profiler_initialize = ProfilerInitialize;
  params->interface->profiler_finalize = ProfilerFinalize;
  params->interface->profiler_start_tracing = ProfilerStart;
  params->interface->profiler_stop_tracing = ProfilerStop;
  params->interface->profiler_prepare_tracing = ProfilerPrepare;

  // TCCL
  params->interface->xccl_all_gather = XcclAllGather;
  params->interface->xccl_all_reduce = XcclAllReduce;
  params->interface->xccl_broadcast = XcclBroadcast;
  params->interface->xccl_comm_init_rank = XcclCommInitRank;
  params->interface->xccl_destroy_comm = XcclDestroyComm;
  params->interface->xccl_get_unique_id = XcclGetUniqueId;
  params->interface->xccl_get_unique_id_size = XcclGetUniqueIdSize;
  params->interface->xccl_group_end = nullptr;
  params->interface->xccl_group_start = nullptr;
  params->interface->xccl_recv = XcclRecv;
  params->interface->xccl_reduce = XcclReduce;
  params->interface->xccl_reduce_scatter = XcclReduceScatter;
  params->interface->xccl_send = XcclSend;
}
