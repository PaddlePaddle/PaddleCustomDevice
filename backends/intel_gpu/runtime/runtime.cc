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
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>

#include "./dnn_support.hpp"
#include "paddle/phi/backends/device_ext.h"

#define MEMORY_FRACTION 0.5f

C_Status Init() {
  std::cout << "custom_cpu plugin compiled with ";
#ifdef __clang__
  std::cout << "clang\n";
#else
  std::cout << "gcc\n";
#endif
  return C_SUCCESS;
}

// **** Types *****
template <class T>
using up_t = std::unique_ptr<T>;

DeviceConfigPtr devconf;
std::mutex mx;
std::recursive_mutex rmux;

auto intel_match = [](const sycl::device &dev) -> bool {
  const auto name = dev.template get_info<sycl::info::device::name>();
  return (name.find("Intel(R) Graphics") != std::string::npos) ? true : false;
};

struct DeviceCtx {
  sycl::device _dev;
  std::vector<std::unique_ptr<sycl::queue>> _streams;
  bool _def_stream;
  size_t allocated_mem;
  size_t _dev_memory_size;
  explicit DeviceCtx(sycl::device dev)
      : _dev{std::move(dev)},
        _def_stream{true},
        allocated_mem{0},
        _dev_memory_size(_dev.get_info<sycl::info::device::global_mem_size>()) {
  }

  sycl::queue *create_stream() {
    auto u_ptr = std::make_unique<sycl::queue>(_dev);
    _streams.push_back(std::move(u_ptr));

    return &(*(*(_streams.rbegin())));
  }

  sycl::queue *getDefaultOrCreate() {
    if (_def_stream && _streams.size()) {
      _def_stream = false;
      return _streams[0].get();
    }

    return create_stream();
  }

  sycl::queue &getStream(size_t index = 0) {
    if (!_streams.size()) create_stream();
    return *(_streams[index]);
  }

  void copy(const sycl::queue &q, void *dst, const void *src, size_t size) {
    q.submit([&](sycl::handler &h) { h.memcpy(dst, src, size); });
    q.wait();
  }

  void copy(void *dst, const void *src, size_t size) {
    copy(getStream(), dst, src, size);
  }

  sycl::queue &getStream(C_Stream stream) {
    auto it = std::find_if(
        _streams.begin(), _streams.end(), [stream](auto &single_stream) {
          return single_stream.get() == reinterpret_cast<sycl::queue *>(stream);
        });

    if (it == _streams.end()) {
      show_error("*FATAL ERROR STREAM not found*");
    }
    return **it;
  }

  size_t getMemorySize() { return _dev_memory_size; }

  size_t getFreeMemorySize() { return (getMemorySize() - allocated_mem) / 8; }

  void alloc_mem(size_t _size) { allocated_mem += _size; }

  void free_mem(size_t _size) { allocated_mem -= _size; }
};

std::vector<DeviceCtx> reg_dev;

C_Status InitDevice(const C_Device device) {
  InitializeDevConf();
  show_debug("init-device : device->id=" << device->id);
  return C_SUCCESS;
}

C_Status SetDevice(const C_Device device) {
  show_debug("set-device : device->id=" << device->id);
  return C_SUCCESS;
}

C_Status GetDevice(const C_Device device) {
  device->id = 0;
  show_debug("get-device() : device->id=" << device->id);
  return C_SUCCESS;
}

C_Status DestroyDevice(const C_Device device) {
  show_debug("destroy-device() : device->id=" << device->id);
  return C_SUCCESS;
}

C_Status Finalize() { return C_SUCCESS; }

C_Status GetDevicesCount(size_t *count) {
  if (!reg_dev.size()) {
    InitializeDevConf();
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);

    std::copy_if(devices.begin(),
                 devices.end(),
                 std::back_inserter(reg_dev),
                 intel_match);

    if (!reg_dev.size()) {
      show_error("No Intel GPUs found");
      return C_FAILED;
    }
  }

  *count = reg_dev.size();
  show_debug("getdevicescount() count=" << *count);

  return C_SUCCESS;
}

C_Status GetDevicesList(size_t *devices) {
  show_debug("getdeviceList() fill=" << reg_dev.size());

  for (size_t i = 0; i < reg_dev.size(); ++i) devices[i] = static_cast<int>(i);

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
  show_debug("async-memcpy  dst=" << dst << " src=" << src << " size=" << size);

  auto &dev_ctx = reg_dev[device->id];

  auto &dev_stream = dev_ctx.getStream(stream);

  dev_ctx.copy(dev_stream, dst, src, size);

  return C_SUCCESS;
}

C_Status MemCpyP2P(const C_Device dst_device,
                   const C_Device src_device,
                   void *dst,
                   const void *src,
                   size_t size) {
  show_debug("memcpy-p2p  memcpy() dst=" << dst << " src=" << src
                                         << " size=" << size);
  memcpy(dst, src, size);
  return C_SUCCESS;
}

C_Status AsyncMemCpyP2P(const C_Device dst_device,
                        const C_Device src_device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size) {
  show_debug("async-memcpy-p2p  memcpy() dst=" << dst << " src=" << src
                                               << " size=" << size);
  memcpy(dst, src, size);
  return C_SUCCESS;
}

C_Status Allocate(const C_Device device, void **ptr, size_t size) {
  show_memory("request allocate size=" << size << " device=" << device->id);

  if (size > reg_dev[device->id].getFreeMemorySize()) {
    show_error("## No free memory INTERNAL ERROR OUT OF MEMORY requested size="
               << size << " left=" << reg_dev[device->id].getFreeMemorySize()
               << " ##");
    return C_FAILED;
  }

  auto &stream = reg_dev[device->id].getStream();

  *ptr = sycl::aligned_alloc_device(64, size, stream);

  if (!*ptr) {
    show_error("#### Error : Can't allocate memory size="
               << size << " free_mem_size="
               << reg_dev[device->id].getFreeMemorySize() << " ####");
    return C_FAILED;
  }

  reg_dev[device->id].alloc_mem(size);

  show_memory("allocate success size="
              << size << " left=" << reg_dev[device->id].getFreeMemorySize());

  return C_SUCCESS;
}

C_Status Deallocate(const C_Device device, void *ptr, size_t size) {
  show_memory("deallocate size=" << size);

  auto &stream = reg_dev[device->id].getStream();

  sycl::free(ptr, stream);

  reg_dev[device->id].free_mem(size);

  return C_SUCCESS;
}

C_Status CreateStream(const C_Device device, C_Stream *stream) {
  show_debug("create-stream for device=" << device->id);

  *stream = reinterpret_cast<C_Stream>(reg_dev[device->id].create_stream());

  return C_SUCCESS;
}

C_Status DestroyStream(const C_Device device, C_Stream stream) {
  show_debug("destroy-stream device->id=" << device->id
                                          << " stream=" << stream);

  auto &_streams = reg_dev[device->id]._streams;
  auto it = std::find_if(
      _streams.begin(), _streams.end(), [stream](auto &single_stream) {
        return single_stream.get() == reinterpret_cast<sycl::queue *>(stream);
      });

  if (it != _streams.end()) _streams.erase(it);

  return C_SUCCESS;
}

C_Status CreateEvent(const C_Device device, C_Event *event) {
  show_debug("create-event devid=" << device->id);
  return C_SUCCESS;
}

C_Status RecordEvent(const C_Device device, C_Stream stream, C_Event event) {
  show_debug("record-event devid=" << device->id);
  return C_SUCCESS;
}

C_Status DestroyEvent(const C_Device device, C_Event event) {
  show_debug("destroy-event devid=" << device->id);
  return C_SUCCESS;
}

C_Status SyncDevice(const C_Device device) {
  show_debug("sync-device devid=" << device->id);
  auto &dev_ctx = reg_dev[device->id];

  for (auto &stream : dev_ctx._streams) {
    stream->wait();  // ???????
  }
  return C_SUCCESS;
}

C_Status SyncStream(const C_Device device, C_Stream stream) {
  show_debug("sync-stream devid=" << device->id);
  auto ret_stream = reg_dev[device->id].getStream(stream);
  ret_stream.wait();

  return C_SUCCESS;
}

C_Status SyncEvent(const C_Device device, C_Event event) {
  show_debug("sync-event devid=" << device->id);
  return C_SUCCESS;
}

C_Status StreamWaitEvent(const C_Device device,
                         C_Stream stream,
                         C_Event event) {
  show_debug("stream-wait-event devid=" << device->id);

  return C_SUCCESS;
}

C_Status VisibleDevices(size_t *devices) {
  show_debug("visible-devices devices=" << *devices);
  return C_SUCCESS;
}

C_Status DeviceMemStats(const C_Device device,
                        size_t *total_memory,
                        size_t *free_memory) {
  auto &dev_ctx = reg_dev[device->id];
  *total_memory = dev_ctx.getMemorySize();
  *free_memory = dev_ctx.getFreeMemorySize();

  show_memory("device-mem-stat device=" << device->id
                                        << " TotalMemory=" << *total_memory
                                        << " FreeMemory=" << *free_memory);

  return C_SUCCESS;
}

C_Status DeviceMinChunkSize(const C_Device device, size_t *size) {
  show_memory("device-min-chunk-size device=" << device->id);
  InitializeDevConf();
  *size = devconf->chunk_size;
  return C_SUCCESS;
}

C_Status MemoryCopyH2D(const C_Device device,
                       void *dst,
                       const void *src,
                       size_t size) {
  show_memory("memory-copy-h2d dst=" << dst << " src=" << src
                                     << " size=" << size);

  reg_dev[device->id].copy(dst, src, size);

  return C_SUCCESS;
}

C_Status MemoryCopyD2H(const C_Device device,
                       void *dst,
                       const void *src,
                       size_t size) {
  show_memory("memory-copy-d2h size=" << size << " dst=" << dst
                                      << " src=" << src);

  reg_dev[device->id].copy(dst, src, size);

  return C_SUCCESS;
}

C_Status MemoryCopyD2D(const C_Device device,
                       void *dst,
                       const void *src,
                       size_t size) {
  show_memory("memory-copy-d2d size=" << size << " dst=" << dst
                                      << " src=" << src);

  reg_dev[device->id].copy(dst, src, size);

  return C_SUCCESS;
}

void InitPlugin(CustomRuntimeParams *params) {
  PADDLE_CUSTOM_RUNTIME_CHECK_VERSION(params);
  params->device_type = "intel_gpu";
  params->sub_device_type = "v0.1";
  show_debug("init-plugin " << params->device_type);

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

  // params->interface->memory_copy_h2d = MemCpy;
  params->interface->memory_copy_h2d = MemoryCopyH2D;
  // params->interface->memory_copy_d2d = MemCpy;
  params->interface->memory_copy_d2d = MemoryCopyD2D;

  params->interface->memory_copy_d2h = MemoryCopyD2H;

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
}
