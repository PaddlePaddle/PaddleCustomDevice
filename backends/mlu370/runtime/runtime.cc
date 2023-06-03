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
  mluStream_t mlu_stream = new CustomMLUStream();

  cnrtQueue_t queue;
  PADDLE_ENFORCE_MLU_SUCCESS(cnrtQueueCreate(&queue));

  cnnlHandle_t handle;
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlCreate(&handle));
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetQueue(handle, queue));

  mlu_stream->queue = queue;
  mlu_stream->handle = handle;

  *stream = reinterpret_cast<C_Stream>(mlu_stream);

  return C_SUCCESS;
}

C_Status DestroyStream(const C_Device device, C_Stream stream) {
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlDestroy(GetHandle(stream)));
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

void InitPlugin(CustomRuntimeParams *params) {
  PADDLE_CUSTOM_RUNTIME_CHECK_VERSION(params);

  params->device_type = "CustomMLU";
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
}
