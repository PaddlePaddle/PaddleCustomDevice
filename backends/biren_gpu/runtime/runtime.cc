#include "runtime.h"

#include <unordered_map>
#include <vector>

#include "glog/logging.h"
#include "sucl/br_cl.h"

#define EXCEPTION_CHECK(statements)                                            \
  try {                                                                        \
    statements                                                                 \
  } catch (br_device::Status & e) {                                            \
    printf("[BR Runtime] exception captured: at %s:%d\n", __FILE__, __LINE__); \
    printf("[Internal] %s, at %s:%lu\n", e.what(), e.File(), e.Line());        \
    return C_FAILED;                                                           \
  }

constexpr std::size_t br_compute_capability = 75;

std::vector<br_device::sucl::Runtime> kRuntimes;

C_Status set_device(const C_Device device) {
  VLOG(6) << "enter " << __FUNCTION__ << ", device " << device;
  PARAM_CHECK_PTR(device, C_ERROR);
  EXCEPTION_CHECK(kRuntimes[device->id].SetDevice();)
  return C_SUCCESS;
}

C_Status get_device(const C_Device device) {
  VLOG(6) << "enter " << __FUNCTION__;
  PARAM_CHECK_PTR(device, C_ERROR);
  EXCEPTION_CHECK(device->id = br_device::sucl::Runtime::GetDevice().id;)
  return C_SUCCESS;
}

C_Status get_device_count(size_t *count) {
  VLOG(10) << "enter " << __FUNCTION__;
  PARAM_CHECK_PTR(count, C_ERROR);

  EXCEPTION_CHECK({
    auto dev_cnt = br_device::sucl::Runtime::GetDeviceCount();
    if (!kRuntimes.size()) {
      for (auto i = 0; i < dev_cnt; ++i) {
        kRuntimes.emplace_back(br_device::sucl::Device({i}));
      }
    }
    *count = dev_cnt;
  })

  return C_SUCCESS;
}

C_Status get_device_list(size_t *device) {
  VLOG(10) << "enter " << __FUNCTION__;
  PARAM_CHECK_PTR(device, C_ERROR);

  EXCEPTION_CHECK({
    auto dev_cnt = br_device::sucl::Runtime::GetDeviceCount();
    for (size_t i = 0; i < dev_cnt; i++) {
      device[i] = i;
    }
  })

  return C_SUCCESS;
}

C_Status get_compute_capability(size_t *compute_capability) {
  VLOG(10) << "enter " << __FUNCTION__;
  PARAM_CHECK_PTR(compute_capability, C_ERROR);
  *compute_capability = br_compute_capability;
  return C_SUCCESS;
}

C_Status get_runtime_version(size_t *version) {
  VLOG(10) << "enter " << __FUNCTION__;
  PARAM_CHECK_PTR(version, C_ERROR);
  EXCEPTION_CHECK(*version = br_device::sucl::Runtime::GetRuntimeVersion();)
  return C_SUCCESS;
}

C_Status get_driver_version(size_t *version) {
  VLOG(10) << "enter " << __FUNCTION__;
  PARAM_CHECK_PTR(version, C_ERROR);
  EXCEPTION_CHECK(*version = br_device::sucl::Runtime::GetDriverVersion();)
  return C_SUCCESS;
}

C_Status memcpy_h2d(const C_Device device, void *dst, const void *src,
                    size_t size) {
  VLOG(4) << "enter " << __FUNCTION__ << ", src " << src << ", dst " << dst;
  PARAM_CHECK_PTR(device, C_ERROR);
  PARAM_CHECK_PTR(dst, C_ERROR);
  PARAM_CHECK_PTR(src, C_ERROR);
  EXCEPTION_CHECK(kRuntimes[device->id].MemcpyH2D(dst, src, size);)
  return C_SUCCESS;
}

C_Status memcpy_d2d(const C_Device device, void *dst, const void *src,
                    size_t size) {
  VLOG(4) << "enter " << __FUNCTION__ << ", src " << src << ", dst " << dst;
  PARAM_CHECK_PTR(device, C_ERROR);
  PARAM_CHECK_PTR(dst, C_ERROR);
  PARAM_CHECK_PTR(src, C_ERROR);
  EXCEPTION_CHECK(kRuntimes[device->id].MemcpyD2D(dst, src, size);)
  return C_SUCCESS;
}

C_Status memcpy_d2h(const C_Device device, void *dst, const void *src,
                    size_t size) {
  VLOG(4) << "enter " << __FUNCTION__ << ", src " << src << ", dst " << dst;
  PARAM_CHECK_PTR(device, C_ERROR);
  PARAM_CHECK_PTR(dst, C_ERROR);
  PARAM_CHECK_PTR(src, C_ERROR);
  EXCEPTION_CHECK(kRuntimes[device->id].MemcpyD2H(dst, src, size);)
  return C_SUCCESS;
}

C_Status async_memcpy_h2d(const C_Device device, C_Stream stream, void *dst,
                          const void *src, size_t size) {
  VLOG(4) << "enter " << __FUNCTION__ << ", src " << src << ", dst " << dst
          << ", stream " << stream;
  PARAM_CHECK_PTR(dst, C_ERROR);
  PARAM_CHECK_PTR(src, C_ERROR);

  EXCEPTION_CHECK(kRuntimes[device->id].AsyncMemcpyH2D(
      reinterpret_cast<br_device::sucl::PStream>(stream), dst, src, size);)

  return C_SUCCESS;
}

C_Status async_memcpy_d2d(const C_Device device, C_Stream stream, void *dst,
                          const void *src, size_t size) {
  VLOG(4) << "enter " << __FUNCTION__ << ", src " << src << ", dst " << dst
          << ", stream " << stream;
  PARAM_CHECK_PTR(dst, C_ERROR);
  PARAM_CHECK_PTR(src, C_ERROR);

  EXCEPTION_CHECK(kRuntimes[device->id].AsyncMemcpyD2D(
      reinterpret_cast<br_device::sucl::PStream>(stream), dst, src, size);)

  return C_SUCCESS;
}

C_Status async_memcpy_d2h(const C_Device device, C_Stream stream, void *dst,
                          const void *src, size_t size) {
  VLOG(4) << "enter " << __FUNCTION__ << ", src " << src << ", dst " << dst
          << ", stream " << stream;
  PARAM_CHECK_PTR(dst, C_ERROR);
  PARAM_CHECK_PTR(src, C_ERROR);

  EXCEPTION_CHECK(kRuntimes[device->id].AsyncMemcpyD2H(
      reinterpret_cast<br_device::sucl::PStream>(stream), dst, src, size);)

  return C_SUCCESS;
}

C_Status allocate(const C_Device device, void **ptr, size_t size) {
  VLOG(10) << "enter " << __FUNCTION__;
  PARAM_CHECK_PTR(device, C_ERROR);

  EXCEPTION_CHECK({
    auto mem_info = kRuntimes[device->id].MemStats();
    size_t gpu_free_size = mem_info.free_memory;
    PARAM_CHECK_MEM_SIZE(size, gpu_free_size, C_FAILED);
    void *temp = kRuntimes[device->id].Malloc(size);
    PARAM_CHECK_PTR(temp, C_ERROR);
    *ptr = (void *)temp;
  })

  return C_SUCCESS;
}

C_Status deallocate(const C_Device device, void *ptr, size_t size) {
  VLOG(10) << "enter " << __FUNCTION__;
  PARAM_CHECK_PTR(device, C_ERROR);
  PARAM_CHECK_PTR(ptr, C_ERROR);

  EXCEPTION_CHECK(kRuntimes[device->id].Free(ptr);)

  return C_SUCCESS;
}

C_Status create_stream(const C_Device device, C_Stream *stream) {
  VLOG(10) << "enter " << __FUNCTION__;
  PARAM_CHECK_PTR(device, C_ERROR);

  EXCEPTION_CHECK({
    *stream = reinterpret_cast<C_Stream>(kRuntimes[device->id].CreateStream());
    VLOG(4) << "create stream : " << *stream;
  })

  return C_SUCCESS;
}

C_Status destroy_stream(const C_Device device, C_Stream stream) {
  VLOG(4) << "enter " << __FUNCTION__;
  PARAM_CHECK_PTR(device, C_ERROR);
  PARAM_CHECK_PTR(stream, C_ERROR);

  EXCEPTION_CHECK(kRuntimes[device->id].DestroyStream(
      reinterpret_cast<br_device::sucl::PStream>(stream));)

  return C_SUCCESS;
}

C_Status create_event(const C_Device device, C_Event *event) {
  VLOG(10) << "enter " << __FUNCTION__;
  PARAM_CHECK_PTR(device, C_ERROR);

  EXCEPTION_CHECK({
    *event = reinterpret_cast<C_Event>(kRuntimes[device->id].CreateEvent());
    VLOG(4) << "create event " << *event;
  })

  return C_SUCCESS;
}

C_Status record_event(const C_Device device, C_Stream stream, C_Event event) {
  VLOG(4) << "enter " << __FUNCTION__ << ", event " << event;
  PARAM_CHECK_PTR(device, C_ERROR);
  PARAM_CHECK_PTR(stream, C_ERROR);
  PARAM_CHECK_PTR(event, C_ERROR);

  EXCEPTION_CHECK({
    kRuntimes[device->id].RecordEvent(
        reinterpret_cast<br_device::sucl::PEvent>(event),
        reinterpret_cast<br_device::sucl::PStream>(stream));
  })

  return C_SUCCESS;
}

C_Status destroy_event(const C_Device device, C_Event event) {
  VLOG(4) << "enter " << __FUNCTION__ << ", event " << event;
  PARAM_CHECK_PTR(device, C_ERROR);
  PARAM_CHECK_PTR(event, C_ERROR);

  EXCEPTION_CHECK(kRuntimes[device->id].DestroyEvent(
      reinterpret_cast<br_device::sucl::PEvent>(event));)

  return C_SUCCESS;
}

C_Status sync_device(const C_Device device) {
  VLOG(4) << "enter " << __FUNCTION__;
  PARAM_CHECK_PTR(device, C_ERROR);

  EXCEPTION_CHECK(kRuntimes[device->id].SynchronizeDevice();)

  return C_SUCCESS;
}

C_Status sync_stream(const C_Device device, C_Stream stream) {
  VLOG(4) << "enter " << __FUNCTION__ << ", stream " << stream;
  PARAM_CHECK_PTR(device, C_ERROR);
  PARAM_CHECK_PTR(stream, C_ERROR);

  VLOG(4) << "device: " << device << ", id: " << device->id;
  VLOG(4) << "runtime size: " << kRuntimes.size();

  EXCEPTION_CHECK(kRuntimes[device->id].SynchronizeStream(
      reinterpret_cast<br_device::sucl::PStream>(stream));)

  return C_SUCCESS;
}

C_Status sync_event(const C_Device device, C_Event event) {
  VLOG(4) << "enter " << __FUNCTION__ << ", event " << event;
  PARAM_CHECK_PTR(device, C_ERROR);
  PARAM_CHECK_PTR(event, C_ERROR);

  EXCEPTION_CHECK(kRuntimes[device->id].SynchronizeEvent(
      reinterpret_cast<br_device::sucl::PEvent>(event));)

  return C_SUCCESS;
}

C_Status stream_wait_event(const C_Device device, C_Stream stream,
                           C_Event event) {
  VLOG(4) << "enter " << __FUNCTION__ << ", stream " << stream << ", event "
          << event;
  PARAM_CHECK_PTR(device, C_ERROR);
  PARAM_CHECK_PTR(stream, C_ERROR);
  PARAM_CHECK_PTR(event, C_ERROR);

  EXCEPTION_CHECK(kRuntimes[device->id].StreamWaitEvent(
      reinterpret_cast<br_device::sucl::PStream>(stream),
      reinterpret_cast<br_device::sucl::PEvent>(event));)

  return C_SUCCESS;
}

C_Status memstats(const C_Device device, size_t *total_memory,
                  size_t *free_memory) {
  VLOG(4) << "enter " << __FUNCTION__;
  PARAM_CHECK_PTR(device, C_ERROR);
  PARAM_CHECK_PTR(total_memory, C_ERROR);
  PARAM_CHECK_PTR(free_memory, C_ERROR);

  EXCEPTION_CHECK({
    auto mem_info = kRuntimes[device->id].MemStats();
    *total_memory = mem_info.total_memory;
    *free_memory = mem_info.free_memory;
  })

  return C_SUCCESS;
}

C_Status get_min_chunk_size(const C_Device device, size_t *size) {
  VLOG(4) << "enter " << __FUNCTION__;
  PARAM_CHECK_PTR(device, C_ERROR);
  PARAM_CHECK_PTR(size, C_ERROR);

  EXCEPTION_CHECK({
    kRuntimes[device->id].SetDevice();
    *size = 1;
  })

  return C_SUCCESS;
}

C_Status get_max_chunk_size(const C_Device device, size_t *size) {
  VLOG(4) << "enter " << __FUNCTION__;
  PARAM_CHECK_PTR(device, C_ERROR);
  PARAM_CHECK_PTR(size, C_ERROR);

  EXCEPTION_CHECK({
    kRuntimes[device->id].SetDevice();
    *size = 0;
  })

  return C_SUCCESS;
}

C_Status init() {
  return C_SUCCESS;
}

C_Status deinit() {
  return C_SUCCESS;
}

void InitPlugin(CustomRuntimeParams *params) {
  PADDLE_CUSTOM_RUNTIME_CHECK_VERSION(params);
  params->device_type = (char *)"SUPA";
  params->sub_device_type = (char *)"V1";

  params->interface->set_device = set_device;
  params->interface->get_device = get_device;
  params->interface->create_stream = create_stream;
  params->interface->destroy_stream = destroy_stream;
  params->interface->create_event = create_event;
  params->interface->destroy_event = destroy_event;
  params->interface->record_event = record_event;
  params->interface->synchronize_device = sync_device;
  params->interface->synchronize_stream = sync_stream;
  params->interface->synchronize_event = sync_event;
  params->interface->stream_wait_event = stream_wait_event;
  params->interface->memory_copy_h2d = memcpy_h2d;
  params->interface->memory_copy_d2d = memcpy_d2d;
  params->interface->memory_copy_d2h = memcpy_d2h;
  params->interface->async_memory_copy_h2d = async_memcpy_h2d;
  params->interface->async_memory_copy_d2d = async_memcpy_d2d;
  params->interface->async_memory_copy_d2h = async_memcpy_d2h;
  params->interface->device_memory_allocate = allocate;
  params->interface->device_memory_deallocate = deallocate;
  params->interface->get_device_count = get_device_count;
  params->interface->get_device_list = get_device_list;
  params->interface->device_memory_stats = memstats;
  params->interface->device_min_chunk_size = get_min_chunk_size;
  params->interface->device_max_chunk_size = get_max_chunk_size;
  params->interface->get_compute_capability = get_compute_capability;
  params->interface->get_runtime_version = get_runtime_version;
  params->interface->get_driver_version = get_driver_version;

  params->interface->initialize = init;
  params->interface->finalize = deinit;
}
