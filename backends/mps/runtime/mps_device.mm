#include <memory>
#include <mutex>

#include "mps_device.h"
#include "mps_runtime.h"
#include "mps_stream.h"

#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

namespace mps {

static std::unique_ptr<MPSDevice> mps_device;
static std::once_flag mpsdev_init;

MPSDevice *MPSDevice::getInstance() {
  std::call_once(mpsdev_init, [] { mps_device = std::unique_ptr<MPSDevice>(new MPSDevice()); });
  return mps_device.get();
}

MPSDevice::~MPSDevice() {
  [_mtl_device release];
  _mtl_device = nil;
}

MPSDevice::MPSDevice() : _mtl_device(nil) { _mtl_device = MTLCreateSystemDefaultDevice(); }

bool init_device() {
  return MPSDevice::getInstance()->device() != nil && getCurrentMPSStream() != nil;
}

bool alloc_memory(void **ptr, size_t size) {
  *ptr =
      (void *)[MPSDevice::getInstance()->device() newBufferWithLength:size
                                                              options:MTLResourceStorageModeShared];
  return *ptr ? true : false;
}

bool dealloc_memory(void *ptr) {
  if (!ptr) return true;
  id<MTLBuffer> buffer = (id<MTLBuffer>)ptr;
  [buffer release];
  ptr = 0;
  return true;
}

bool memcpy_d2d(void *dst, const void *src, size_t size) {
  if (!dst || !src) return false;
  id<MTLBuffer> dst_buffer = (id<MTLBuffer>)dst;
  id<MTLBuffer> src_buffer = (id<MTLBuffer>)src;
  memcpy([dst_buffer contents], [src_buffer contents], size);
  return true;
}

bool memcpy_d2h(void *dst, const void *src, size_t size) {
  if (!dst || !src) return false;
  id<MTLBuffer> src_buffer = (id<MTLBuffer>)src;
  memcpy(dst, [src_buffer contents], size);
  return true;
}

bool memcpy_h2d(void *dst, const void *src, size_t size) {
  if (!dst || !src) return false;
  id<MTLBuffer> dst_buffer = (id<MTLBuffer>)dst;
  memcpy([dst_buffer contents], src, size);
  return true;
}

}  // namespace mps
