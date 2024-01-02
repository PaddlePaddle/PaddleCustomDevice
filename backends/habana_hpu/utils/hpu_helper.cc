#include <map>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

#include "paddle/phi/backends/device_ext.h"
#include "synapse_api.h"
#include "synapse_common_types.h"
#include "utils/hpu_utils.h"
#include "utils/hpu_helper.h"

typedef std::pair<synSectionHandle, bool> sectionWithFirstIndication;
static std::unordered_map<std::string, sectionWithFirstIndication> sectionMap;
static std::mutex mut_;
static std::map<synDeviceId, C_Stream> streamMap;

synStatus HostMap(const synDeviceId deviceId,
                  const uint64_t size,
                  const void* buffer) {
  // std::lock_guard<std::mutex> lock_guard(mut_);
  return synHostMap(deviceId, size, buffer);
}

synStatus HostUnmap(const synDeviceId deviceId, const void* buffer) {
  // std::lock_guard<std::mutex> lock_guard(mut_);
  return synHostUnmap(deviceId, buffer);
}

bool waitForIdleDevice(uint32_t* deviceId,
                       synDeviceType& deviceType,
                       int maxSecondsWait) {
  int oneSecond = 1;
  synDeviceInfo deviceInfo;
  auto start = std::chrono::high_resolution_clock::now();
  auto maxWaitTime = start + std::chrono::seconds(maxSecondsWait);
  synStatus status = synFail;
  char* pci_address = getenv("PCI_ADDRESS");
  while (status != synSuccess &&
         std::chrono::high_resolution_clock::now() < maxWaitTime) {
    status = synDeviceAcquire(deviceId, pci_address);
    if (status != synSuccess) {
      sleep(oneSecond);
    } else {
      synDeviceGetInfo(*deviceId, &deviceInfo);
      deviceType = deviceInfo.deviceType;
    }
  }

  return status == synSuccess;
}

synStatus hbmAlloc(synDeviceId deviceId,
                   uint64_t size,
                   uint64_t* addr,
                   std::string name) {
  static std::map<std::string, uint64_t> dramMap;
  if (dramMap.find(name) != dramMap.end()) {
    *addr = dramMap[name];
    return synSuccess;
  } else {
    synStatus status = synDeviceMalloc(deviceId, size, 0, 0, addr);
    dramMap[name] = *addr;
    dramMap[name + "_wu"] = *addr;
    dramMap[name + "_wu_out"] = *addr;
    return status;
  }
}

synStatus hbmFree(synDeviceId deviceId, uint64_t addr, const char* name) {
  return synDeviceFree(deviceId, addr, 0);
}

synStatus hostAlloc(synDeviceId deviceId,
                    uint64_t size,
                    uint64_t* addr,
                    std::string name) {}

synStatus hostFree(synDeviceId deviceId, uint64_t addr, const char* name) {}

synStatus createStream(synDeviceId deviceId, C_Stream* stream) {
  const auto& hpustream = streamMap.find(deviceId);
  if (hpustream != streamMap.end()) {
    *stream = hpustream->second;
    LOG(INFO) << "find stream from record and return directly";
    return synSuccess;
  }

  synStreamHandle memcpyStreamDevToHost;
  synStreamHandle memcpyStreamHostToDev;
  synStreamHandle computeStream;

  synStatus status = synStreamCreateGeneric(
      &memcpyStreamDevToHost, deviceId, 0);
  if (status != synSuccess) return status;

  status = synStreamCreateGeneric(
      &memcpyStreamHostToDev, deviceId, 0);
  if (status != synSuccess) return status;

  status = synStreamCreateGeneric(&computeStream, deviceId, 0);
  if (status != synSuccess) return status;

  *stream = new C_Stream_st(
      {deviceId, memcpyStreamDevToHost, memcpyStreamHostToDev, computeStream});

  streamMap[deviceId] = *stream;
  return status;
}

synStatus destroyStream(synDeviceId deviceId, C_Stream stream) {

  synStatus status = synStreamDestroy(stream->memcpyStreamDevToHost);
  if (status != synSuccess) return status;
  status = synStreamDestroy(stream->memcpyStreamHostToDev);
  if (status != synSuccess) return status;
  status = synStreamDestroy(stream->computeStream);
  if (status != synSuccess) return status;

  delete stream;
  
  const auto& hpustream = streamMap.find(deviceId);
  if (hpustream != streamMap.end()) {
    if (hpustream->second == stream)
      streamMap.erase(hpustream);
  }
  return status;
}

synStatus createEvent(synDeviceId deviceId, C_Event* event) {
  synEventHandle eventHandle;
  synStatus status = synEventCreate(&eventHandle, deviceId, 0);
  if (status != synSuccess) return status;
  *event = new C_Event_st({eventHandle});

  return status;
}

synStatus recordEvent(synDeviceId deviceId, C_Stream stream, C_Event event) {
  // TODO: identify which event for which stream, e.g. d2h, h2d or compute
  synStatus status = synEventRecord(event->eventHandle, stream->computeStream);
  return status;
}

synStatus destroyEvent(synDeviceId deviceId, C_Event event) {
  synStatus status = synEventDestroy(event->eventHandle);

  delete event;
  return status;
}

// void setSizesInTensorInfo(const unsigned* sizes, unsigned dims,
// build_graph_tensor_info& tensor_info)
// {
//     for(int i=0; i < c_tensor_dim; i++)
//     {
//         tensor_info.sizes[i] = 1;
//     }
//     for(int i=0; i < dims; i++)
//     {
//         tensor_info.sizes[i] = sizes[i];
//     }
// }

void resetTensorSections() {
  for (const auto& mapElement : sectionMap) {
    if (mapElement.second.second) {
      synSectionDestroy(mapElement.second.first);
    }
  }
  sectionMap.clear();
}