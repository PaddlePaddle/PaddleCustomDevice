#pragma once

#include <map>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

#include "synapse_api.h"
#include "synapse_common_types.h"
#include "utils/hpu_utils.h"

synStatus HostMap(const synDeviceId deviceId,
                  const uint64_t size,
                  const void* buffer);
synStatus HostUnmap(const synDeviceId deviceId, const void* buffer);
bool waitForIdleDevice(uint32_t* deviceId,
                       synDeviceType& deviceType,
                       int maxSecondsWait);
synStatus hbmAlloc(synDeviceId deviceId,
                   uint64_t size,
                   uint64_t* addr,
                   std::string name);
synStatus hbmFree(synDeviceId deviceId, uint64_t addr, const char* name);
synStatus hostAlloc(synDeviceId deviceId,
                    uint64_t size,
                    uint64_t* addr,
                    std::string name);
synStatus hostFree(synDeviceId deviceId, uint64_t addr, const char* name);

struct C_Stream_st {
  synDeviceId deviceId;
  synStreamHandle memcpyStreamDevToHost;
  synStreamHandle memcpyStreamHostToDev;
  synStreamHandle computeStream;
};

synStatus createStream(synDeviceId deviceId, C_Stream* stream);
synStatus destroyStream(synDeviceId deviceId, C_Stream stream);

struct C_Event_st {
  synEventHandle eventHandle;
};

synStatus createEvent(synDeviceId deviceId, C_Event* event);
synStatus recordEvent(synDeviceId deviceId, C_Stream stream, C_Event event);
synStatus destroyEvent(synDeviceId deviceId, C_Event event);
void resetTensorSections();
