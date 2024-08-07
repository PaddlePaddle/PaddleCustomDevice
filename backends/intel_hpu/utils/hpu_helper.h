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

void resetTensorSections();
