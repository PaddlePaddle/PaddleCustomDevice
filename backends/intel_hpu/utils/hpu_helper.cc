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
  return synHostMap(deviceId, size, buffer);
}

synStatus HostUnmap(const synDeviceId deviceId, const void* buffer) {
  return synHostUnmap(deviceId, buffer);
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