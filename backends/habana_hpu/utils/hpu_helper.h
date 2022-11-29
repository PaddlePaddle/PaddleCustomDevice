#pragma once

#include <set>
#include <map>
#include <vector>
#include <unordered_map>
#include <mutex>

#include "synapse_api.h"
#include "synapse_common_types.h"

typedef std::pair<synSectionHandle, bool> sectionWithFirstIndication;
static std::unordered_map<std::string, sectionWithFirstIndication> sectionMap;
static std::mutex mut_;

synStatus  HostMap( const synDeviceId          deviceId,
                                   const uint64_t             size,
                                   const void*                buffer )
{
    // std::lock_guard<std::mutex> lock_guard(mut_);
    return synHostMap(deviceId, size, buffer);
}

synStatus HostUnmap( const synDeviceId    deviceId,
                                     const void*          buffer )
{
    // std::lock_guard<std::mutex> lock_guard(mut_);
    synHostUnmap(deviceId, buffer);
}

bool waitForIdleDevice(uint32_t* deviceId, synDeviceType& deviceType, int maxSecondsWait)
{
    int oneSecond = 1;
    synDeviceInfo deviceInfo;
    auto start = std::chrono::high_resolution_clock::now();
    auto maxWaitTime = start + std::chrono::seconds(maxSecondsWait);
    synStatus status      = synFail;
    char* pci_address = getenv("PCI_ADDRESS");
    while (status != synSuccess && std::chrono::high_resolution_clock::now() < maxWaitTime)
    {
        status = synDeviceAcquire(deviceId, pci_address);
        if (status != synSuccess)
        {
            sleep(oneSecond);
        }
        else
        {
            synDeviceGetInfo(*deviceId, &deviceInfo);
            deviceType = deviceInfo.deviceType;
        }
    }

    return status == synSuccess;
}

synStatus hbmAlloc(synDeviceId deviceId, uint64_t size, uint64_t* addr, std::string name)
{
    static std::map<std::string, uint64_t> dramMap;
    if (dramMap.find(name) != dramMap.end())
    {
        *addr = dramMap[name];
        return synSuccess;
    }
    else
    {
        synStatus status = synDeviceMalloc(deviceId, size, 0, 0, addr);
        dramMap[name] = *addr;
        dramMap[name + "_wu"] = *addr;
        dramMap[name + "_wu_out"] = *addr;
        return status;
    }
}

synStatus hbmFree(synDeviceId deviceId, uint64_t addr, const char* name)
{
    return synDeviceFree(deviceId, addr, 0);
}

struct C_Stream_st {
  synDeviceId deviceId;
  synStreamHandle memcpyStreamDevToHost;
  synStreamHandle memcpyStreamHostToDev;
  synStreamHandle computeStream;
};

synStatus createStream(synDeviceId deviceId, C_Stream *stream)
{
  synStreamHandle memcpyStreamDevToHost;
  synStreamHandle memcpyStreamHostToDev;
  synStreamHandle computeStream;

  synStatus status = synStreamCreate(&memcpyStreamDevToHost, deviceId, STREAM_TYPE_COPY_DEVICE_TO_HOST, 0);
  if (status != synSuccess) return status;

  status = synStreamCreate(&memcpyStreamHostToDev, deviceId, STREAM_TYPE_COPY_HOST_TO_DEVICE, 0);
  if (status != synSuccess) return status;

  status = synStreamCreate(&computeStream, deviceId, STREAM_TYPE_COMPUTE, 0);
  if (status != synSuccess) return status;

  *stream = new C_Stream_st({deviceId, memcpyStreamDevToHost, memcpyStreamHostToDev, computeStream});
  return status;
}

synStatus destroyStream(synDeviceId deviceId, C_Stream stream)
{
  synStatus status = synStreamDestroy(stream->memcpyStreamDevToHost);
  if (status != synSuccess) return status;
  status = synStreamDestroy(stream->memcpyStreamHostToDev);
  if (status != synSuccess) return status;
  status = synStreamDestroy(stream->computeStream);
  if (status != synSuccess) return status;

  delete stream;
  return status;
}

struct C_Event_st {
  synEventHandle  eventHandle;
};

synStatus createEvent(synDeviceId deviceId, C_Event *event)
{
  synEventHandle  eventHandle;
  synStatus status = synEventCreate(&eventHandle, deviceId, 0);
  if (status != synSuccess) return status;
  *event = new C_Event_st({eventHandle});

  return status;
}

synStatus recordEvent(synDeviceId deviceId, C_Stream stream, C_Event event)
{
  //TODO: identify which event for which stream, e.g. d2h, h2d or compute
  synStatus status = synEventRecord(event->eventHandle, stream->computeStream);
  return status;
}

synStatus destroyEvent(synDeviceId deviceId, C_Event event)
{
  synStatus status = synEventDestroy(event->eventHandle);

  delete event;
  return status;
}

synTensor createTensor(unsigned dims, synDataType data_type, const unsigned* tensor_size,
                       bool is_presist, const char* name)

{
    synStatus             status;
    synTensorDescriptor   desc{};
    // input
    desc.m_dataType     = data_type;
    desc.m_dims         = dims;
    desc.m_name         = name;
    memset(desc.m_strides, 0, sizeof(desc.m_strides));

    for (unsigned i = 0; i < dims; ++i)
    {
        desc.m_sizes[i] = tensor_size[dims - 1 - i];
    }
    synSectionHandle sectionHandle = nullptr;
    if (is_presist)
    {
        const auto& section = sectionMap.find(name);
        if (section == sectionMap.end())
        {
            status = synSectionCreate(&sectionHandle, 0, nullptr);
            assert(status == synSuccess && "Create section failed!");
            sectionMap[name + std::string()] = std::make_pair(sectionHandle, true);
            sectionMap[name + std::string("_wu_out")] = std::make_pair(sectionHandle, false);
            sectionMap[name + std::string("_out")] = std::make_pair(sectionHandle, false);
        }
        else
        {
            sectionHandle = section->second.first;
        }
    }
    synTensor     tensor;
    status = synTensorCreate(&tensor, &desc, sectionHandle, 0);
    assert(status == synSuccess && "Create tensor failed!");
    return tensor;
}

// void setSizesInTensorInfo(const unsigned* sizes, unsigned dims, build_graph_tensor_info& tensor_info)
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

void resetTensorSections()
{
    for (const auto& mapElement : sectionMap)
    {
        if (mapElement.second.second)
        {
            synSectionDestroy(mapElement.second.first);
        }
    }
    sectionMap.clear();
}
