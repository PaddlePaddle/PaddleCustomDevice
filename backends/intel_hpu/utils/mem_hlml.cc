// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "utils/mem_hlml.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "glog/logging.h"
#include "habanalabs/synapse_api.h"

int ResolveDeviceIndex(int device_index) {
  synDeviceInfoV2 device_info;
  auto status = synDeviceGetInfoV2(device_index, &device_info);
  if (status == synFail) {
    throw HlMlMemoryReporter::Error("synDeviceGetInfo", EINVAL);
  }
  return device_info.deviceIndex;
}

std::string HlMlMemoryReporter::MakePath(int device_index) {
  std::string path;
  path += HLML_SHM_DEVICE_NAME_PREFIX;
  path += std::to_string(device_index);
  return path;
}

HlMlMemoryReporter::HlMlMemoryReporter(int device_index) {
  m_path = MakePath(device_index);
  m_data = MmapSharedObject();
}

HlMlMemoryReporter::~HlMlMemoryReporter() {
  if (m_data != nullptr) {
    std::size_t file_size = sizeof(*m_data);
    ::munmap(m_data, file_size);
  }
  if (m_fd != -1) {
    ::close(m_fd);
  }
  ::shm_unlink(m_path.c_str());
}

void HlMlMemoryReporter::PublishMemory(std::uint64_t bytes) {
  m_data->used_mem_in_bytes = bytes;
}

void HlMlMemoryReporter::PublishTimestamp(std::uint64_t timestamp) {
  m_data->timestamp = timestamp;
}

int HlMlMemoryReporter::OpenSharedObject() {
  int flags = O_CREAT | O_TRUNC | O_RDWR;

  int fd = shm_open(m_path.c_str(), flags, 0666);
  if (fd == -1) {
    throw Error("shm_open", errno);
  }

  return fd;
}

hlml_shm_data* HlMlMemoryReporter::MmapSharedObject() {
  m_fd = OpenSharedObject();

  try {
    auto data = PrepareSharedObject(m_fd);
    return data;
  } catch (const Error&) {
    // If something went wrong just clean up resources and continue
    // exceptional flow.
    ::close(m_fd);
    m_fd = -1;
    ::shm_unlink(m_path.c_str());
    throw;
  }
}

hlml_shm_data* HlMlMemoryReporter::PrepareSharedObject(int fd) {
  std::size_t file_size = sizeof(*m_data);

  int err = ::ftruncate(fd, file_size);
  if (err == -1) {
    throw Error("ftruncate", errno);
  }

  void* ptr = ::mmap(0, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (ptr == reinterpret_cast<void*> MAP_FAILED) {
    throw Error("mmap", errno);
  }

  VLOG(6) << "Allocated hlml shared memory " << ptr;
  auto* object = reinterpret_cast<hlml_shm_data*>(ptr);
  object->version = HLML_SHM_VERSION;
  return object;
}

std::string HlMlMemoryReporter::GetPath() const { return m_path; }

const char* FillErrorMessage(char* buffer, const char* operation, int error) {
  snprintf(buffer,
           HlMlMemoryReporter::Error::MAXLEN - 1,
           "mem_hlml failed: %s: errno %i: %s",
           operation,
           error,
           strerror(error));
  buffer[HlMlMemoryReporter::Error::MAXLEN - 1] = 0;
  return buffer;
}

HlMlMemoryReporter::Error::Error(const char* operation, int error)
    : runtime_error(FillErrorMessage(buffer, operation, error)) {}

HlMlMemoryUpdater::HlMlMemoryUpdater(
    std::shared_ptr<HlMlMemoryReporter> reporter,
    std::function<std::uint64_t()> get_used_memory)
    : m_reporter(reporter),
      m_get_used_memory(get_used_memory),
      m_thread([&] { thread_main(); }) {}

HlMlMemoryUpdater::~HlMlMemoryUpdater() {
  stop();
  m_thread.join();
}

void HlMlMemoryUpdater::stop() { m_quit.store(true); }

void HlMlMemoryUpdater::thread_main() {
  while (!m_quit.load()) {
    std::uint64_t usage = m_get_used_memory();
    std::uint64_t timestamp = ::time(NULL);
    m_reporter->PublishMemory(usage);
    m_reporter->PublishTimestamp(timestamp);
    synUpdateMemoryConsumption(usage, timestamp);
    sleep(INTERVAL);
  }
}
