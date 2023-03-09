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
// clang-format off
#pragma once

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <thread>
#include <algorithm>
#include <utility>
#include <vector>
#include <CL/sycl.hpp>
#include "glog/logging.h"
#include "oneapi/dnnl/dnnl_sycl.hpp"
// clang-format on
namespace config {
enum SUBSYS_LEVELS { vError = 16 };
template <int v>
struct DeviceConfig {
  size_t chunk_size;
  size_t plugin_verbose;

  template <class T>
  T getEnvValue(const char* name, T defaultValue) {
    T ret = defaultValue;

    auto p = std::getenv(name);

    if (p) {
      std::stringstream ss;
      ss << p;
      ss >> ret;
      if (ss.fail()) {
        throw std::runtime_error("getEnvValue(): Can't convert type");
      }
    }

    return ret;
  }

  DeviceConfig() : chunk_size{4}, plugin_verbose{config::vError} {
    chunk_size = getEnvValue("PLUGIN_CHUNK_SIZE", chunk_size);
    plugin_verbose = getEnvValue("PLUGIN_VERBOSE", plugin_verbose);
    if (plugin_verbose) {
      plugin_verbose |= config::vError;
    }
  }
};

}  // namespace config

using DeviceConfig = config::DeviceConfig<0>;
using DeviceConfigPtr = std::unique_ptr<DeviceConfig>;
extern DeviceConfigPtr devconf;
extern std::mutex mx;
extern std::recursive_mutex rmux;

inline void InitializeDevConf() {
  if (!devconf) {
    std::lock_guard<decltype(mx)> l(mx);
    if (!devconf) {
      devconf = std::make_unique<DeviceConfig>();
    }
  }
}

template <class T>
const T* shortPath(const T* p) {
  const char* r = p;
  while (*p) {
    if (*p == '/') r = p;
    ++p;
  }
  return r;
}

#define show_msg(title, vbit, x)                                      \
  if (devconf && devconf->plugin_verbose & vbit) {                    \
    std::lock_guard<std::recursive_mutex> l(rmux);                    \
    std::cout << "[" << title << "][" << std::hex                     \
              << std::this_thread::get_id() << std::dec << "]["       \
              << shortPath(__FILE__) << ":" << __LINE__ << "]: " << x \
              << std::endl;                                           \
  }

#define show_debug(x) \
  VLOG(5) << x;       \
  show_msg("debug", 1, x)
#define show_memory(x) \
  VLOG(4) << x;        \
  show_msg("mem", 2, x)
#define show_kernel(x) \
  VLOG(3) << x;        \
  show_msg("kernel", 4, x)
#define show_error(x) \
  VLOG(0) << x;       \
  show_msg("error", config::vError, x)
#define rise_error(x)                                                  \
  {                                                                    \
    std::stringstream ss;                                              \
    ss << "[" << shortPath(__FILE__) << ":" << __LINE__ << "] :" << x; \
    throw std::runtime_error(ss.str());                                \
  }
