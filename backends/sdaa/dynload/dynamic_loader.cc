// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "dynload/dynamic_loader.h"

#include <cstdlib>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/flags.h"

namespace custom_dynload {

static inline std::vector<std::string> split(
    const std::string& str, const std::string separator = " ") {
  std::vector<std::string> str_list;
  std::string::size_type firstPos;
  firstPos = str.find_first_not_of(separator, 0);
  std::string::size_type lastPos;
  lastPos = str.find_first_of(separator, firstPos);
  while (std::string::npos != firstPos && std::string::npos != lastPos) {
    str_list.push_back(str.substr(firstPos, lastPos - firstPos));
    firstPos = str.find_first_not_of(separator, lastPos);
    lastPos = str.find_first_of(separator, firstPos);
  }
  if (std::string::npos == lastPos) {
    str_list.push_back(str.substr(firstPos, lastPos - firstPos));
  }
  return str_list;
}

static inline void* GetDsoHandleFromDefaultPath(const std::string& dso_path,
                                                int dynload_flags) {
  // default search from LD_LIBRARY_PATH/DYLD_LIBRARY_PATH
  // and /usr/local/lib path
  void* dso_handle = dlopen(dso_path.c_str(), dynload_flags);
  VLOG(3) << "Try to find library: " << dso_path
          << " from default system path.";

  return dso_handle;
}

/*
 * We define one priority for dynamic library search:
 * Search the stheystem default path
 */

static inline void* GetDsoHandleFromSearchPath(const std::string& dso_name,
                                               bool throw_on_error = true) {
  int dynload_flags = RTLD_LAZY | RTLD_LOCAL;
  std::vector<std::string> dso_names = split(dso_name, ";");
  void* dso_handle = nullptr;
  for (auto dso : dso_names) {
    // search in system default path
    dso_handle = GetDsoHandleFromDefaultPath(dso, dynload_flags);
    if (nullptr != dso_handle) break;
  }

  // [If Failed for All dso_names] logging or throw error info
  if (nullptr == dso_handle) {
    auto error_msg =
        "The dynamic library (%s) that Paddle depends on is not "
        "configured correctly. (error code is %s)\n"
        "  Suggestions:\n"
        "  1. Check if the dynamic library is installed correctly "
        "and its version is matched with paddle-sdaa you installed.\n"
        "  2. Configure dynamic library environment variables as "
        "follows:\n"
        "  - Linux: set LD_LIBRARY_PATH by `export LD_LIBRARY_PATH=...`\n";
    auto errorno = dlerror();
    if (throw_on_error) {
      // NOTE: Special error report case, no need to change its format
      PADDLE_THROW(
          phi::errors::PreconditionNotMet(error_msg, dso_name, errorno));
    } else {
      LOG(WARNING) << paddle::string::Sprintf(error_msg, dso_name, errorno);
    }
  }

  return dso_handle;
}

void* GetSDPTIDsoHandle() {
  return GetDsoHandleFromSearchPath("libsdpti.so;libsdpti.so.1", false);
}
}  // namespace custom_dynload
