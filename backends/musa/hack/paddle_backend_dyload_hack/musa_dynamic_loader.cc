/* Copyright (c) 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All
rights reserved. Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Modifications:
Copyright (c) 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All
rights reserved.

- [Modify the relevant code to adapt to musa backend] */
#include "musa_dynamic_loader.h"  // NOLINT

#include <cstdlib>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "paddle/common/flags.h"
#include "paddle/phi/common/port.h"
#include "paddle/phi/core/enforce.h"

PHI_DEFINE_string(mudnn_dir,
                  "",
                  "Specify path for loading libmudnn.so. For instance, "
                  "/usr/local/musa/lib. If empty [default], dlopen "
                  "will search mudnn from LD_LIBRARY_PATH");

PHI_DEFINE_string(musa_dir,
                  "",
                  "Specify path for loading rocm library, such as libmublas, "
                  "For instance, /usr/local/musa/lib. "
                  "If default, dlopen will search rocm from LD_LIBRARY_PATH");

PHI_DEFINE_string(mccl_dir,
                  "",
                  "Specify path for loading mccl library, such as libmccl.so. "
                  "For instance, /usr/local/musa/lib. If default, "
                  "dlopen will search rccl from LD_LIBRARY_PATH");

namespace phi {
namespace dynload {
namespace musa_dl_utils {

struct PathNode {
  PathNode() = default;
  std::string path = "";
};

static inline std::string join(const std::string& part1,
                               const std::string& part2) {
  // directory separator
  const char sep = '/';
  if (!part2.empty() && part2.front() == sep) {
    return part2;
  }
  std::string ret;
  ret.reserve(part1.size() + part2.size() + 1);
  ret = part1;
  if (!ret.empty() && ret.back() != sep) {
    ret += sep;
  }
  ret += part2;
  return ret;
}

static inline std::vector<std::string> split(
    const std::string& str, const std::string separator = " ") {
  std::vector<std::string> str_list;
  std::string::size_type firstPos = 0;
  firstPos = str.find_first_not_of(separator, 0);
  std::string::size_type lastPos = 0;
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

static inline void* GetDsoHandleFromSpecificPath(const std::string& spec_path,
                                                 const std::string& dso_name,
                                                 int dynload_flags) {
  void* dso_handle = nullptr;
  if (!spec_path.empty()) {
    // search xxx.so from custom path
    VLOG(3) << "Try to find library: " << dso_name
            << " from specific path: " << spec_path;
    std::string dso_path = join(spec_path, dso_name);
    dso_handle = dlopen(dso_path.c_str(), dynload_flags);
  }
  return dso_handle;
}

static inline void* GetDsoHandleFromDefaultPath(const std::string& dso_path,
                                                int dynload_flags) {
  // default search from LD_LIBRARY_PATH/DYLD_LIBRARY_PATH
  // and /usr/local/lib path
  void* dso_handle = dlopen(dso_path.c_str(), dynload_flags);
  VLOG(3) << "Try to find library: " << dso_path
          << " from default system path.";

// TODO(chenweihang): This path is used to search which libs?
// DYLD_LIBRARY_PATH is disabled after Mac OS 10.11 to
// bring System Integrity Projection (SIP), if dso_handle
// is null, search from default package path in Mac OS.
#if defined(__APPLE__) || defined(__OSX__)
  if (nullptr == dso_handle) {
    dso_handle =
        dlopen(join("/usr/local/cuda/lib/", dso_path).c_str(), dynload_flags);
  }
#endif

  return dso_handle;
}

static void* GetDsoHandleFromSearchPath(
    const std::string& config_path,
    const std::string& dso_name,
    bool throw_on_error = true,
    const std::vector<std::string>& extra_paths = std::vector<std::string>(),
    const std::string& warning_msg = std::string()) {
  int dynload_flags = 0;
  std::vector<std::string> dso_names = split(dso_name, ";");
  void* dso_handle = nullptr;
  for (auto const& dso : dso_names) {
    // 1. search in user config path by FLAGS
    dso_handle = GetDsoHandleFromSpecificPath(config_path, dso, dynload_flags);
    // 2. search in system default path
    if (nullptr == dso_handle) {
      dso_handle = GetDsoHandleFromDefaultPath(dso, dynload_flags);
    }
    // 3. search in extra paths
    if (nullptr == dso_handle) {
      for (auto const& path : extra_paths) {
        VLOG(3) << "extra_paths: " << path;
        dso_handle = GetDsoHandleFromSpecificPath(path, dso, dynload_flags);
      }
    }
    if (nullptr != dso_handle) break;
  }

  // 4. [If Failed for All dso_names] logging warning if exists
  if (nullptr == dso_handle && !warning_msg.empty()) {
    LOG(WARNING) << warning_msg;
  }

  // 5. [If Failed for All dso_names] logging or throw error info
  if (nullptr == dso_handle) {
    auto error_msg =
        "The third-party dynamic library (%s) that Paddle depends on is not "
        "configured correctly. (error code is %s)\n"
        "  Suggestions:\n"
        "  1. Check if the third-party dynamic library (e.g. CUDA, CUDNN) "
        "is installed correctly and its version is matched with paddlepaddle "
        "you installed.\n"
        "  2. Configure third-party dynamic library environment variables as "
        "follows:\n"
        "  - Linux: set LD_LIBRARY_PATH by `export LD_LIBRARY_PATH=...`\n"
        "  - Windows: set PATH by `set PATH=XXX;%PATH%`\n"
        "  - Mac: set  DYLD_LIBRARY_PATH by `export DYLD_LIBRARY_PATH=...` "
        "[Note: After Mac OS 10.11, using the DYLD_LIBRARY_PATH is "
        "impossible unless System Integrity Protection (SIP) is disabled.]";
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

}  // namespace musa_dl_utils

void* GetMublasDsoHandle() {
  return musa_dl_utils::GetDsoHandleFromSearchPath(FLAGS_musa_dir,
                                                   "libmublas.so");
}

void* GetMUDNNDsoHandle() {
  return musa_dl_utils::GetDsoHandleFromSearchPath(FLAGS_mudnn_dir,
                                                   "libmudnn.so");
}

void* GetMUPTIDsoHandle() {
  // TODO(someone): implement mupti load
}
void* GetMusolverDsoHandle() {
  // TODO(someone): implement muslover load
}

void* GetMurandDsoHandle() {
  return musa_dl_utils::GetDsoHandleFromSearchPath(FLAGS_musa_dir,
                                                   "libmurand.so");
}

void* GetMUFFTDsoHandle() {
  return musa_dl_utils::GetDsoHandleFromSearchPath(FLAGS_musa_dir,
                                                   "libmufft.so");
}
void* GetMusparseDsoHandle() {
  return musa_dl_utils::GetDsoHandleFromSearchPath(FLAGS_musa_dir,
                                                   "libmusparse.so");
}

void* GetMURTCDsoHandle() {
  return musa_dl_utils::GetDsoHandleFromSearchPath(
      FLAGS_musa_dir, "libmusart.so", false);
}

void* GetMUSADsoHandle() {
  return musa_dl_utils::GetDsoHandleFromSearchPath(
      FLAGS_musa_dir, "libmusa.so", false);
}

void* GetMCCLDsoHandle() {
  std::string warning_msg(
      "You may need to install 'mccl' from musa official website.");

  return musa_dl_utils::GetDsoHandleFromSearchPath(
      FLAGS_mccl_dir, "libmccl.so", true, {}, warning_msg);
}

}  // namespace dynload
}  // namespace phi
