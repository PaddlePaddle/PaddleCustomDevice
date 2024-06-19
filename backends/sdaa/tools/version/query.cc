// BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
// reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
// OF SUCH DAMAGE.

#include "tools/version/query.h"

#include <dlfcn.h>

#include <deque>
#include <mutex>
#include <regex>
#include <tuple>
#include <utility>

#include "sdpti.h"  //NOLINT

using Table = tabulate::Table;

namespace {
std::string ExtractVariableValue(const std::string &filename,
                                 const std::string &variable_name) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    // use version path while building
    std::ifstream build_version_path(PADDLE_BUILD_ENV_VERSION_PATH);
    file.swap(build_version_path);
  }

  if (!file.is_open()) {
    return "";
  }

  std::string line;
  std::regex variableRegex(variable_name + "\\s*=\\s*'(.*)'");
  while (std::getline(file, line)) {
    std::smatch match;
    if (std::regex_search(line, match, variableRegex)) {
      if (match.size() == 2) {
        return match[1].str();
      }
    }
  }
  return "";
}

std::string GetPaddleVersionPath() {
  std::string paddle_path;
  const char *custom_kernel_root_p = std::getenv("CUSTOM_DEVICE_ROOT");
  if (custom_kernel_root_p != nullptr) {
    paddle_path =
        std::string(custom_kernel_root_p) + "/../paddle/version/__init__.py";
  }
  return std::move(paddle_path);
}

std::string GetVersionStr(const std::vector<int> &version_items) {
  std::deque<std::string> sdaa_version_queue;
  auto add = [&sdaa_version_queue](std::string &&str) {
    sdaa_version_queue.emplace_front(std::move(str));
  };
  // [major, minor, patch, type, type_no]
  bool is_detailed = version_items.size() == 5;
  if (is_detailed) {
    int type_no = version_items[4];
    int typeVal = version_items[3];
    // release version doesn't have type_no
    if (typeVal) add(TypeMapping[--typeVal] + std::to_string(type_no));
  }

  if (version_items.size() >= 3) {
    int patch = version_items[2];
    add(std::to_string(patch));

    int minor = version_items[1];
    add(std::to_string(minor) + ".");

    int major = version_items[0];
    add(std::to_string(major) + ".");
  }
  std::string result;
  for (const auto &str : sdaa_version_queue) {
    result += str;
  }
  return result;
}

std::vector<int> GetVersionItems(int version, const std::vector<int> &formula) {
  size_t size = formula.size();
  std::vector<int> result(size);
  for (int i = size - 1; i >= 0; i--) {
    result[i] = version % formula[i];
    version /= formula[i];
  }
  return result;
}
}  // namespace

int GetVersionNum(const std::string &version) {
  std::regex version_regex(R"((\d+)\.(\d+)\.(\d+)([a-z](\d*))?)");

  std::smatch matches;

  if (std::regex_match(version, matches, version_regex)) {
    int major = std::stoi(matches[1]);
    int minor = std::stoi(matches[2]);
    int patch = std::stoi(matches[3]);
    auto type = matches[4].str().empty() ? std::string("") : matches[4].str();
    int type_no = matches[5].str().empty() ? 0 : std::stoi(matches[5]);

    return 100000000 * major + 1000000 * minor + 10000 * patch +
           100 * TYPE_VALUE(type.c_str()) + type_no;
  }
  return 0;
}

Version GetSdaaRuntimeVersion() {
  int sdaa_rt_version = 0;
  sdaaRuntimeGetVersion(&sdaa_rt_version);
  auto items = GetVersionItems(sdaa_rt_version, SDAA_DETAILED_FORMULA);
  return {SDAA_RUNTIME_NAME, GetVersionStr(items), sdaa_rt_version};
}

Version GetSdaaDriverVersion() {
  int sdaa_driver_version = 0;
  sdaaDriverGetVersion(&sdaa_driver_version);
  auto items = GetVersionItems(sdaa_driver_version, SDAA_DETAILED_FORMULA);
  return {SDAA_DRIVER_NAME, GetVersionStr(items), sdaa_driver_version};
}

Version GetTecoDNNVersion() {
  int teco_dnn_version = tecodnnGetVersion();

  auto items = GetVersionItems(teco_dnn_version, DNN_BLAS_DETAILED_FORMULA);
  return {TECO_DNN_NAME, GetVersionStr(items), teco_dnn_version};
}

Version GetTecoBLASVersion() {
  int teco_blas_version = 0;
  tblasHandle_t handle;
  tecoblasCreate(&handle);
  tecoblasGetVersion(handle, &teco_blas_version);
  tecoblasDestroy(handle);

  auto items = GetVersionItems(teco_blas_version, DNN_BLAS_DETAILED_FORMULA);
  return {TECO_BLAS_NAME, GetVersionStr(items), teco_blas_version};
}

Version GetTecoCustomVersion() {
  void *handle = dlopen("libtecodnn_ext.so", RTLD_LAZY);
  if (!handle) {
    return {TECO_CUSTOM_NAME, ""};
  }

  // Check if a symbol exists, if no, record ""
  void *symbol = dlsym(handle, "tecodnnExtGetVersion");
  if (!symbol) {
    dlclose(handle);
    return {TECO_CUSTOM_NAME, ""};
  }

  int teco_custom_version = reinterpret_cast<size_t (*)()>(symbol)();

  // Symbol found
  dlclose(handle);

  auto items = GetVersionItems(teco_custom_version, DNN_BLAS_DETAILED_FORMULA);
  return {TECO_CUSTOM_NAME, GetVersionStr(items), teco_custom_version};
}

Version GetTCCLVersion() {
  int tccl_version = 0;
  tcclGetVersion(&tccl_version);
  auto items = GetVersionItems(tccl_version, TCCL_DETAILED_FORMULA);
  return {TCCL_NAME, GetVersionStr(items), tccl_version};
}

Version GetSDptiVersion() {
  void *handle = dlopen("libsdpti.so", RTLD_LAZY);
  if (!handle) {
    return {SDPTI_NAME, ""};
  }

  // Check if a symbol exists, if no, record ""
  void *symbol = dlsym(handle, "sdptiGetVersion");
  if (!symbol) {
    dlclose(handle);
    return {SDPTI_NAME, ""};
  }

  uint32_t sdpti_version = 0;
  reinterpret_cast<SDptiResult (*)(uint32_t *)>(symbol)(&sdpti_version);

  // Symbol found
  dlclose(handle);

  auto items =
      GetVersionItems(static_cast<int>(sdpti_version), SDPTI_DETAILED_FORMULA);
  return {SDPTI_NAME, GetVersionStr(items), static_cast<int>(sdpti_version)};
}

Version GetPaddlePaddleSDAACommit() {
#ifdef GIT_COMMIT_ID
  return {PLUGIN_COMMIT, GIT_COMMIT_ID};
#else
  return {PLUGIN_COMMIT, ""};
#endif
}

Version GetPaddlePaddleCommit() {
#ifdef PADDLE_COMMIT_ID
  return {PADDLE_COMMIT, PADDLE_COMMIT_ID};
#else
  return {PADDLE_COMMIT, ""};
#endif
}

Version GetPaddlePaddleVersion() {
#ifdef PADDLE_FULL_VERSION
  return {PADDLE_VERSION, PADDLE_FULL_VERSION};
#else
  return {PADDLE_VERSION, ""};
#endif
}

Version GetPaddleCurrentCommit() {
  auto paddle_version_path = GetPaddleVersionPath();
  auto paddle_commit = ExtractVariableValue(paddle_version_path, "commit");
  return {PADDLE_CURRENT_COMMIT, paddle_commit.substr(0, 7)};
}

Version GetPaddleCurrentVersion() {
  auto paddle_version_path = GetPaddleVersionPath();
  auto paddle_version =
      ExtractVariableValue(paddle_version_path, "full_version");
  return {
      PADDLE_CURRENT_VERSION, paddle_version, GetVersionNum(paddle_version)};
}

std::vector<Version> GetAllDepVersions() {
  using VersionFuncVec = std::vector<Version (*)()>;
  VersionFuncVec version_func{GetPaddleCurrentVersion,
                              GetPaddleCurrentCommit,
                              GetSdaaRuntimeVersion,
                              GetSdaaDriverVersion,
                              GetTecoDNNVersion,
                              GetTecoBLASVersion,
                              GetTCCLVersion,
                              GetTecoCustomVersion,
                              GetSDptiVersion};
#ifdef GIT_COMMIT_ID
  version_func.push_back(GetPaddlePaddleSDAACommit);
#endif
  std::vector<Version> result;
  for (const auto &f : version_func) {
    auto single_version = f();
    if (single_version.version.empty()) {
      continue;
    }
    result.emplace_back(std::move(single_version));
  }
  return result;
}

Table GetVersionTable() {
  Table deps;
  Table::Row_t first_row{""};
  Table::Row_t second_row{"Version"};
  auto all_versions = GetAllDepVersions();
  for (const auto &v : all_versions) {
    first_row.push_back(v.name);
    second_row.push_back(v.version);
  }
  deps.add_row(first_row);
  deps.add_row(second_row);
  return deps;
}

VersionCheckType CheckVersions() {
  // version check
  const char *custom_kernel_root_p = std::getenv("CUSTOM_DEVICE_ROOT");

  Table diff_table;
  Table::Row_t first_row{
      "Dependence", "Current Version", "Minimum Supported Version"};
  diff_table.add_row(first_row);

  auto compare_func = [](int cur_version, int min_version) -> bool {
    std::vector<std::tuple<std::string, int, int>> vec_version;
    // major version
    vec_version.emplace_back(std::make_tuple(
        "major", cur_version / 100000000, min_version / 100000000));

    // minor version
    cur_version %= 100000000;
    min_version %= 100000000;
    vec_version.emplace_back(
        std::make_tuple("minor", cur_version / 1000000, min_version / 1000000));

    // patch version
    cur_version %= 1000000;
    min_version %= 1000000;
    vec_version.emplace_back(
        std::make_tuple("patch", cur_version / 10000, min_version / 10000));

    // type version
    cur_version %= 10000;
    min_version %= 10000;
    vec_version.emplace_back(
        std::make_tuple("type", cur_version / 100, min_version / 100));

    // type no version
    cur_version %= 100;
    min_version %= 100;
    vec_version.emplace_back(
        std::make_tuple("type_no", cur_version, min_version));

    // compare major
    for (int i = 0; i < vec_version.size(); ++i) {
      std::string tag;
      int cur_version;
      int min_version;
      std::tie(tag, cur_version, min_version) = vec_version[i];
      if (cur_version != min_version) {
        return tag != "type" ? cur_version > min_version
                             : cur_version < min_version;
      }
    }

    return true;
  };

  auto local_version = GetAllDepVersions();

  for (const auto &i : local_version) {
    if (UNCOMPARE_KER.find(i.name) != UNCOMPARE_KER.end()) {
      continue;
    }

    // compare current version with minimum version
    const auto min_version_num = MINIMUM_SUPPORTED_VERSIONS_NUM.at(i.name);

    if (!compare_func(i.version_num, min_version_num)) {
      const auto min_version = MINIMUM_SUPPORTED_VERSIONS.at(i.name);
      Table::Row_t row;
      row.emplace_back(i.name);
      row.emplace_back(i.version);
      row.emplace_back(min_version);
      diff_table.add_row(row);
    }
  }

  if (1 != diff_table.size()) {
    // if different, print the difference table
    std::cout << "The following libraries are unsupported:\n"
              << diff_table << std::endl;

    return VersionCheckType::INCOMPATIBLE;
  }

  return VersionCheckType::CONSISTENT;
}

void PrintEnvFlags() {
  Table env_flags;
  Table::Row_t head{"Variable Name", "Default Value", "Current Value"};
  env_flags.add_row(head);

#define ADD_ENV_FLAGS_HELPER(var, default)                \
  const char *value_of_##var = #default;                  \
  if (std::getenv(#var) != nullptr) {                     \
    value_of_##var = std::getenv(#var);                   \
  }                                                       \
  Table::Row_t var_##var{#var, #default, value_of_##var}; \
  env_flags.add_row(var_##var)

  ADD_ENV_FLAGS_HELPER(CUSTOM_DEVICE_BLACK_LIST, );  // NOLINT
  ADD_ENV_FLAGS_HELPER(ENABLE_SDPTI, 1);
  ADD_ENV_FLAGS_HELPER(HIGH_PERFORMANCE_CONV, 0);
  ADD_ENV_FLAGS_HELPER(HIGH_PERFORMANCE_GEMM, 0);
  ADD_ENV_FLAGS_HELPER(RANDOM_ALIGN_NV_DEVICE, );  // NOLINT
  ADD_ENV_FLAGS_HELPER(PADDLE_XCCL_BACKEND, );     // NOLINT
  ADD_ENV_FLAGS_HELPER(HIGH_PRECISION_OP_LIST, );  // NOLINT
  ADD_ENV_FLAGS_HELPER(FLAGS_sdaa_reuse_event, true);
  ADD_ENV_FLAGS_HELPER(FLAGS_sdaa_runtime_debug, false);

#undef ADD_ENV_FLAGS_HELPER

  std::cout << env_flags << std::endl;
}

void PrintExtraInfo() { PrintEnvFlags(); }
