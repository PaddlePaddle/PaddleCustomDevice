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

#pragma once

#include <fstream>
#include <iostream>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "sdaa_runtime.h"  //NOLINT
#include "tabulate/tabulate.hpp"
#include "tccl.h"            //NOLINT
#include "tecoblas.h"        //NOLINT
#include "tecodnn.h"         //NOLINT
#include "tecodnn_custom.h"  //NOLINT
#include "tools/version/minimum_supported_version.h"

struct Version {
  Version(const std::string &dep_name,
          const std::string &dep_version,
          const int dep_version_num = 0)
      : name(dep_name), version(dep_version), version_num(dep_version_num) {}
  std::string name;
  std::string version;
  int version_num;
};

enum class VersionCheckType { CONSISTENT, COMPATIBLE, INCOMPATIBLE };

static const std::vector<std::string> TypeMapping{"rc", "b", "a"};
static const char *const SDAA_RUNTIME_NAME = "sdaa_runtime";
static const char *const SDAA_DRIVER_NAME = "sdaa_driver";
static const char *const TECO_DNN_NAME = "teco_dnn";
static const char *const TECO_BLAS_NAME = "teco_blas";
static const char *const TECO_CUSTOM_NAME = "teco_custom";
static const char *const TCCL_NAME = "teco_tccl";
static const char *const SDPTI_NAME = "sdpti";
static const char *const PLUGIN_COMMIT = "paddle_sdaa_commit";
static const char *const PADDLE_COMMIT = "compilation_paddle_commit";
static const char *const PADDLE_VERSION = "compilation_paddle_version";
static const char *const PADDLE_CURRENT_COMMIT = "paddle_commit";
static const char *const PADDLE_CURRENT_VERSION = "paddle";
static const std::unordered_set<std::string> UNCOMPARE_KER{
    PLUGIN_COMMIT, PADDLE_CURRENT_COMMIT};
static const std::unordered_map<std::string, std::string>
    MINIMUM_SUPPORTED_VERSIONS{
        {PADDLE_CURRENT_VERSION, PADDLE_MINIMUM_SUPPORTED_VERSION},
        {SDAA_RUNTIME_NAME, RUNTIME_MINIMUM_SUPPORTED_VERSION},
        {SDAA_DRIVER_NAME, DRIVER_MINIMUM_SUPPORTED_VERSION},
        {TECO_DNN_NAME, TECO_DNN_MINIMUM_SUPPORTED_VERSION},
        {TECO_BLAS_NAME, TECO_BLAS_MINIMUM_SUPPORTED_VERSION},
        {TECO_CUSTOM_NAME, TECO_CUSTOM_MINIMUM_SUPPORTED_VERSION},
        {TCCL_NAME, TCCL_MINIMUM_SUPPORTED_VERSION},
        {SDPTI_NAME, SDPTI_MINIMUM_SUPPORTED_VERSION}};
static const std::unordered_map<std::string, int>
    MINIMUM_SUPPORTED_VERSIONS_NUM{
        {PADDLE_CURRENT_VERSION, PADDLE_MINIMUM_SUPPORTED_VERSION_NUM},
        {SDAA_RUNTIME_NAME, RUNTIME_MINIMUM_SUPPORTED_VERSION_NUM},
        {SDAA_DRIVER_NAME, DRIVER_MINIMUM_SUPPORTED_VERSION_NUM},
        {TECO_DNN_NAME, TECO_DNN_MINIMUM_SUPPORTED_VERSION_NUM},
        {TECO_BLAS_NAME, TECO_BLAS_MINIMUM_SUPPORTED_VERSION_NUM},
        {TECO_CUSTOM_NAME, TECO_CUSTOM_MINIMUM_SUPPORTED_VERSION_NUM},
        {TCCL_NAME, TCCL_MINIMUM_SUPPORTED_VERSION_NUM},
        {SDPTI_NAME, SDPTI_MINIMUM_SUPPORTED_VERSION_NUM}};

static const std::vector<int> SDAA_DETAILED_FORMULA{100, 100, 100, 100, 100};

static const std::vector<int> DNN_BLAS_DETAILED_FORMULA{
    100, 100, 100, 100, 100};

static const std::vector<int> TCCL_DETAILED_FORMULA{100, 100, 100, 100, 100};

static const std::vector<int> SDPTI_DETAILED_FORMULA{100, 100, 100, 100, 100};

Version GetSdaaRuntimeVersion();

Version GetSdaaDriverVersion();

Version GetTecoDNNVersion();

Version GetTecoBLASVersion();

// Custom dnn and blas use the same version
Version GetTecoCustomVersion();

Version GetTCCLVersion();

Version GetSDptiVersion();

Version GetPaddlePaddleSDAACommit();

Version GetPaddlePaddleCommit();

Version GetPaddlePaddleVersion();

Version GetPaddleCurrentCommit();

Version GetPaddleCurrentVersion();

std::vector<Version> GetAllDepVersions();

tabulate::Table GetVersionTable();

std::unordered_map<std::string, std::string> GetAllCompileDepVersions();

VersionCheckType CheckVersions();

void PrintExtraInfo();
