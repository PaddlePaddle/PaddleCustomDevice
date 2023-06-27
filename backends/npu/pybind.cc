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

#include <string.h>

#include <fstream>
#include <iostream>
#include <map>
#include <vector>

#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
#include "pybind11/pybind11.h"

static std::map<std::string, aclCompileOpt> NAME_2_ACL_COMPILE_OPT = {
    {"ACL_PRECISION_MODE", ACL_PRECISION_MODE},
    {"ACL_OP_SELECT_IMPL_MODE", ACL_OP_SELECT_IMPL_MODE},
};

void AclSetCompileOpt(const std::string& cfg_path = "") {
  if (cfg_path == "") {
    std::cout << "AclSetCompileOpt" << std::endl;
    return;
  }
  // read config file
  std::ifstream cfg_file(cfg_path);
  if (!cfg_file.is_open()) {
    return;
  }
  std::string line;
  while (getline(cfg_file, line)) {
    if (line.empty()) {
      continue;
    }
    const char* delim = " ";
    std::vector<std::string> res;
    std::string strs = line + delim;
    size_t pos = strs.find(delim);
    while (pos != strs.npos) {
      std::string temp = strs.substr(0, pos);
      res.push_back(temp);
      strs = strs.substr(pos + 1, strs.size());
      pos = strs.find(delim);
    }
    aclError ret;
    ret = aclSetCompileopt(ACL_PRECISION_MODE, "force_fp32");
    // ACL_CHECK(ret);
  }
}

PYBIND11_MODULE(test_dyh, m) {
  m.def("set_compile_config",
        [](std::string& cfg_file) { AclSetCompileOpt(cfg_file); });
}
