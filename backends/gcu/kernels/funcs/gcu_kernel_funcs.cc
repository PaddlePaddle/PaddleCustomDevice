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

#include "kernels/funcs/gcu_kernel_funcs.h"

namespace custom_kernel {

constexpr char kGradVarSuffix[] = "@GRAD";
constexpr size_t kGradVarSuffixSize = 5U;

std::string GradVarName(const std::string& var_name) {
  std::string result;
  result.reserve(var_name.size() + kGradVarSuffixSize);
  result += var_name;
  result += kGradVarSuffix;
  return result;
}

std::vector<int> GetIntList(const phi::IntArray& int_array) {
  std::vector<int64_t> data = int_array.GetData();
  std::vector<int> data_ret =
      std::move(std::vector<int>(data.begin(), data.end()));
  return data_ret;
}

}  // namespace custom_kernel
