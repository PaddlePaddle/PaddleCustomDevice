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

#include "common/utils.h"

#include <string>
#include <vector>

#include "backend/utils/utils.h"
#include "runtime/runtime.h"

namespace custom_kernel {

std::string HLIRTensorToString(const std::vector<hlir::Tensor*>& tensors,
                               bool is_inputs) {
  const auto numel = tensors.size();
  std::stringstream ss;
  hlir::Tensor* tensor = nullptr;
  const std::string name = is_inputs ? "input_" : "output_";
  for (size_t i = 0; i < numel; i++) {
    tensor = tensors[i];
    ss << "tensor " << name << i << ": {\n";
    ss << "  mem_handle: {\n" << DumpHbm(tensor->mem_handle);
    ss << "  }\n";
    ss << "  bytes_size: " << tensor->bytes_size << "\n";
    ss << "  element_type: " << tensor->element_type << "\n";
    ss << "  dimensions: " << backend::VectorToString(tensor->dimensions)
       << "\n";
    ss << "  strides: " << backend::VectorToString(tensor->strides) << "\n";
    ss << "  layouts: " << backend::VectorToString(tensor->layout) << "\n";
    ss << "}\n";
  }
  return ss.str();
}

}  // namespace custom_kernel
