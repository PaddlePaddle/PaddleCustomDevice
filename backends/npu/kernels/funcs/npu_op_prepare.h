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

#pragma once

#include "kernels/funcs/format_utils.h"

class OpPreparation {
 public:
  static int64_t PrepareTensorWithFormat(phi::DenseTensor& tensor, aclFormat format);

  // static void PrepareTensorWithTensor(const phi::DenseTensor& src, phi::DenseTensor& dst);

  static std::string DebugString(const phi::DenseTensor& tensor) {
    std::stringstream ss;
    if (tensor.initialized()) {
      ss << ": format: " << tensor.layout() << ", dims: [" << tensor.dims() << "]" 
        << ", capacity: <" << tensor.capacity() << ">, "
        << ", offset: <" <<  tensor.meta().offset << ">, ";
    } else {
      ss << ": format: " << tensor.layout() << ", dims: [" << tensor.dims() << "]" 
        << ", offset: <" <<  tensor.meta().offset << ">, ";
    }

    // LOG(INFO) << "storage_properties_initialized = " << tensor.storage_properties_initialized();
    if (!tensor.storage_properties_initialized()) {
      ss << " ==> storage properties not initialized.";
      return ss.str();
    }
    auto npu_properties = tensor.storage_properties<phi::NPUStorageProperties>();
    ss << ", storage_format: " << npu_properties.storage_format
       << ", storage_dims: [" << npu_properties.storage_dims << "]";
    return ss.str();
  }
};
