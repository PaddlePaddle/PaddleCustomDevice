/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <string>

#include "paddle/phi/core/dense_tensor.h"

namespace backend {

struct GcuNode {
  explicit GcuNode(const phi::DenseTensor& tensor) {
    dims = tensor.dims();
    dtype = tensor.dtype();
    layout = tensor.layout();
  }

  GcuNode(phi::DDim in_dims,
          phi::DataType in_dtype,
          phi::DataLayout in_layout = phi::DataLayout::NCHW) {
    dims = in_dims;
    dtype = in_dtype;
    layout = in_layout;
  }

  GcuNode() = default;

  std::ostream& operator<<(std::ostream& out) {
    out << "dims: " << dims.to_str()
        << " dtype: " << phi::DataTypeToString(dtype)
        << " layout: " << common::DataLayoutToString(layout);
    return out;
  }

  std::string to_str() {
    std::ostringstream os;
    os << "dims: " << dims.to_str()
       << " dtype: " << phi::DataTypeToString(dtype)
       << " layout: " << common::DataLayoutToString(layout);
    return os.str();
  }

  phi::DDim dims;
  phi::DataType dtype{phi::DataType::UNDEFINED};
  phi::DataLayout layout{phi::DataLayout::NCHW};
};

inline bool operator==(const GcuNode& lhs, const GcuNode& rhs) {
  if (lhs.dims != rhs.dims || lhs.dtype != rhs.dtype ||
      lhs.layout != rhs.layout) {
    return false;
  }
  return true;
}
inline bool operator!=(const GcuNode& lhs, const GcuNode& rhs) {
  return !operator==(lhs, rhs);
}

}  // namespace backend
