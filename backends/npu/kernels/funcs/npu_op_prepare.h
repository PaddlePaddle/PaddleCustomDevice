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

/**
 * Custom Device NPU related FLAG
 * Name: FLAGS_npu_storage_format
 * Since Version: 2.5.0
 * Value Range: bool, default=false
 * Example:
 * Note: Enable NPU Storage Format for Ascend910 performance improvement.
 */
DECLARE_bool(npu_storage_format);

namespace custom_kernel {

inline std::string DebugNPUTensor(const phi::DenseTensor& tensor) {
  std::stringstream ss;
  if (tensor.initialized()) {
    ss << ": dtype: " << tensor.dtype() << ", format: " << tensor.layout()
       << ", dims: [" << tensor.dims() << "]"
       << ", capacity: <" << tensor.capacity() << ">, ";
  } else {
    ss << ": dtype: " << tensor.dtype() << ", format: " << tensor.layout()
       << ", dims: [" << tensor.dims() << "]";
  }

  if (!tensor.storage_properties_initialized()) {
    ss << " ==> storage properties not initialized.";
    return ss.str();
  }
  auto npu_properties = tensor.storage_properties<phi::NPUStorageProperties>();
  ss << ", storage_format: " << npu_properties.storage_format
     << ", storage_dims: [" << npu_properties.storage_dims << "]";
  return ss.str();
}

inline int64_t PrepareTensorWithFormat(phi::DenseTensor* tensor,
                                       aclFormat format) {
  PADDLE_ENFORCE(
      tensor->valid(),
      phi::errors::InvalidArgument(
          "The input tensor of PrepareTensorWithFormat must be valid."));

  VLOG(4) << "Start PrepareTensorWithFormat with format: "
          << static_cast<int>(format) << ", tensor" << DebugNPUTensor(*tensor);

  if (tensor->storage_properties_initialized()) {
    auto npu_properties =
        tensor->storage_properties<phi::NPUStorageProperties>();
    int64_t storage_format = npu_properties.storage_format;
    if (storage_format == static_cast<int>(format)) {
      return phi::product(npu_properties.storage_dims);
    }
  }

  auto npu_properties = std::make_unique<phi::NPUStorageProperties>();
  FormatShape origin_shape = phi::vectorize<int64_t>(tensor->dims());
  FormatShape storage_shape =
      FormatHelper::GetStorageShape(format, origin_shape);
  npu_properties->storage_format = static_cast<int64_t>(format);
  npu_properties->storage_dims = phi::make_ddim(storage_shape);

  // get requested_size before move unique ptr
  int64_t requested_size = phi::product(npu_properties->storage_dims);
  tensor->set_storage_properties(std::move(npu_properties));

  VLOG(4) << "Finish PrepareTensorWithFormat with format: "
          << static_cast<int>(format)
          << ", tensor: " << DebugNPUTensor(*tensor);

  return requested_size;
}

template <typename T, typename Context>
inline void AllocNPUTensor(const Context& dev_ctx,
                           const aclFormat format,
                           phi::DenseTensor* tensor) {
  auto requested_size = PrepareTensorWithFormat(tensor, format);
  dev_ctx.template Alloc<T>(tensor,
                            requested_size * phi::SizeOf(tensor->dtype()));
}

}  // namespace custom_kernel
