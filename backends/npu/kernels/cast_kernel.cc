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

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/op_command.h"
#include "paddle/phi/core/tensor_meta.h"

namespace custom_kernel {

template <typename T, typename Context>
void CastKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensorMeta::DataType dtype,
                phi::DenseTensor* out) {
  if (dtype == phi::DenseTensorMeta::DataType::FLOAT32) {
    dev_ctx.template Alloc<float>(out);
  } else if (dtype == phi::DenseTensorMeta::DataType::FLOAT64) {
    dev_ctx.template Alloc<double>(out);
  } else if (dtype == phi::DenseTensorMeta::DataType::FLOAT16) {
    dev_ctx.template Alloc<phi::dtype::float16>(out);
  } else if (dtype == phi::DenseTensorMeta::DataType::INT16) {
    dev_ctx.template Alloc<int16_t>(out);
  } else if (dtype == phi::DenseTensorMeta::DataType::INT32) {
    dev_ctx.template Alloc<int32_t>(out);
  } else if (dtype == phi::DenseTensorMeta::DataType::INT64) {
    dev_ctx.template Alloc<int64_t>(out);
  } else if (dtype == phi::DenseTensorMeta::DataType::BOOL) {
    dev_ctx.template Alloc<bool>(out);
  } else if (dtype == phi::DenseTensorMeta::DataType::UINT8) {
    dev_ctx.template Alloc<uint8_t>(out);
  } else if (dtype == phi::DenseTensorMeta::DataType::INT8) {
    dev_ctx.template Alloc<int8_t>(out);
  } else {
    phi::errors::InvalidArgument("Unsupported cast dtype %s", dtype);
  }

  experimental::OpCommand("Cast")
      .Input(x,
             experimental::TensorDescMaker("x").FromTensor(x).SetDataLayout(
                 phi::DataLayout::ANY))
      .Output(*out,
              experimental::TensorDescMaker("y").FromTensor(*out).SetDataLayout(
                  phi::DataLayout::ANY))
      .Attr("dst_type", static_cast<int32_t>(ConvertToNpuDtype(dtype)))
      .Run(dev_ctx);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(cast,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::CastKernel,
                          phi::dtype::float16,
                          float,
                          double,
                          int16_t,
                          int32_t,
                          int64_t,
                          bool) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
