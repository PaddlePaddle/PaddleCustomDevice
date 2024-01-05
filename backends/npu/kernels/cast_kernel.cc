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
#include "kernels/funcs/npu_op_runner.h"
#include "paddle/phi/core/tensor_meta.h"

namespace custom_kernel {

template <typename T, typename Context>
void CastKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DataType dtype,
                phi::DenseTensor* out) {
  if (x.dtype() == dtype) {
    dev_ctx.template Alloc<T>(out);
    TensorCopy(dev_ctx, x, false, out);
    return;
  }

  int aclDtype = ConvertToNpuDtype(dtype);

  if (dtype == phi::DataType::FLOAT32) {
    dev_ctx.template Alloc<float>(out);
  } else if (dtype == phi::DataType::FLOAT64) {
    dev_ctx.template Alloc<double>(out);
  } else if (dtype == phi::DataType::FLOAT16) {
    dev_ctx.template Alloc<phi::dtype::float16>(out);
  } else if (dtype == phi::DataType::INT16) {
    dev_ctx.template Alloc<int16_t>(out);
  } else if (dtype == phi::DataType::INT32) {
    dev_ctx.template Alloc<int32_t>(out);
  } else if (dtype == phi::DataType::INT64) {
    dev_ctx.template Alloc<int64_t>(out);
  } else if (dtype == phi::DataType::BOOL) {
    dev_ctx.template Alloc<bool>(out);
  } else if (dtype == phi::DataType::UINT8) {
    dev_ctx.template Alloc<uint8_t>(out);
  } else if (dtype == phi::DataType::INT8) {
    dev_ctx.template Alloc<int8_t>(out);
  } else if (dtype == phi::DataType::COMPLEX64) {
    dev_ctx.template Alloc<phi::dtype::complex<float>>(out);
  } else if (dtype == phi::DataType::COMPLEX128) {
    dev_ctx.template Alloc<phi::dtype::complex<double>>(out);
  } else {
    phi::errors::InvalidArgument("Unsupported cast dtype %s", dtype);
  }

  aclrtStream stream = static_cast<aclrtStream>(dev_ctx.stream());

  if (x.dtype() == phi::DataType::INT64) {
    phi::DenseTensor tmp_out;
    tmp_out.Resize(out->dims());
    dev_ctx.template Alloc<int32_t>(&tmp_out);
    const auto& runner =
        NpuOpRunner("Cast",
                    {x},
                    {tmp_out},
                    {{"dst_type", static_cast<int32_t>(ACL_INT32)}});
    runner.Run(stream);

    const auto& runner1 =
        NpuOpRunner("Cast",
                    {tmp_out},
                    {*out},
                    {{"dst_type", static_cast<int32_t>(aclDtype)}});
    runner1.Run(stream);
  } else {
    const auto& runner = NpuOpRunner(
        "Cast", {x}, {*out}, {{"dst_type", static_cast<int32_t>(aclDtype)}});
    runner.Run(stream);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(cast,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::CastKernel,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          float,
                          double,
                          int8_t,
                          uint8_t,
                          int16_t,
                          int32_t,
                          int64_t,
                          bool) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}