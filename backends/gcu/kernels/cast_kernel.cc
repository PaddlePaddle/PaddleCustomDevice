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

#include "common/gcu_op_runner.h"
#include "kernels/funcs/gcu_kernel_funcs.h"

namespace custom_kernel {
template <typename T, typename Context>
void CastKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DataType dtype,
                phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("cast");
  if (LaunchAOTKernel()) {
    custom_kernel::Cast(dev_ctx, x, dtype, out);

  } else {  // kernel impl base on JIT
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
    TensorNameMap input_names;
    input_names["X"] = {"x"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};

    TensorValueMap outputs;
    outputs["Out"] = {out};

    GcuAttributeMap attrs;
    attrs["in_dtype"] = static_cast<int>(x.dtype());
    attrs["out_dtype"] = static_cast<int>(dtype);

    GcuRunner(
        input_names, inputs, output_names, outputs, attrs, "cast", dev_ctx);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(cast,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::CastKernel,
                          phi::dtype::float16,
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
