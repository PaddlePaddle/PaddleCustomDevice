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

#include "kernels/funcs/gcu_funcs.h"
#include "kernels/funcs/gcu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void ArgMinMaxKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::Scalar& axis,
                     bool keepdims,
                     bool flatten,
                     int dtype,
                     phi::DenseTensor* out,
                     const std::string& op_type) {
  dev_ctx.Alloc(out, out->dtype());

  TensorNameMap input_names;
  input_names["X"] = {"x"};

  TensorValueMap inputs;
  inputs["X"] = {const_cast<DenseTensor*>(&x)};

  TensorNameMap output_names;
  output_names["Out"] = {"out"};

  TensorValueMap outputs;
  outputs["Out"] = {out};

  GcuAttributeMap attrs;
  attrs["axis"] = axis.to<int64_t>();
  attrs["keepdims"] = keepdims;
  attrs["flatten"] = flatten;
  attrs["dtype"] = dtype;

  GcuRunner(
      input_names, inputs, output_names, outputs, attrs, op_type, dev_ctx);
}

template <typename T, typename Context>
void ArgMinKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::Scalar& axis,
                  bool keepdims,
                  bool flatten,
                  int dtype,
                  phi::DenseTensor* out) {
  ArgMinMaxKernel<T, Context>(
      dev_ctx, x, axis, keepdims, flatten, dtype, out, "arg_min");
}

template <typename T, typename Context>
void ArgMaxKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::Scalar& axis,
                  bool keepdims,
                  bool flatten,
                  int dtype,
                  phi::DenseTensor* out) {
  ArgMinMaxKernel<T, Context>(
      dev_ctx, x, axis, keepdims, flatten, dtype, out, "arg_max");
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(argmin,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::ArgMinKernel,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}

PD_REGISTER_PLUGIN_KERNEL(argmax,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::ArgMaxKernel,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
