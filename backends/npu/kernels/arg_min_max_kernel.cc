/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void ArgMinKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::Scalar& axis,
                  bool keepdims,
                  bool flatten,
                  int dtype,
                  phi::DenseTensor* out) {
  dev_ctx.Alloc(out, out->dtype());
  auto stream = dev_ctx.stream();

  // NOTE: The dtype of output is decided by param dtype, if param dtype is 2,
  // output's dtype is int, and if param dtype is 3, output's dtype is int64.
  // See the detail in
  // https://github.com/PaddlePaddle/Paddle/blob/f9043c78e55b182e05d1d1efa7da930d8abc28d2/paddle/phi/infermeta/unary.cc#L209
  if (dtype == 2) {
    NpuOpRunner runner;
    runner.SetType("ArgMin")
        .AddInput(x)
        .AddInput(dev_ctx, std::vector<int64_t>({axis.to<int64_t>()}))
        .AddOutput(*out)
        .AddAttr("dtype", dtype);
    runner.Run(stream);
  } else if (dtype == 3) {
    // TODO(songkai05): core dump happend when the dtype of CANN op ArgMin's
    // output is int64, so we compute the int32 result and cast it to int64 when
    // param dtype is 3 temporarily.
    NPUAttributeMap attrs = {{"dtype", dtype}};

    auto op_runner = [](const std::vector<phi::DenseTensor>& inputs,
                        const std::vector<phi::DenseTensor>& outputs,
                        const NPUAttributeMap& attrs,
                        const phi::CustomContext& dev_ctx,
                        const auto& host_vecs) {
      NpuOpRunner runner;
      runner.SetType("ArgMin")
          .AddInput(inputs[0])
          .AddInput(dev_ctx, std::move(host_vecs[0]))
          .AddOutput(outputs[0])
          .AddAttrs(attrs);
      runner.Run(dev_ctx.stream());
    };

    NpuOpRunner::TypeAdapter<int64_t>(
        {x},
        {*out},
        attrs,
        dev_ctx,
        op_runner,
        {x.dtype()},
        {phi::DataType::INT32},
        {std::vector<int64_t>({axis.to<int64_t>()})});
  }
}

template <typename T, typename Context>
void ArgMaxKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::Scalar& axis,
                  bool keepdims,
                  bool flatten,
                  int dtype,
                  phi::DenseTensor* out) {
  dev_ctx.Alloc(out, out->dtype());
  auto stream = dev_ctx.stream();

  phi::DenseTensor transformed_x;
  // TODO(songkai05): CANN512 doesn't support double dtype for ArgMax NPU op,
  // we cast double to float32 to support double dtype for now.
  if (x.dtype() == phi::DataType::FLOAT64 ||
      x.dtype() == phi::DataType::INT32) {
    phi::DenseTensorMeta meta = {phi::DataType::FLOAT32, x.dims()};
    transformed_x.set_meta(meta);
    dev_ctx.template Alloc<float>(&transformed_x);
    const auto& cast_runner =
        NpuOpRunner("Cast", {x}, {transformed_x}, {{"dst_type", ACL_FLOAT}});
    cast_runner.Run(stream);
  } else {
    transformed_x = x;
  }
  if (flatten) {
    transformed_x.Resize(phi::make_ddim({x.numel()}));
  }

  std::vector<int64_t> axis_v;
  axis_v.push_back(axis.to<int64_t>());

  // NOTE: The dtype of output is decided by param dtype, if param dtype is 2,
  // output's dtype is int, and if param dtype is 3, output's dtype is int64.
  // See the detail in
  // https://github.com/PaddlePaddle/Paddle/blob/f9043c78e55b182e05d1d1efa7da930d8abc28d2/paddle/phi/infermeta/unary.cc#L209
  int out_dtype;
  if (dtype == 2) {
    out_dtype = static_cast<int>(phi::DataType::INT32);
  } else if (dtype == 3) {
    out_dtype = static_cast<int>(phi::DataType::INT64);
  }

  NpuOpRunner runner;
  runner.SetType("ArgMaxV2")
      .AddInput(transformed_x)
      .AddInput(dev_ctx, std::move(axis_v))
      .AddOutput(*out)
      .AddAttrDataType("dtype", out_dtype)
      .Run(stream);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(argmin,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::ArgMinKernel,
                          float,
                          phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}

PD_REGISTER_PLUGIN_KERNEL(argmax,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::ArgMaxKernel,
                          int,
                          float,
                          double,
                          phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
