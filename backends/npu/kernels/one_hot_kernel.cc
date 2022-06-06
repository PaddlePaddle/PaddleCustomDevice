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

namespace custom_kernel {

template <typename T, typename Context>
void OneHotRawKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::Scalar& depth_scalar,
                     phi::DataType dtype,
                     bool allow_out_of_range,
                     phi::DenseTensor* out) {
  auto stream = dev_ctx.stream();

  int depth = depth_scalar.to<int>();
  auto out_dims = out->dims();
  out_dims[out_dims.size() - 1] = depth;
  out->Resize(out_dims);

  dev_ctx.template Alloc<float>(out);

  float on_value = 1.0f, off_value = 0.0f;
  if (x.dtype() == phi::DataType::INT32) {
    NpuOpRunner runner;
    runner.SetType("OneHot")
        .AddInput(x)
        .AddInput(dev_ctx, std::vector<int32_t>({static_cast<int32_t>(depth)}))
        .AddInput(dev_ctx, std::vector<float>({on_value}))
        .AddInput(dev_ctx, std::vector<float>({off_value}))
        .AddAttr("axis", -1)
        .AddOutput(*out);
    runner.Run(stream);
  } else {
    phi::DenseTensor transformed_in;
    transformed_in.Resize(x.dims());
    dev_ctx.template Alloc<int32_t>(&transformed_in);
    const auto& cast_runner =
        NpuOpRunner("Cast", {x}, {transformed_in}, {{"dst_type", ACL_INT32}});
    cast_runner.Run(stream);
    NpuOpRunner runner;
    runner.SetType("OneHot")
        .AddInput(transformed_in)
        .AddInput(dev_ctx, std::vector<int32_t>({static_cast<int32_t>(depth)}))
        .AddInput(dev_ctx, std::vector<float>({on_value}))
        .AddInput(dev_ctx, std::vector<float>({off_value}))
        .AddAttr("axis", -1)
        .AddOutput(*out);
    runner.Run(stream);
  }
}

template <typename T, typename Context>
void OneHotKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::Scalar& num_classes_s,
                  phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  PADDLE_THROW(phi::errors::Unimplemented("OneHotKernel is not need?"));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(one_hot_raw,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::OneHotRawKernel,
                          int32_t,
                          int64_t) {}

PD_REGISTER_PLUGIN_KERNEL(one_hot,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::OneHotKernel,
                          int32_t,
                          int64_t) {}
