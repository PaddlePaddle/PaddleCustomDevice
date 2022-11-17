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
#include "kernels/funcs/op_command.h"

namespace custom_kernel {

template <typename T, typename Context>
void FullLikeKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::Scalar& val,
                    phi::DataType dtype,
                    phi::DenseTensor* out);

template <typename T, typename Context>
void ScaleKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const phi::Scalar& in_scale,
                 float bias,
                 bool bias_after_scale,
                 phi::DenseTensor* out) {
  auto scale = in_scale.to<float>();
  VLOG(4) << "scale:" << scale << ", bias:" << bias
          << " ,bias_after_scale:" << bias_after_scale;
  if (std::isinf(scale)) {
    if (std::signbit(scale)) {
      scale = -std::numeric_limits<float>::max();
    } else {
      scale = std::numeric_limits<float>::max();
    }

    phi::DenseTensor scale_tensor;
    scale_tensor.Resize({1});
    dev_ctx.template HostAlloc<T>(&scale_tensor);
    *(scale_tensor.data<T>()) = scale;
    custom_kernel::FullLikeKernel<T, Context>(
        dev_ctx, x, scale_tensor, x.dtype(), out);
    return;
  }
  if (!bias_after_scale) {
    bias *= scale;
  }
  dev_ctx.template Alloc<T>(out);

  phi::DenseTensor x_mul_scale;
  x_mul_scale.Resize(x.dims());
  dev_ctx.template Alloc<T>(&x_mul_scale);

  experimental::OpCommand("Muls")
      .Input(x)
      .Output(x_mul_scale)
      .Attr("value", scale)
      .Run(dev_ctx);
  experimental::OpCommand("Adds")
      .Input(x_mul_scale)
      .Output(*out)
      .Attr("value", bias)
      .Run(dev_ctx);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(scale,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::ScaleKernel,
                          phi::dtype::float16,
                          float,
                          int) {}
