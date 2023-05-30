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
void ClipKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::Scalar& min,
                const phi::Scalar& max,
                phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  auto max_ = max.to<T>();
  auto min_ = min.to<T>();

  PADDLE_ENFORCE_LE(min_,
                    max_,
                    phi::errors::InvalidArgument(
                        "max should be greater than or equal to min. "
                        "But received min = %f, max = %f",
                        static_cast<float>(min_),
                        static_cast<float>(max_)));

  phi::DenseTensor max_tensor;
  max_tensor.Resize(phi::make_ddim({1}));
  FillNpuTensorWithConstant<T>(&max_tensor, dev_ctx, max_);

  phi::DenseTensor min_tensor;
  min_tensor.Resize(phi::make_ddim({1}));
  FillNpuTensorWithConstant<T>(&min_tensor, dev_ctx, min_);

  auto stream = dev_ctx.stream();
  const auto& runner =
      NpuOpRunner("ClipByValue", {x, min_tensor, max_tensor}, {*out}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void ClipGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& dout,
                    const phi::Scalar& min,
                    const phi::Scalar& max,
                    phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);

  auto max_ = max.to<T>();
  auto min_ = min.to<T>();

  float max_val = static_cast<float>(max_);
  float min_val = static_cast<float>(min_);

  auto stream = dev_ctx.stream();
  const auto& runner =
      NpuOpRunner("HardtanhGrad",
                  {x, dout},
                  {*dx},
                  {{"min_val", min_val}, {"max_val", max_val}});
  runner.Run(stream);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(clip,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::ClipKernel,
                          float,
                          phi::dtype::float16,
                          double,
                          int64_t) {}

PD_REGISTER_PLUGIN_KERNEL(clip_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::ClipGradKernel,
                          float,
                          phi::dtype::float16,
                          double) {}
