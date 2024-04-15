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
void AclopClipKernel(const Context& dev_ctx,
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

  if (x.dtype() == phi::DataType::INT64) {
    auto op_func = [](const std::vector<phi::DenseTensor>& inputs,
                      const std::vector<phi::DenseTensor>& outputs,
                      const NPUAttributeMap& attrs,
                      const phi::CustomContext& dev_ctx) {
      const auto& runner = NpuOpRunner("ClipByValue",
                                       {inputs[0], inputs[1], inputs[2]},
                                       {outputs[0]},
                                       attrs);
      runner.Run(dev_ctx.stream());
    };
    NpuOpRunner::TypeAdapter(
        {x, min_tensor, max_tensor},
        {*out},
        {},
        dev_ctx,
        op_func,
        {phi::DataType::INT32, phi::DataType::INT32, phi::DataType::INT32},
        {phi::DataType::INT32});
  } else {
    auto stream = dev_ctx.stream();
    const auto& runner =
        NpuOpRunner("ClipByValue", {x, min_tensor, max_tensor}, {*out}, {});
    runner.Run(stream);
  }
}

template <typename T, typename Context>
void ClipKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::Scalar& min,
                const phi::Scalar& max,
                phi::DenseTensor* out) {
  DO_COMPATIBILITY(
      aclnnClamp,
      (custom_kernel::AclopClipKernel<T, Context>(dev_ctx, x, min, max, out)));
  dev_ctx.template Alloc<T>(out);
  EXEC_NPU_CMD(aclnnClamp, dev_ctx, x, min, max, *out);
}

template <typename T, typename Context>
void AclopClipGradKernel(const Context& dev_ctx,
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

template <typename T, typename Context>
void ClipGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& dout,
                    const phi::Scalar& min,
                    const phi::Scalar& max,
                    phi::DenseTensor* dx) {
  DO_COMPATIBILITY(aclnnHardtanhBackward,
                   (custom_kernel::AclopClipGradKernel<T, Context>(
                       dev_ctx, x, dout, min, max, dx)));
  dev_ctx.template Alloc<T>(dx);
  EXEC_NPU_CMD(aclnnHardtanhBackward, dev_ctx, dout, x, min, max, *dx);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(clip,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::ClipKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(clip_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::ClipGradKernel,
                          float,
                          phi::dtype::float16,
                          double) {}
