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
void MultiplyKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    phi::DenseTensor* out);

template <typename T, typename Context>
void AddKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::DenseTensor& y,
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

  if (std::isinf(scale) || std::isnan(scale)) {
    FillNpuTensorWithConstant<T>(out, dev_ctx, static_cast<T>(scale));
    return;
  }
  if (!bias_after_scale) {
    bias *= scale;
  }

  phi::DenseTensor scale_tensor;
  scale_tensor.Resize(phi::make_ddim({1}));
  FillNpuTensorWithConstant<float>(&scale_tensor, dev_ctx, scale);

  phi::DenseTensor bias_tensor;
  bias_tensor.Resize(phi::make_ddim({1}));
  FillNpuTensorWithConstant<float>(&bias_tensor, dev_ctx, bias);

  custom_kernel::MultiplyKernel<T, Context>(dev_ctx, x, scale_tensor, out);

  custom_kernel::AddKernel<T, Context>(dev_ctx, *out, bias_tensor, out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(scale,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::ScaleKernel,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          float,
                          double,
                          int,
                          int64_t) {}
