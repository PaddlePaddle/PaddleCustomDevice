// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
void RmsNormKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const paddle::optional<phi::DenseTensor>& bias,
                   const paddle::optional<phi::DenseTensor>& residual,
                   const phi::DenseTensor& norm_weight,
                   const paddle::optional<phi::DenseTensor>& norm_bias,
                   const float epsilon,
                   const int begin_norm_axis,
                   const float quant_scale,
                   const int quant_round_type,
                   const float quant_max_bound,
                   const float quant_min_bound,
                   phi::DenseTensor* out,
                   phi::DenseTensor* residual_out,
                   phi::DenseTensor* inv_var) {
  PADDLE_THROW(phi::errors::Unimplemented("Only supports model export"));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    rms_norm, npu, ALL_LAYOUT, custom_kernel::RmsNormKernel, phi::float16) {}
