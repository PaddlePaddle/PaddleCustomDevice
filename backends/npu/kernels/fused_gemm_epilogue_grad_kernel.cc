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
void FusedGemmEpilogueGradKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& x,
    const phi::DenseTensor& y,
    const paddle::optional<phi::DenseTensor>& reserve_space,
    const phi::DenseTensor& out_grad,
    const bool trans_x,
    const bool trans_y,
    const std::string& activation_grad,
    phi::DenseTensor* x_grad,
    phi::DenseTensor* y_grad,
    phi::DenseTensor* bias_grad) {
  dev_ctx.template Alloc<T>(x_grad);
  dev_ctx.template Alloc<T>(y_grad);
  dev_ctx.template Alloc<T>(bias_grad);

  float alpha = 1.0;
  float beta = 0.0;
  int8_t cube_math_type = 0;
  phi::IntArray axis = {0};
  bool keep_dim = false;
  auto dst_dtype = ConvertToNpuDtype(out_grad.dtype());

  int64_t trans_a1, trans_b1, trans_a2, trans_b2;
  phi::DenseTensor bias;
  phi::DenseTensorMeta bias_meta = {out_grad.dtype(), {1}};
  bias.set_meta(bias_meta);
  FillNpuTensorWithConstant<T>(&bias, dev_ctx, static_cast<T>(0));
  phi::DenseTensor x_grad_input_0, x_grad_input_1, y_grad_input_0,
      y_grad_input_1;

  if (trans_x) {
    x_grad_input_0 = y;
    x_grad_input_1 = out_grad;
    trans_b1 = 1;
    if (trans_y) {
      trans_a1 = 1;
      trans_a2 = 1;
      trans_b2 = 1;
      y_grad_input_0 = out_grad;
      y_grad_input_1 = x;
    } else {
      trans_a1 = 0;
      trans_a2 = 0;
      trans_b2 = 0;
      y_grad_input_0 = x;
      y_grad_input_1 = out_grad;
    }
  } else {
    x_grad_input_0 = out_grad;
    x_grad_input_1 = y;
    trans_a1 = 0;
    trans_a2 = 1;
    trans_b2 = 0;
    if (trans_y) {
      trans_b1 = 0;
      y_grad_input_0 = out_grad;
      y_grad_input_1 = x;
    } else {
      trans_b1 = 1;
      y_grad_input_0 = x;
      y_grad_input_1 = out_grad;
    }
  }
  EXEC_NPU_CMD(aclnnGemm,
               dev_ctx,
               x_grad_input_0,
               x_grad_input_1,
               bias,
               alpha,
               beta,
               trans_a1,
               trans_b1,
               *x_grad,
               cube_math_type);
  EXEC_NPU_CMD(aclnnGemm,
               dev_ctx,
               y_grad_input_0,
               y_grad_input_1,
               bias,
               alpha,
               beta,
               trans_a2,
               trans_b2,
               *y_grad,
               cube_math_type);

  EXEC_NPU_CMD(
      aclnnReduceSum, dev_ctx, out_grad, axis, keep_dim, dst_dtype, *bias_grad);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(fused_gemm_epilogue_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::FusedGemmEpilogueGradKernel,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
