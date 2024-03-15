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
void TransposeKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const std::vector<int>& axis,
                     phi::DenseTensor* out);

template <typename T, typename Context>
phi::DenseTensor TransposeInput(const Context& dev_ctx,
                                phi::DenseTensor input) {
  phi::DenseTensor transed_input;
  int64_t input_shape_size = input.dims().size();
  std::vector<int> input_axis, input_shape;
  for (int64_t i = 0; i < input_shape_size; i++) {
    input_axis.push_back(input_shape_size - i - 1);
    input_shape.push_back(input.dims()[input_shape_size - i - 1]);
  }
  transed_input.Resize(phi::make_ddim(input_shape));
  custom_kernel::TransposeKernel<T, Context>(
      dev_ctx, input, input_axis, &transed_input);
  return transed_input;
}

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

  int8_t cube_math_type = 0;
  phi::IntArray axis = {0};
  bool keep_dim = false;
  auto dst_dtype = ConvertToNpuDtype(out_grad.dtype());

  phi::DenseTensor transed_x, transed_y, transed_grad;

  if (trans_x) {
    transed_grad = TransposeInput<T, Context>(dev_ctx, out_grad);
    if (trans_y) {
      transed_x = TransposeInput<T, Context>(dev_ctx, x);
      transed_y = TransposeInput<T, Context>(dev_ctx, y);
      EXEC_NPU_CMD(aclnnMatmul,
                   dev_ctx,
                   transed_y,
                   transed_grad,
                   *x_grad,
                   cube_math_type);
      EXEC_NPU_CMD(aclnnMatmul,
                   dev_ctx,
                   transed_grad,
                   transed_x,
                   *y_grad,
                   cube_math_type);
    } else {
      EXEC_NPU_CMD(
          aclnnMatmul, dev_ctx, y, transed_grad, *x_grad, cube_math_type);
      EXEC_NPU_CMD(aclnnMatmul, dev_ctx, x, out_grad, *y_grad, cube_math_type);
    }
  } else {
    if (trans_y) {
      transed_grad = TransposeInput<T, Context>(dev_ctx, out_grad);
      EXEC_NPU_CMD(aclnnMatmul, dev_ctx, out_grad, y, *x_grad, cube_math_type);
      EXEC_NPU_CMD(
          aclnnMatmul, dev_ctx, transed_grad, x, *y_grad, cube_math_type);
    } else {
      transed_x = TransposeInput<T, Context>(dev_ctx, x);
      transed_y = TransposeInput<T, Context>(dev_ctx, y);
      EXEC_NPU_CMD(
          aclnnMatmul, dev_ctx, out_grad, transed_y, *x_grad, cube_math_type);
      EXEC_NPU_CMD(
          aclnnMatmul, dev_ctx, transed_x, out_grad, *y_grad, cube_math_type);
    }
  }

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
