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
void FusedGemmEpilogueKernel(const Context& dev_ctx,
                             const phi::DenseTensor& x,
                             const phi::DenseTensor& y,
                             const phi::DenseTensor& bias,
                             const bool trans_x,
                             const bool trans_y,
                             const std::string& activation,
                             phi::DenseTensor* out,
                             phi::DenseTensor* reserve_space) {
  float alpha = 1.0f;
  float beta = 1.0f;
  int64_t transpose_x = 0;
  int64_t transpose_y = 0;
  if (trans_x) {
    transpose_x = 1;
  }
  if (trans_y) {
    transpose_y = 1;
  }
  int8_t cube_math_type = 0;
  dev_ctx.template Alloc<T>(out);
  EXEC_NPU_CMD(aclnnGemm,
               dev_ctx,
               x,
               y,
               bias,
               alpha,
               beta,
               trans_x,
               trans_y,
               *out,
               cube_math_type);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(fused_gemm_epilogue,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::FusedGemmEpilogueKernel,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
