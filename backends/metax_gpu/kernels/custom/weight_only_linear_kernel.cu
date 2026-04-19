/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "mctlass/epilogue/thread/scale_type.h"
#include "mctlass/half.h"
#include "mctlass/layout/matrix.h"
#include "mctlass/mctlass_ex.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/datatype_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/weight_only_gemv.h"
#include "paddle/phi/kernels/weight_only_linear_kernel.h"

namespace phi {

template <typename T, typename Context>
void WeightOnlyLinearKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& weight,
                            const paddle::optional<DenseTensor>& bias,
                            const DenseTensor& weight_scale,
                            const std::string& weight_dtype,
                            const int32_t arch,
                            const int32_t group_size,
                            DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();
  const T* x_data = x.data<T>();
  const int8_t* weight_data = weight.data<int8_t>();
  const T* bias_data = bias ? bias.get().data<T>() : nullptr;
  const T* weight_scale_data = weight_scale.data<T>();
  T* out_data = out->data<T>();
  const auto x_dims = x.dims();
  const auto w_dims = weight.dims();
  int n = group_size > 0 ? weight_scale.dims()[1] : weight_scale.dims()[0];
  int k = w_dims[1];
  int m = x.numel() / k;

  using ElementA = maca_bfloat16;
  using ElementB_w8a16 = int8_t;
  using ElementB_w4a16 = uint8_t;
  using ElementC = maca_bfloat16;
  using ElementCompute = float;
  using ElementOutput = ElementC;
  using LayoutA = mctlass::layout::RowMajor;
  using LayoutB = mctlass::layout::ColumnMajor;
  using LayoutC = mctlass::layout::RowMajor;
  using ArchTag = mctlass::arch::Sm80;

  using mctlassGemmScaleOp_w8a16_nobias =
      mctlassGemmScale<ElementA,
                       LayoutA,
                       ElementB_w8a16,
                       LayoutB,
                       ElementC,
                       LayoutC,
                       ElementCompute,
                       ArchTag,
                       mctlass::epilogue::thread::ScaleType::NoScaleAsBs>;

  using mctlassGemmScaleOp_w8a16_bias =
      mctlassGemmScale<ElementA,
                       LayoutA,
                       ElementB_w8a16,
                       LayoutB,
                       ElementC,
                       LayoutC,
                       ElementCompute,
                       ArchTag,
                       mctlass::epilogue::thread::ScaleType::ScaleOnlyBias>;

  using mctlassGemmScaleOp_w4a16_nobias =
      mctlassGemmScale<ElementA,
                       LayoutA,
                       ElementB_w4a16,
                       LayoutB,
                       ElementC,
                       LayoutC,
                       ElementCompute,
                       ArchTag,
                       mctlass::epilogue::thread::ScaleType::NoScaleAsBs>;

  using mctlassGemmScaleOp_w4a16_bias =
      mctlassGemmScale<ElementA,
                       LayoutA,
                       ElementB_w4a16,
                       LayoutB,
                       ElementC,
                       LayoutC,
                       ElementCompute,
                       ArchTag,
                       mctlass::epilogue::thread::ScaleType::ScaleOnlyBias>;

  mctlass::gemm::GemmCoord problem_size(m, n, k);

  if (weight_dtype == "int8") {
    if (bias_data == nullptr) {
      mctlassGemmScaleOp_w8a16_nobias mctlass_op;
      typename mctlassGemmScaleOp_w8a16_nobias::Arguments arguments{
          mctlass::gemm::GemmUniversalMode::kGemmQuantB,
          problem_size,
          1,
          mctlassGemmScaleOp_w8a16_nobias::epilogueParams(
              reinterpret_cast<const maca_bfloat16*>(bias_data)),
          mctlassGemmScaleOp_w8a16_nobias::quantscaleParams(
              1,
              group_size,
              reinterpret_cast<const maca_bfloat16*>(weight_scale_data)),
          reinterpret_cast<const maca_bfloat16*>(x_data),
          weight_data,
          reinterpret_cast<const maca_bfloat16*>(out_data),
          out_data,
          m * k,
          n * k,
          m * n,
          m * n,
          k,
          k,
          n,
          n};
      mctlass_op(arguments, NULL, stream);
    } else {
      mctlassGemmScaleOp_w8a16_bias mctlass_op;
      typename mctlassGemmScaleOp_w8a16_bias::Arguments arguments{
          mctlass::gemm::GemmUniversalMode::kGemmQuantB,
          problem_size,
          1,
          mctlassGemmScaleOp_w8a16_bias::epilogueParams(
              reinterpret_cast<const maca_bfloat16*>(bias_data)),
          mctlassGemmScaleOp_w8a16_bias::quantscaleParams(
              1,
              group_size,
              reinterpret_cast<const maca_bfloat16*>(weight_scale_data)),
          reinterpret_cast<const maca_bfloat16*>(x_data),
          weight_data,
          reinterpret_cast<const maca_bfloat16*>(out_data),
          out_data,
          m * k,
          n * k,
          m * n,
          m * n,
          k,
          k,
          n,
          n};
      mctlass_op(arguments);
    }
  } else if (weight_dtype == "int4") {
    if (bias_data == nullptr) {
      mctlassGemmScaleOp_w4a16_nobias mctlass_op;
      typename mctlassGemmScaleOp_w4a16_nobias::Arguments arguments{
          mctlass::gemm::GemmUniversalMode::kGemmQuantB,
          problem_size,
          1,
          mctlassGemmScaleOp_w4a16_nobias::epilogueParams(
              reinterpret_cast<const maca_bfloat16*>(bias_data)),
          mctlassGemmScaleOp_w4a16_nobias::quantscaleParams(
              2,
              group_size,
              reinterpret_cast<const maca_bfloat16*>(weight_scale_data)),
          reinterpret_cast<const maca_bfloat16*>(x_data),
          weight_data,
          reinterpret_cast<const maca_bfloat16*>(out_data),
          out_data,
          m * k,
          n * k,
          m * n,
          m * n,
          k,
          k,
          n,
          n};
      mctlass_op(arguments);
    } else {
      mctlassGemmScaleOp_w4a16_bias mctlass_op;
      typename mctlassGemmScaleOp_w4a16_bias::Arguments arguments{
          mctlass::gemm::GemmUniversalMode::kGemmQuantB,
          problem_size,
          1,
          mctlassGemmScaleOp_w4a16_bias::epilogueParams(
              reinterpret_cast<const maca_bfloat16*>(bias_data)),
          mctlassGemmScaleOp_w4a16_bias::quantscaleParams(
              2,
              group_size,
              reinterpret_cast<const maca_bfloat16*>(weight_scale_data)),
          reinterpret_cast<const maca_bfloat16*>(x_data),
          weight_data,
          reinterpret_cast<const maca_bfloat16*>(out_data),
          out_data,
          m * k,
          n * k,
          m * n,
          m * n,
          k,
          k,
          n,
          n};
      mctlass_op(arguments);
    }
  }
}
}  // namespace phi

PD_REGISTER_PLUGIN_KERNEL(weight_only_linear,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::WeightOnlyLinearKernel,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
