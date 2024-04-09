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
void CastKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DataType dtype,
                phi::DenseTensor* out);

using Tensor = phi::DenseTensor;

template <typename Context>
static void CumsumImp(const Tensor& input,
                      Tensor* output,
                      const NPUAttributeMap& attr_input,
                      const Context& dev_ctx) {
  auto stream = dev_ctx.stream();
  if (input.dtype() == phi::DataType::INT64) {
    Tensor tmp_input;
    tmp_input.Resize(input.dims());
    dev_ctx.template Alloc<float>(&tmp_input);

    auto dst_acl_dtype = ConvertToNpuDtype(tmp_input.dtype());
    const auto& cast_runner_1 =
        NpuOpRunner("Cast",
                    {input},
                    {tmp_input},
                    {{"dst_type", static_cast<int>(dst_acl_dtype)}});
    cast_runner_1.Run(stream);

    Tensor tmp_output;
    tmp_output.Resize(output->dims());
    dev_ctx.template Alloc<float>(&tmp_output);

    const auto& runner =
        NpuOpRunner("CumsumD", {tmp_input}, {tmp_output}, attr_input);
    runner.Run(stream);

    dst_acl_dtype = ConvertToNpuDtype(output->dtype());
    const auto& cast_runner_2 =
        NpuOpRunner("Cast",
                    {tmp_output},
                    {*output},
                    {{"dst_type", static_cast<int>(dst_acl_dtype)}});
    cast_runner_2.Run(stream);
  } else {
    const auto& runner = NpuOpRunner("CumsumD", {input}, {*output}, attr_input);
    runner.Run(stream);
  }
}

template <typename T, typename Context>
void AclopCumsumKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::Scalar& axis_scalar,
                       bool flatten,
                       bool exclusive,
                       bool reverse,
                       phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  auto axis = axis_scalar.to<int>();

  NPUAttributeMap attr_input = {
      {"axis", axis}, {"exclusive", exclusive}, {"reverse", reverse}};

  if (flatten) {
    PADDLE_ENFORCE_EQ(
        axis,
        -1,
        phi::errors::InvalidArgument(
            "when flatten is true, attr axis must be default %d, but got %d",
            -1,
            axis));

    Tensor new_x(x);
    new_x.Resize(phi::make_ddim({x.numel()}));

    CumsumImp<Context>(new_x, out, attr_input, dev_ctx);
  } else {
    CumsumImp<Context>(x, out, attr_input, dev_ctx);
  }
}

template <typename T, typename Context>
static void CumsumAclnnImp(const Tensor& input,
                           Tensor* output,
                           int64_t axis,
                           const Context& dev_ctx) {
  if (input.dtype() == phi::DataType::INT64) {
    Tensor tmp_input;
    tmp_input.Resize(input.dims());

    custom_kernel::CastKernel<T, Context>(
        dev_ctx, input, tmp_input.dtype(), &tmp_input);

    Tensor tmp_output;
    tmp_output.Resize(output->dims());
    dev_ctx.template Alloc<float>(&tmp_output);
    auto dst_acl_dtype = ConvertToNpuDtype(tmp_input.dtype());

    EXEC_NPU_CMD(
        aclnnReduceSum, dev_ctx, tmp_input, axis, dst_acl_dtype, tmp_output);

    custom_kernel::CastKernel<T, Context>(
        dev_ctx, tmp_output, output->dtype(), output);
  } else {
    auto dst_acl_dtype = ConvertToNpuDtype(input.dtype());

    EXEC_NPU_CMD(aclnnReduceSum, dev_ctx, input, axis, dst_acl_dtype, *output);
  }
}

template <typename T, typename Context>
void CumsumKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::Scalar& axis_scalar,
                  bool flatten,
                  bool exclusive,
                  bool reverse,
                  phi::DenseTensor* out) {
  DO_COMPATIBILITY(
      aclnnCumSum,
      (custom_kernel::AclopCumsumKernel<T, Context>(
          dev_ctx, x, axis_scalar, flatten, exclusive, reverse, out)));
  dev_ctx.template Alloc<T>(out);

  auto axis = axis_scalar.to<int64_t>();

  if (flatten) {
    PADDLE_ENFORCE_EQ(
        axis,
        -1,
        phi::errors::InvalidArgument(
            "when flatten is true, attr axis must be default %d, but got %d",
            -1,
            axis));

    Tensor new_x(x);
    new_x.Resize(phi::make_ddim({x.numel()}));

    CumsumAclnnImp<T, Context>(new_x, out, axis, dev_ctx);
  } else {
    CumsumAclnnImp<T, Context>(x, out, axis, dev_ctx);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(cumsum,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::CumsumKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16) {}
