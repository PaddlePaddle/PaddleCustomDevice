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

template <typename T, typename Context>
void AclopOneHotRawKernel(const Context& dev_ctx,
                          const phi::DenseTensor& x,
                          const phi::Scalar& depth_scalar,
                          phi::DataType dtype,
                          bool allow_out_of_range,
                          phi::DenseTensor* out) {
  auto stream = dev_ctx.stream();

  int depth = depth_scalar.to<int>();
  auto out_dims = out->dims();
  out_dims[out_dims.size() - 1] = depth;
  out->Resize(out_dims);

  dev_ctx.template Alloc<float>(out);

  float on_value = 1.0f, off_value = 0.0f;
  if (x.dtype() == phi::DataType::INT32) {
    NpuOpRunner runner;
    runner.SetType("OneHot")
        .AddInput(x)
        .AddInput(dev_ctx, std::vector<int32_t>({static_cast<int32_t>(depth)}))
        .AddInput(dev_ctx, std::vector<float>({on_value}))
        .AddInput(dev_ctx, std::vector<float>({off_value}))
        .AddAttr("axis", -1)
        .AddOutput(*out);
    runner.Run(stream);
  } else {
    phi::DenseTensor transformed_in;
    transformed_in.Resize(x.dims());
    custom_kernel::CastKernel<T, Context>(
        dev_ctx, x, phi::DataType::INT32, &transformed_in);
    NpuOpRunner runner;
    runner.SetType("OneHot")
        .AddInput(transformed_in)
        .AddInput(dev_ctx, std::vector<int32_t>({static_cast<int32_t>(depth)}))
        .AddInput(dev_ctx, std::vector<float>({on_value}))
        .AddInput(dev_ctx, std::vector<float>({off_value}))
        .AddAttr("axis", -1)
        .AddOutput(*out);
    runner.Run(stream);
  }
}

template <typename T, typename Context>
void OneHotRawKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::Scalar& depth_scalar,
                     phi::DataType dtype,
                     bool allow_out_of_range,
                     phi::DenseTensor* out) {
#if (CANN_VERSION_CODE < 800000)
  custom_kernel::AclopOneHotRawKernel<T, Context>(
      dev_ctx, x, depth_scalar, dtype, allow_out_of_range, out);
#else
  DO_COMPATIBILITY(
      aclnnOneHot,
      (custom_kernel::AclopOneHotRawKernel<T, Context>(
          dev_ctx, x, depth_scalar, dtype, allow_out_of_range, out)));

  auto stream = dev_ctx.stream();

  int depth = depth_scalar.to<int>();
  auto out_dims = out->dims();
  out_dims[out_dims.size() - 1] = depth;
  out->Resize(out_dims);
  auto out_shape_vec = phi::vectorize(out_dims);

  dev_ctx.template Alloc<T>(out);

  int64_t axis = -1;
  phi::DenseTensor on_value_tensor, off_value_tensor;
  phi::DenseTensorMeta meta = {phi::DataType::INT64, {1}};
  on_value_tensor.set_meta(meta);
  off_value_tensor.set_meta(meta);
  dev_ctx.template Alloc<int64_t>(&on_value_tensor);
  dev_ctx.template Alloc<int64_t>(&off_value_tensor);
  custom_kernel::FillNpuTensorWithConstant<T>(
      &on_value_tensor, dev_ctx, static_cast<int64_t>(1));
  custom_kernel::FillNpuTensorWithConstant<T>(
      &off_value_tensor, dev_ctx, static_cast<int64_t>(0));

  if (x.dtype() == phi::DataType::INT32) {
    EXEC_NPU_CMD(aclnnOneHot,
                 dev_ctx,
                 x,
                 depth,
                 on_value_tensor,
                 off_value_tensor,
                 axis,
                 *out);
  } else {
    phi::DenseTensor transformed_in;
    transformed_in.Resize(x.dims());
    custom_kernel::CastKernel<T, Context>(
        dev_ctx, x, phi::DataType::INT32, &transformed_in);
    EXEC_NPU_CMD(aclnnOneHot,
                 dev_ctx,
                 transformed_in,
                 depth,
                 on_value_tensor,
                 off_value_tensor,
                 axis,
                 *out);
  }
#endif
}

template <typename T, typename Context>
void OneHotKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::Scalar& num_classes_s,
                  phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  custom_kernel::OneHotRawKernel<T, Context>(
      dev_ctx, x, num_classes_s, phi::DataType::FLOAT32, false, out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(one_hot_raw,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::OneHotRawKernel,
                          int32_t,
                          int64_t) {}

PD_REGISTER_PLUGIN_KERNEL(
    one_hot, npu, ALL_LAYOUT, custom_kernel::OneHotKernel, int32_t, int64_t) {
  kernel->OutputAt(0).SetDataType(phi::DataType::FLOAT32);
}
