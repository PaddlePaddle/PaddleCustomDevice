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
void MeanRawKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::IntArray& axes,
                   bool keep_dim,
                   bool reduce_all,
                   phi::DenseTensor* out) {
  auto dims = axes.GetData();
  dev_ctx.template Alloc<T>(out);

  aclrtStream stream = static_cast<aclrtStream>(dev_ctx.stream());

  auto input_dims = x.dims();
  if (reduce_all || dims.size() == 0) {
    dims.clear();
    for (int i = 0; i < input_dims.size(); i++) {
      dims.push_back(static_cast<int>(i));
    }
  }
  NpuOpRunner runner;
  runner.SetType("ReduceMean")
      .AddInput(x)
      .AddInput(dev_ctx, std::move(dims))
      .AddOutput(*out)
      .AddAttrs({{"keep_dims", keep_dim}})
      .Run(stream);
}

template <typename T, typename Context>
void MeanKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::IntArray& dims,
                bool keep_dim,
                phi::DenseTensor* out) {
  bool reduce_all = false;
  custom_kernel::MeanRawKernel<T>(dev_ctx, x, dims, keep_dim, reduce_all, out);
}

template <typename T, typename Context>
void MeanGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& out_grad,
                    const phi::IntArray& axes,
                    bool keep_dim,
                    bool reduce_all,
                    phi::DenseTensor* x_grad) {
  aclrtStream stream = static_cast<aclrtStream>(dev_ctx.stream());
  auto reduce_dims = axes.GetData();
  dev_ctx.template Alloc<T>(x_grad);

  int reduce_numel = 1;

  auto input_dims = x.dims();

  if (reduce_all || reduce_dims.size() == 0) {
    reduce_dims.clear();
    for (int d = 0; d < input_dims.size(); ++d) {
      reduce_dims.push_back(static_cast<int>(d));
    }
  }
  for (auto& d : reduce_dims) {
    if (d < 0) {
      d = d + input_dims.size();
    }
    reduce_numel *= input_dims[d];
  }

  phi::DenseTensor tensor_value;
  phi::DenseTensorMeta value_meta = {x_grad->dtype(), {1}};
  tensor_value.set_meta(value_meta);
  dev_ctx.template Alloc<T>(&tensor_value);
  FillNpuTensorWithConstant<T>(
      &tensor_value,
      dev_ctx,
      static_cast<T>(1.0f / static_cast<T>(reduce_numel)));

  NpuOpRunner runner;
  runner.SetType("Fill")
      .AddInput(dev_ctx, phi::vectorize(input_dims))
      .AddInput(tensor_value)
      .AddOutput(*x_grad)
      .Run(stream);

  phi::DenseTensor transformed_x_grad, transformed_out_grad;
  phi::DenseTensor tmp_out_grad;
  auto tmp_output_dims = input_dims;
  for (auto d : reduce_dims) {
    tmp_output_dims[d] = 1;
  }
  tmp_out_grad = out_grad;
  tmp_out_grad.ResizeAndAllocate(tmp_output_dims);
  NpuElementWiseOpBroadcast<T>(dev_ctx,
                               x_grad,
                               &tmp_out_grad,
                               0,
                               &transformed_x_grad,
                               &transformed_out_grad);
  const auto& runner2 = NpuOpRunner(
      "Mul", {transformed_x_grad, transformed_out_grad}, {*x_grad}, {});
  runner2.Run(stream);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(mean_raw,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::MeanRawKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(mean,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::MeanKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(mean_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::MeanGradKernel,
                          float,
                          phi::dtype::float16) {}
