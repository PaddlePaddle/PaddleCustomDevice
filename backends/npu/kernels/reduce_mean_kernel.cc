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

  auto input_dims_vec = phi::vectorize(x.dims());
  if (reduce_all) {
    dims.clear();
    for (size_t i = 0; i < input_dims_vec.size(); i++) {
      dims.push_back(static_cast<int>(i));
    }
  }
  const auto& runner = NpuOpRunner(
      "ReduceMeanD", {x}, {*out}, {{"axes", dims}, {"keep_dims", keep_dim}});
  aclrtStream stream = static_cast<aclrtStream>(dev_ctx.stream());
  runner.Run(stream);
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
  auto input_dims_vec = phi::vectorize(x.dims());
  dev_ctx.template Alloc<T>(x_grad);

  int reduce_numel = 1;
  if (reduce_all) {
    reduce_dims.clear();
    for (size_t d = 0; d < input_dims_vec.size(); ++d) {
      reduce_dims.push_back(static_cast<int>(d));
    }
  }
  for (auto& d : reduce_dims) {
    if (d < 0) {
      d = d + input_dims_vec.size();
    }
    reduce_numel *= input_dims_vec[d];
  }

  auto tmp = input_dims_vec;
  const auto& runner = NpuOpRunner(
      "FillV2D",
      {},
      {*x_grad},
      {{"value", 1.0f / static_cast<float>(reduce_numel)}, {"dims", tmp}});
  runner.Run(stream);

  phi::DenseTensor transformed_x_grad, transformed_out_grad;
  phi::DenseTensor tmp_out_grad;
  auto tmp_output_dims_vec = input_dims_vec;
  for (auto d : reduce_dims) {
    tmp_output_dims_vec[d] = 1;
  }
  tmp_out_grad = out_grad;
  tmp_out_grad.Resize(phi::make_ddim(tmp_output_dims_vec));
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

PD_REGISTER_PLUGIN_KERNEL(
    mean_raw, ascend, ALL_LAYOUT, custom_kernel::MeanRawKernel, float) {}

PD_REGISTER_PLUGIN_KERNEL(
    mean, ascend, ALL_LAYOUT, custom_kernel::MeanKernel, float) {}

PD_REGISTER_PLUGIN_KERNEL(
    mean_grad, ascend, ALL_LAYOUT, custom_kernel::MeanGradKernel, float) {}
