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
void MaxRawKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::IntArray& axes,
                  bool keep_dim,
                  bool reduce_all,
                  phi::DenseTensor* out) {
  auto dims = axes.GetData();
  dev_ctx.template Alloc<T>(out);

  NPUAttributeMap attr_input = {{"axes", dims}, {"keep_dims", keep_dim}};

  if (reduce_all) {
    std::vector<int> dim_vec;
    for (int i = 0; i < x.dims().size(); i++) {
      dim_vec.push_back(i);
    }

    attr_input = {{"axes", dim_vec}, {"keep_dims", keep_dim}};
  }

  if (x.dtype() == phi::DenseTensorMeta::DataType::INT64) {
    auto op_func = [](const std::vector<phi::DenseTensor>& inputs,
                      const std::vector<phi::DenseTensor>& outputs,
                      const NPUAttributeMap& attrs,
                      const phi::CustomContext& dev_ctx) {
      const auto& runner =
          NpuOpRunner("ReduceMaxD", {inputs[0]}, {outputs[0]}, attrs);
      runner.Run(dev_ctx.stream());
    };

    NpuOpRunner::TypeAdapter({x},
                             {*out},
                             attr_input,
                             dev_ctx,
                             op_func,
                             {phi::DenseTensorMeta::DataType::INT32},
                             {phi::DenseTensorMeta::DataType::INT32});
  } else {
    const auto& runner = NpuOpRunner("ReduceMaxD", {x}, {*out}, attr_input);
    runner.Run(dev_ctx.stream());
  }
}

template <typename T, typename Context>
void MaxKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::IntArray& dims,
               bool keep_dim,
               phi::DenseTensor* out) {
  bool reduce_all = false;
  if (dims.size() == 0) {
    reduce_all = true;
  }
  custom_kernel::MaxRawKernel<T>(dev_ctx, x, dims, keep_dim, reduce_all, out);
}

template <typename T, typename Context>
void MaxGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& out,
                   const phi::DenseTensor& out_grad,
                   const phi::IntArray& reduce_dims_in,
                   bool keep_dim,
                   bool reduce_all,
                   phi::DenseTensor* x_grad) {
  auto reduce_dims = reduce_dims_in.GetData();
  dev_ctx.template Alloc<T>(x_grad);
  auto stream = dev_ctx.stream();

  // broadcast
  auto x_dims_vec = phi::vectorize(x.dims());
  if (reduce_all) {
    reduce_dims.clear();
    for (size_t d = 0; d < x_dims_vec.size(); ++d) {
      reduce_dims.push_back(static_cast<int>(d));
    }
  }

  phi::DenseTensor tmp_out(out), tmp_out_grad(out_grad);
  auto tmp_out_dims_vec = x_dims_vec;
  for (auto d : reduce_dims) {
    if (d < 0) {
      d += x_dims_vec.size();
    }
    tmp_out_dims_vec[d] = 1;
  }

  tmp_out.Resize(phi::make_ddim(tmp_out_dims_vec));
  tmp_out_grad.Resize(phi::make_ddim(tmp_out_dims_vec));

  phi::DenseTensor transformed_out;
  phi::DenseTensorMeta meta = {x.dtype(), phi::make_ddim(x_dims_vec)};
  transformed_out.set_meta(meta);
  dev_ctx.template Alloc<T>(&transformed_out);
  NpuOpRunner r_brd_out;
  r_brd_out.SetType("BroadcastTo")
      .AddInput(tmp_out)
      .AddInput(dev_ctx, std::move(x_dims_vec))
      .AddOutput(transformed_out)
      .Run(stream);
  phi::DenseTensor transformed_out_grad;
  phi::DenseTensorMeta grad_meta = {x.dtype(), phi::make_ddim(x_dims_vec)};
  transformed_out_grad.set_meta(grad_meta);
  dev_ctx.template Alloc<T>(&transformed_out_grad);
  NpuOpRunner r_brd_out_grad;
  r_brd_out_grad.SetType("BroadcastTo")
      .AddInput(tmp_out_grad)
      .AddInput(dev_ctx, std::move(x_dims_vec))
      .AddOutput(transformed_out_grad)
      .Run(stream);

  // compare
  phi::DenseTensor equal_cond;
  equal_cond.Resize(x_grad->dims());
  dev_ctx.template Alloc<T>(&equal_cond);
  const auto& r_equal =
      NpuOpRunner("Equal", {x, transformed_out}, {equal_cond}, {});
  r_equal.Run(stream);

  // select
  phi::DenseTensor t_zero;
  t_zero.Resize(x_grad->dims());
  dev_ctx.template Alloc<T>(&t_zero);
  FillNpuTensorWithConstant(&t_zero, dev_ctx, static_cast<T>(0));
  t_zero.Resize(x_grad->dims());

  const auto& r_sel = NpuOpRunner(
      "SelectV2", {equal_cond, transformed_out_grad, t_zero}, {*x_grad}, {});
  r_sel.Run(stream);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(max_raw,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::MaxRawKernel,
                          bool,
                          int32_t,
                          int64_t,
                          phi::dtype::float16,
                          float) {}

PD_REGISTER_PLUGIN_KERNEL(max,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::MaxKernel,
                          bool,
                          int32_t,
                          int64_t,
                          phi::dtype::float16,
                          float) {}

PD_REGISTER_PLUGIN_KERNEL(max_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::MaxGradKernel,
                          bool,
                          int32_t,
                          int64_t,
                          phi::dtype::float16,
                          float) {}
