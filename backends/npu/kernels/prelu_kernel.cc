// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
void PReluKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const phi::DenseTensor& alpha,
                 const std::string& data_format,
                 const std::string& mode,
                 phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  if (mode == "element") {
    phi::DenseTensor weight, zero_tensor, condition, alpha_broadcast;
    phi::DenseTensorMeta meta = {x.dtype(), x.dims()};
    weight.set_meta(meta);
    zero_tensor.set_meta(meta);
    alpha_broadcast.set_meta(meta);
    FillNpuTensorWithConstant<T>(&weight, dev_ctx, static_cast<T>(1));
    weight.Resize(x.dims());
    FillNpuTensorWithConstant<T>(&zero_tensor, dev_ctx, static_cast<T>(0));
    zero_tensor.Resize(x.dims());
    condition.Resize(x.dims());
    dev_ctx.template Alloc<bool>(&condition);
    dev_ctx.template Alloc<T>(&alpha_broadcast);

    std::vector<int64_t> dims;
    for (int i = 0; i < x.dims().size(); ++i) {
      dims.push_back(x.dims()[i]);
    }

    NpuOpRunner bd_runner;
    bd_runner.SetType("BroadcastTo")
        .AddInput(alpha)
        .AddInput(dev_ctx, std::move(dims))
        .AddOutput(alpha_broadcast)
        .Run(stream);

    const auto& greater_runner =
        NpuOpRunner("Greater", {x, zero_tensor}, {condition});
    greater_runner.Run(stream);

    const auto& select_runner =
        NpuOpRunner("Select", {condition, weight, alpha_broadcast}, {weight});
    select_runner.Run(stream);

    const auto& mul_runner = NpuOpRunner("Mul", {x, weight}, {*out});
    mul_runner.Run(stream);
  } else if (mode == "channel") {
    phi::DenseTensor weight(alpha);
    weight.Resize({alpha.numel()});

    if (data_format == "NCHW") {
      const auto& runner = NpuOpRunner("PRelu", {x, weight}, {*out}, {});
      runner.Run(stream);
    } else if (data_format == "NHWC") {
      PADDLE_ENFORCE_EQ(x.dims().size(),
                        4,
                        phi::errors::InvalidArgument(
                            "Input dims of Prelu must be 4, but received %d",
                            x.dims().size()));

      auto x_dims = x.dims();
      std::vector<int> transformed_dim;
      transformed_dim.push_back(x_dims[0]);
      transformed_dim.push_back(x_dims[x_dims.size() - 1]);
      for (int i = 1; i < x_dims.size() - 1; ++i) {
        transformed_dim.push_back(x_dims[i]);
      }

      phi::DenseTensor transformed_x, transformed_out;
      transformed_x.Resize(phi::make_ddim(transformed_dim));
      dev_ctx.template Alloc<T>(&transformed_x);
      transformed_out.Resize(phi::make_ddim(transformed_dim));
      dev_ctx.template Alloc<T>(&transformed_out);

      const auto& trans_runner1 =
          NpuOpRunner("TransData",
                      {x},
                      {transformed_x},
                      {{"src_format", std::string("NHWC")},
                       {"dst_format", std::string("NCHW")}});
      trans_runner1.Run(stream);

      const auto& runner =
          NpuOpRunner("PRelu", {transformed_x, weight}, {transformed_out}, {});
      runner.Run(stream);

      const auto& trans_runner2 =
          NpuOpRunner("TransData",
                      {transformed_out},
                      {*out},
                      {{"src_format", std::string("NCHW")},
                       {"dst_format", std::string("NHWC")}});
      trans_runner2.Run(stream);
    } else {
      phi::errors::Unimplemented(
          "Only NCHW and NHWC format is supported for input of PRelu kernel, "
          "but received %s .",
          data_format);
    }
  } else {
    const auto& runner = NpuOpRunner("PRelu", {x, alpha}, {*out}, {});
    runner.Run(stream);
  }
}

template <typename T, typename Context>
void PReluGradKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& alpha,
                     const phi::DenseTensor& out_grad,
                     const std::string& data_format,
                     const std::string& mode,
                     phi::DenseTensor* x_grad,
                     phi::DenseTensor* alpha_grad) {
  dev_ctx.template Alloc<T>(x_grad);
  dev_ctx.template Alloc<T>(alpha_grad);
  auto stream = dev_ctx.stream();

  if (mode == "element") {
    phi::DenseTensor weight, zero_tensor, condition, alpha_broadcast;
    phi::DenseTensorMeta meta = {x.dtype(), x.dims()};
    weight.set_meta(meta);
    zero_tensor.set_meta(meta);
    alpha_broadcast.set_meta(meta);
    FillNpuTensorWithConstant<T>(&weight, dev_ctx, static_cast<T>(1));
    weight.Resize(x.dims());
    FillNpuTensorWithConstant<T>(&zero_tensor, dev_ctx, static_cast<T>(0));
    zero_tensor.Resize(x.dims());
    condition.Resize(x.dims());
    dev_ctx.template Alloc<bool>(&condition);
    dev_ctx.template Alloc<T>(&alpha_broadcast);

    std::vector<int64_t> dims;
    for (int i = 0; i < x.dims().size(); ++i) {
      dims.push_back(x.dims()[i]);
    }

    NpuOpRunner bd_runner;
    bd_runner.SetType("BroadcastTo")
        .AddInput(alpha)
        .AddInput(dev_ctx, std::move(dims))
        .AddOutput(alpha_broadcast)
        .Run(stream);

    const auto& greater_runner =
        NpuOpRunner("Greater", {x, zero_tensor}, {condition});
    greater_runner.Run(stream);

    const auto& select_runner1 =
        NpuOpRunner("Select", {condition, weight, alpha_broadcast}, {weight});
    select_runner1.Run(stream);

    const auto& mul_runner1 = NpuOpRunner("Mul", {out_grad, weight}, {*x_grad});
    mul_runner1.Run(stream);

    const auto& select_runner2 =
        NpuOpRunner("Select", {condition, zero_tensor, x}, {zero_tensor});
    select_runner2.Run(stream);

    const auto& mul_runner2 =
        NpuOpRunner("Mul", {out_grad, zero_tensor}, {alpha_broadcast});
    mul_runner2.Run(stream);

    std::vector<int64_t> axis = {0};
    NpuOpRunner reduce_sum_runner;
    reduce_sum_runner.SetType("ReduceSum")
        .AddInput(alpha_broadcast)
        .AddInput(dev_ctx, std::move(axis))
        .AddOutput(*alpha_grad)
        .AddAttr("keep_dim", true)
        .Run(stream);
  } else if (mode == "channel") {
    phi::DenseTensor weight(alpha);
    weight.Resize({alpha.numel()});

    if (data_format == "NCHW") {
      const auto& runner = NpuOpRunner(
          "PReluGrad", {out_grad, x, weight}, {*x_grad, *alpha_grad}, {});
      runner.Run(stream);
    } else if (data_format == "NHWC") {
      PADDLE_ENFORCE_EQ(x.dims().size(),
                        4,
                        phi::errors::InvalidArgument(
                            "Input dims of Prelu must be 4, but received %d",
                            x.dims().size()));

      auto x_dims = x.dims();
      std::vector<int> transformed_dim;
      transformed_dim.push_back(x_dims[0]);
      transformed_dim.push_back(x_dims[x_dims.size() - 1]);
      for (int i = 1; i < x_dims.size() - 1; ++i) {
        transformed_dim.push_back(x_dims[i]);
      }

      phi::DenseTensor transformed_x, transformed_out_grad, transformed_x_grad;
      transformed_x.Resize(phi::make_ddim(transformed_dim));
      dev_ctx.template Alloc<T>(&transformed_x);
      transformed_out_grad.Resize(phi::make_ddim(transformed_dim));
      dev_ctx.template Alloc<T>(&transformed_out_grad);
      transformed_x_grad.Resize(phi::make_ddim(transformed_dim));
      dev_ctx.template Alloc<T>(&transformed_x_grad);

      const auto& trans_runner1 =
          NpuOpRunner("TransData",
                      {x},
                      {transformed_x},
                      {{"src_format", std::string("NHWC")},
                       {"dst_format", std::string("NCHW")}});
      trans_runner1.Run(stream);

      const auto& trans_runner2 =
          NpuOpRunner("TransData",
                      {out_grad},
                      {transformed_out_grad},
                      {{"src_format", std::string("NHWC")},
                       {"dst_format", std::string("NCHW")}});
      trans_runner2.Run(stream);

      const auto& runner =
          NpuOpRunner("PReluGrad",
                      {transformed_out_grad, transformed_x, weight},
                      {transformed_x_grad, *alpha_grad},
                      {});
      runner.Run(stream);

      const auto& trans_runner3 =
          NpuOpRunner("TransData",
                      {transformed_x_grad},
                      {*x_grad},
                      {{"src_format", std::string("NCHW")},
                       {"dst_format", std::string("NHWC")}});
      trans_runner3.Run(stream);

    } else {
      phi::errors::Unimplemented(
          "Only NCHW and NHWC format is supported for input of PReluGrad "
          "kernel, "
          "but received %s .",
          data_format);
    }
  } else {
    phi::DenseTensor weight(alpha);
    const auto& runner = NpuOpRunner(
        "PReluGrad", {out_grad, x, weight}, {*x_grad, *alpha_grad}, {});
    runner.Run(stream);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(prelu,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::PReluKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(prelu_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::PReluGradKernel,
                          float,
                          phi::dtype::float16) {}
