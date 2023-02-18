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

  // NOTE(songkai05): PRelu op in CANN do not support element mode,
  // so we compute the result directly.
  // TODO(songkai05): PRelu op in CANN600 or higher versions would
  // return false results when weight is a 1-D tensor, we compute the
  // result directly until this bug is fixed.
  if (mode == "element" || mode == "channel") {
    phi::DenseTensor alpha_r(alpha);
    if (mode == "channel" && alpha.dims().size() != x.dims().size()) {
      std::vector<int64_t> reshape_dims(x.dims().size(), 1);
      if (data_format == "NCHW") {
        reshape_dims[1] = alpha.numel();
      } else {
        reshape_dims[x.dims().size() - 1] = alpha.numel();
      }
      alpha_r.Resize(phi::make_ddim(reshape_dims));
    }

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
        .AddInput(alpha_r)
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
  } else {
    if (x.dims().size() == 0) {
      std::vector<T> x_vct;
      TensorToVector(dev_ctx, x, dev_ctx, &x_vct);
      std::vector<T> alpha_vct;
      TensorToVector(dev_ctx, alpha, dev_ctx, &alpha_vct);
      auto val =
          x_vct[0] > static_cast<T>(0) ? x_vct[0] : alpha_vct[0] * x_vct[0];
      dev_ctx.template Alloc<T>(out);
      FillNpuTensorWithConstant<T>(out, dev_ctx, val);
    } else {
      const auto& runner = NpuOpRunner("PRelu", {x, alpha}, {*out}, {});
      runner.Run(stream);
    }
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
  auto alpha_dims = alpha.dims();

  if (mode == "element" || mode == "channel") {
    phi::DenseTensor alpha_r(alpha);
    if (mode == "channel" && alpha_dims.size() != x.dims().size()) {
      std::vector<int64_t> reshape_dims(x.dims().size(), 1);
      if (data_format == "NCHW") {
        reshape_dims[1] = alpha.numel();
      } else {
        reshape_dims[x.dims().size() - 1] = alpha.numel();
      }
      alpha_r.Resize(phi::make_ddim(reshape_dims));
      alpha_grad->Resize(phi::make_ddim(reshape_dims));
    }

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
        .AddInput(alpha_r)
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

    std::vector<int64_t> axis;
    if (mode == "element") {
      axis.push_back(0);
    } else {
      if (data_format == "NCHW") {
        axis.push_back(0);
        for (size_t i = 2; i < x.dims().size(); ++i) {
          axis.push_back(i);
        }
      } else {
        for (size_t i = 0; i < x.dims().size() - 1; ++i) {
          axis.push_back(i);
        }
      }
    }

    NpuOpRunner reduce_sum_runner;
    reduce_sum_runner.SetType("ReduceSum")
        .AddInput(alpha_broadcast)
        .AddInput(dev_ctx, std::move(axis))
        .AddOutput(*alpha_grad)
        .AddAttr("keep_dim", true)
        .Run(stream);
    alpha_grad->Resize(alpha_dims);
  } else {
    if (x.dims().size() == 0) {
      std::vector<T> x_vct;
      TensorToVector(dev_ctx, x, dev_ctx, &x_vct);
      std::vector<T> alpha_vct;
      TensorToVector(dev_ctx, alpha, dev_ctx, &alpha_vct);
      std::vector<T> out_grad_vct;
      TensorToVector(dev_ctx, out_grad, dev_ctx, &out_grad_vct);
      auto val = x_vct[0] > static_cast<T>(0) ? out_grad_vct[0]
                                              : out_grad_vct[0] * alpha_vct[0];
      FillNpuTensorWithConstant<T>(x_grad, dev_ctx, val);
    } else {
      phi::DenseTensor weight(alpha);
      const auto& runner = NpuOpRunner(
          "PReluGrad", {out_grad, x, weight}, {*x_grad, *alpha_grad}, {});
      runner.Run(stream);
    }
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
