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
void KLDivLossKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& label,
                     const std::string& reduction,
                     phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  auto stream = dev_ctx.stream();

  if ("none" == reduction) {
    // log(label)
    phi::DenseTensor ones_tensor;
    ones_tensor.Resize(label.dims());
    dev_ctx.template Alloc<T>(&ones_tensor);
    const auto& ones_runner =
        NpuOpRunner("OnesLike", {label}, {ones_tensor}, {});
    ones_runner.Run(stream);

    phi::DenseTensor sub_tensor;
    sub_tensor.Resize(label.dims());
    dev_ctx.template Alloc<T>(&sub_tensor);
    const auto& sub_runner =
        NpuOpRunner("Sub", {label, ones_tensor}, {sub_tensor}, {});
    sub_runner.Run(stream);

    phi::DenseTensor log_target;
    log_target.Resize(label.dims());
    dev_ctx.template Alloc<T>(&log_target);
    const auto& log_runner =
        NpuOpRunner("Log1p", {sub_tensor}, {log_target}, {});
    log_runner.Run(stream);

    // log(label) - input
    const auto& sub_runner2 = NpuOpRunner("Sub", {log_target, x}, {*out}, {});
    sub_runner2.Run(stream);

    // label * (log(label) - input)
    phi::DenseTensor min_value, max_value;
    min_value.Resize({1});
    max_value.Resize({1});
    dev_ctx.template Alloc<T>(&min_value);
    dev_ctx.template Alloc<T>(&max_value);
    FillNpuTensorWithConstant(&min_value, dev_ctx, static_cast<T>(0));
    FillNpuTensorWithConstant(
        &max_value, dev_ctx, std::numeric_limits<T>::max());

    phi::DenseTensor cliped_target;
    cliped_target.Resize(label.dims());
    dev_ctx.template Alloc<T>(&cliped_target);
    const auto& clip_runner = NpuOpRunner(
        "ClipByValue", {label, min_value, max_value}, {cliped_target}, {});
    clip_runner.Run(stream);

    const auto& mul_runner =
        NpuOpRunner("Mul", {*out, cliped_target}, {*out}, {});
    mul_runner.Run(stream);
  } else if ("batchmean" == reduction || "sum" == reduction) {
    const auto& runner =
        NpuOpRunner("KLDiv", {x, label}, {*out}, {{"reduction", reduction}});
    runner.Run(stream);
  } else if ("mean" == reduction) {
    const auto& runner = NpuOpRunner(
        "KLDiv", {x, label}, {*out}, {{"reduction", std::string("sum")}});
    runner.Run(stream);

    const int numel = x.numel();
    const auto& muls_runner = NpuOpRunner(
        "Muls", {*out}, {*out}, {{"value", static_cast<float>(1.0 / numel)}});
    muls_runner.Run(stream);
  }
}

template <typename T, typename Context>
void KLDivLossGradKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const phi::DenseTensor& label,
                         const phi::DenseTensor& d_out,
                         const std::string& reduction,
                         phi::DenseTensor* d_x) {
  dev_ctx.template Alloc<T>(d_x);

  auto stream = dev_ctx.stream();

  phi::DenseTensor d_out_transformed;
  if ("none" == reduction) {
    d_out_transformed = d_out;
  } else {
    d_out_transformed.Resize(d_x->dims());
    dev_ctx.template Alloc<T>(&d_out_transformed);

    NpuOpRunner broadcast_runner;
    broadcast_runner.SetType("BroadcastTo");
    broadcast_runner.AddInput(d_out);
    broadcast_runner.AddInput(dev_ctx, phi::vectorize<int>(d_x->dims()));
    broadcast_runner.AddOutput(d_out_transformed);
    broadcast_runner.Run(stream);
  }
  phi::DenseTensor min_value, max_value;
  min_value.Resize({1});
  max_value.Resize({1});
  dev_ctx.template Alloc<T>(&min_value);
  dev_ctx.template Alloc<T>(&max_value);
  FillNpuTensorWithConstant(&min_value, dev_ctx, static_cast<T>(0));
  FillNpuTensorWithConstant(&max_value, dev_ctx, std::numeric_limits<T>::max());

  phi::DenseTensor cliped_label;
  cliped_label.Resize(label.dims());
  dev_ctx.template Alloc<T>(&cliped_label);
  const auto& clip_runner = NpuOpRunner(
      "ClipByValue", {label, min_value, max_value}, {cliped_label}, {});
  clip_runner.Run(stream);

  const auto& mul_runner =
      NpuOpRunner("Mul", {cliped_label, d_out_transformed}, {*d_x}, {});
  mul_runner.Run(stream);

  float k = -1.0f;

  if ("mean" == reduction) {
    k = static_cast<float>(-1.0 / d_x->numel());
  } else if ("batchmean" == reduction) {
    k = static_cast<float>(-1.0 / d_x->dims()[0]);
  }

  const auto& muls_runner = NpuOpRunner("Muls", {*d_x}, {*d_x}, {{"value", k}});
  muls_runner.Run(stream);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(kldiv_loss,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::KLDivLossKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(kldiv_loss_grad,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::KLDivLossGradKernel,
                          float,
                          phi::dtype::float16) {}
