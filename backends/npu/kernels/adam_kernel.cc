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
void AdamKernel(const Context& dev_ctx,
                const phi::DenseTensor& param,
                const phi::DenseTensor& grad,
                const phi::DenseTensor& learning_rate,
                const phi::DenseTensor& moment1,
                const phi::DenseTensor& moment2,
                const phi::DenseTensor& beta1_pow_in,
                const phi::DenseTensor& beta2_pow_in,
                const paddle::optional<phi::DenseTensor>& master_param,
                const paddle::optional<phi::DenseTensor>& skip_update,
                const phi::Scalar& beta1_in,
                const phi::Scalar& beta2_in,
                const phi::Scalar& epsilon_in,
                bool lazy_mode,
                int64_t min_row_size_to_use_multithread,
                bool multi_precision,
                bool use_global_beta_pow,
                phi::DenseTensor* param_out,
                phi::DenseTensor* moment1_out,
                phi::DenseTensor* moment2_out,
                phi::DenseTensor* beta1_pow_out,
                phi::DenseTensor* beta2_pow_out,
                phi::DenseTensor* master_param_out) {
  bool skip_update_ = false;
  if (skip_update.is_initialized()) {
    PADDLE_ENFORCE_EQ(skip_update->numel(),
                      1,
                      phi::errors::InvalidArgument(
                          "Input(SkipUpdate) size must be 1, but get %d",
                          skip_update->numel()));
    std::vector<bool> skip_update_vec;
    custom_kernel::TensorToVector(
        dev_ctx, *skip_update, dev_ctx, &skip_update_vec);
    skip_update_ = skip_update_vec[0];
  }

  // skip_update=true, just copy input to output, and TensorCopy will call
  // mutable_data
  if (skip_update_) {
    VLOG(4) << "Adam skip update";
    TensorCopy(dev_ctx, param, false, param_out);
    TensorCopy(dev_ctx, moment1, false, moment1_out);
    TensorCopy(dev_ctx, moment2, false, moment2_out);
    TensorCopy(dev_ctx, beta1_pow_in, false, beta1_pow_out);
    TensorCopy(dev_ctx, beta2_pow_in, false, beta2_pow_out);
    return;
  }

  phi::DenseTensor* beta1_pow = const_cast<phi::DenseTensor*>(&beta1_pow_in);
  phi::DenseTensor* beta2_pow = const_cast<phi::DenseTensor*>(&beta2_pow_in);

  VLOG(4) << "use_global_beta_pow:" << use_global_beta_pow;

  dev_ctx.template Alloc<T>(param_out);
  dev_ctx.template Alloc<T>(moment1_out);
  dev_ctx.template Alloc<T>(moment2_out);

  // NOTE(zhiqiu): beta1_pow and beta2_pow may on CPU and not transform
  // place.
  phi::DenseTensor beta1_pow_tmp;
  phi::DenseTensor beta2_pow_tmp;
  if (beta1_pow->place().GetType() == phi::AllocationType::CPU) {
    T beta1 = *beta1_pow->data<T>();
    beta1_pow_tmp.Resize({1});
    dev_ctx.template Alloc<T>(&beta1_pow_tmp);
    FillNpuTensorWithConstant<T>(&beta1_pow_tmp, dev_ctx, beta1);
    beta1_pow = &beta1_pow_tmp;
  }
  if (beta2_pow->place().GetType() == phi::AllocationType::CPU) {
    T beta2 = *beta2_pow->data<T>();
    beta2_pow_tmp.Resize({1});
    dev_ctx.template Alloc<T>(&beta2_pow_tmp);
    FillNpuTensorWithConstant<T>(&beta2_pow_tmp, dev_ctx, beta2);
    beta2_pow = &beta2_pow_tmp;
  }

  const phi::DenseTensor* beta1_tensor = nullptr;
  const phi::DenseTensor* beta2_tensor = nullptr;
  const phi::DenseTensor* epsilon_tensor = nullptr;

  phi::DenseTensor beta1_tmp;
  phi::DenseTensor beta2_tmp;
  phi::DenseTensor epsilon_tmp;
  phi::DenseTensorMeta meta = {phi::DataType::FLOAT32, {1}};
  beta1_tmp.set_meta(meta);
  beta2_tmp.set_meta(meta);
  epsilon_tmp.set_meta(meta);

  T beta1 = beta1_in.to<T>();
  dev_ctx.template Alloc<T>(&beta1_tmp);
  FillNpuTensorWithConstant<T>(&beta1_tmp, dev_ctx, beta1);
  beta1_tensor = &beta1_tmp;

  T beta2 = beta2_in.to<T>();
  dev_ctx.template Alloc<T>(&beta2_tmp);
  FillNpuTensorWithConstant<T>(&beta2_tmp, dev_ctx, beta2);
  beta2_tensor = &beta2_tmp;

  T epsilon = epsilon_in.to<T>();
  dev_ctx.template Alloc<T>(&epsilon_tmp);
  FillNpuTensorWithConstant<T>(&epsilon_tmp, dev_ctx, epsilon);
  epsilon_tensor = &epsilon_tmp;

  VLOG(3) << "beta1_pow.numel() : " << beta1_pow->numel()
          << "beta2_pow.numel() : " << beta2_pow->numel();
  VLOG(3) << "param.numel(): " << param.numel();

  auto stream = dev_ctx.stream();
  const auto& runner = NpuOpRunner("ApplyAdamD",
                                   {
                                       param,
                                       moment1,
                                       moment2,
                                       *beta1_pow,
                                       *beta2_pow,
                                       learning_rate,
                                       *beta1_tensor,
                                       *beta2_tensor,
                                       *epsilon_tensor,
                                       grad,
                                   },
                                   {
                                       *param_out,
                                       *moment1_out,
                                       *moment2_out,
                                   },
                                   {});
  runner.Run(stream);

  // NOTE(zhiqiu): ApplyAdamD updates params inplace, so
  // if param and param_out is not same, we need to do copy.
  if (param_out->data<T>() != param.data<T>()) {
    TensorCopy(dev_ctx, param, false, param_out);
  }
  if (moment1_out->data<T>() != moment1.data<T>()) {
    TensorCopy(dev_ctx, moment1, false, moment1_out);
  }
  if (moment2_out->data<T>() != moment2.data<T>()) {
    TensorCopy(dev_ctx, moment2, false, moment2_out);
  }
  if (!use_global_beta_pow) {
    dev_ctx.template Alloc<T>(beta1_pow_out);
    dev_ctx.template Alloc<T>(beta2_pow_out);
    const auto& runner_m1 =
        NpuOpRunner("Mul", {*beta1_pow, *beta1_tensor}, {*beta1_pow_out}, {});
    runner_m1.Run(stream);
    const auto& runner_m2 =
        NpuOpRunner("Mul", {*beta2_pow, *beta2_tensor}, {*beta2_pow_out}, {});
    runner_m2.Run(stream);
  }
}

template <typename T, typename Context>
void AdamwKernel(const Context& dev_ctx,
                 const phi::DenseTensor& param,
                 const phi::DenseTensor& grad,
                 const phi::DenseTensor& learning_rate,
                 const phi::DenseTensor& moment1,
                 const phi::DenseTensor& moment2,
                 const phi::DenseTensor& beta1_pow,
                 const phi::DenseTensor& beta2_pow,
                 const paddle::optional<phi::DenseTensor>& master_param,
                 const paddle::optional<phi::DenseTensor>& skip_update,
                 const phi::Scalar& beta1,
                 const phi::Scalar& beta2,
                 const phi::Scalar& epsilon,
                 float lr_ratio,
                 float coeff,
                 bool with_decay,
                 bool lazy_mode,
                 int64_t min_row_size_to_use_multithread,
                 bool multi_precision,
                 bool use_global_beta_pow,
                 phi::DenseTensor* param_out,
                 phi::DenseTensor* moment1_out,
                 phi::DenseTensor* moment2_out,
                 phi::DenseTensor* beta1_pow_out,
                 phi::DenseTensor* beta2_pow_out,
                 phi::DenseTensor* master_param_outs) {
  bool skip_update_ = false;
  if (skip_update.is_initialized()) {
    PADDLE_ENFORCE_EQ(skip_update->numel(),
                      1,
                      phi::errors::InvalidArgument(
                          "Input(SkipUpdate) size must be 1, but get %d",
                          skip_update->numel()));
    std::vector<bool> skip_update_vec;
    custom_kernel::TensorToVector(
        dev_ctx, *skip_update, dev_ctx, &skip_update_vec);
    skip_update_ = skip_update_vec[0];
  }

  VLOG(3) << "Skip update" << skip_update_;

  if (!skip_update_ && with_decay) {
    phi::DenseTensor one;
    phi::DenseTensor decay;
    phi::DenseTensor tmp;
    phi::DenseTensorMeta meta = {phi::DataType::FLOAT32, {1}};
    one.set_meta(meta);
    decay.set_meta(meta);
    tmp.set_meta(meta);

    dev_ctx.template Alloc<float>(&tmp);
    dev_ctx.template Alloc<float>(&one);
    dev_ctx.template Alloc<float>(&decay);

    FillNpuTensorWithConstant<float>(&one, dev_ctx, 1.0f);
    NPUAttributeMap attr_input = {{"value", coeff}};

    auto stream = dev_ctx.stream();

    const auto& runner1 =
        NpuOpRunner("Muls", {learning_rate}, {tmp}, attr_input);
    runner1.Run(stream);

    const auto& runner2 = NpuOpRunner("Sub", {one, tmp}, {decay}, {});
    runner2.Run(stream);

    // Master Parma is not supported on npu
    if (master_param.is_initialized()) {
      PADDLE_THROW(
          phi::errors::Unimplemented("Master Parma is not supported on npu"));
    } else {
      dev_ctx.template Alloc<T>(param_out);

      const auto& runner = NpuOpRunner(
          "Mul", {param, decay}, {*const_cast<phi::DenseTensor*>(&param)}, {});
      runner.Run(stream);
    }
  }

  custom_kernel::AdamKernel<T, Context>(dev_ctx,
                                        param,
                                        grad,
                                        learning_rate,
                                        moment1,
                                        moment2,
                                        beta1_pow,
                                        beta2_pow,
                                        master_param,
                                        skip_update,
                                        beta1,
                                        beta2,
                                        epsilon,
                                        lazy_mode,
                                        min_row_size_to_use_multithread,
                                        multi_precision,
                                        use_global_beta_pow,
                                        param_out,
                                        moment1_out,
                                        moment2_out,
                                        beta1_pow_out,
                                        beta2_pow_out,
                                        master_param_outs);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(adam,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::AdamKernel,
                          phi::dtype::float16,
                          float,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(adamw,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::AdamwKernel,
                          phi::dtype::float16,
                          float,
                          double) {}
