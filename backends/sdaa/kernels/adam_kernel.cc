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

#include <cmath>
#include <iostream>

#include "kernels/funcs/sdaa_baseop.h"
#include "sdcops.h"   // NOLINT
#include "tecodnn.h"  // NOLINT
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
                const paddle::optional<phi::DenseTensor>& master_param,  // fp32
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
  VLOG(4) << "call sdaa AdamKernel";
  using MPDType = typename MPTypeTrait<T>::Type;  // fp32

  if (isEnvEnable("HIGH_PERFORMANCE_CONV") &&
      grad.storage_properties_initialized()) {
    // update conv_filter
    SDAAStorageProperties grad_properties =
        grad.storage_properties<SDAAStorageProperties>();
    PADDLE_ENFORCE_EQ(
        &param,
        param_out,
        phi::errors::InvalidArgument(
            "HIGH_PERFORMANCE_CONV only support param inplace update"));
    PADDLE_ENFORCE_EQ(
        &moment1,
        moment1_out,
        phi::errors::InvalidArgument(
            "HIGH_PERFORMANCE_CONV only support moment1 inplace update"));
    PADDLE_ENFORCE_EQ(
        &moment2,
        moment2_out,
        phi::errors::InvalidArgument(
            "HIGH_PERFORMANCE_CONV only support moment2 inplace update"));

    if (!moment1.storage_properties_initialized()) {
      sdaa_ops::swapTensorData(dev_ctx, moment1, grad_properties);
    }
    if (!moment2.storage_properties_initialized()) {
      sdaa_ops::swapTensorData(dev_ctx, moment2, grad_properties);
    }
    if (!param.storage_properties_initialized()) {
      sdaa_ops::swapTensorData(dev_ctx, param, grad_properties);
    }
  }
  std::vector<bool> cpu_if_skip = {false};

  if (skip_update.is_initialized()) {
    const phi::DenseTensor& skip_update_tmp =
        static_cast<const phi::DenseTensor&>(*skip_update);
    TensorToVector(dev_ctx, skip_update_tmp, dev_ctx, &cpu_if_skip);
    PADDLE_ENFORCE_EQ(skip_update->numel(),
                      1,
                      phi::errors::InvalidArgument(
                          "Input(SkipUpdate) size must be 1, but get %d",
                          skip_update->numel()));
  }

  if (static_cast<bool>(cpu_if_skip[0])) {
    dev_ctx.template Alloc<T>(beta1_pow_out);
    dev_ctx.template Alloc<T>(beta2_pow_out);
    VLOG(4) << "Adam skip update";
    TensorCopy(dev_ctx, param, false, param_out);
    TensorCopy(dev_ctx, moment1, false, moment1_out);
    TensorCopy(dev_ctx, moment2, false, moment2_out);
    TensorCopy(dev_ctx, beta1_pow_in, false, beta1_pow_out);
    TensorCopy(dev_ctx, beta2_pow_in, false, beta2_pow_out);
    return;
  }

  dev_ctx.template Alloc<T>(param_out);
  dev_ctx.template Alloc<MPDType>(moment1_out);
  dev_ctx.template Alloc<MPDType>(moment2_out);
  VLOG(4) << "Adam not skip update";
  VLOG(4) << "use_global_beta_pow:" << use_global_beta_pow;

  PADDLE_ENFORCE_EQ(
      moment1.dims(),
      grad.dims(),
      phi::errors::InvalidArgument("moment1 and grad should have same dims"));

  int device_scale = 0;
  if (beta1_pow_in.place().GetType() == phi::AllocationType::CPU) {
    VLOG(6) << "beta_pow_in is in CPU!";
  } else {
    VLOG(6) << "beta_pow_in is in SDAA!";
    device_scale += 1;
  }

  phi::DenseTensor* beta1_pow = const_cast<phi::DenseTensor*>(&beta1_pow_in);
  phi::DenseTensor* beta2_pow = const_cast<phi::DenseTensor*>(&beta2_pow_in);
  phi::DenseTensor* lr = const_cast<phi::DenseTensor*>(&learning_rate);
  phi::DenseTensor* grad_in = const_cast<phi::DenseTensor*>(&grad);
  float beta1 = beta1_in.to<float>();  // cpu
  float beta2 = beta2_in.to<float>();  // cpu
  float epsilon = epsilon_in.to<float>();
  int n_total = static_cast<int>(param.numel());

  phi::DenseTensor param_in = multi_precision ? master_param.get() : param;
  phi::DenseTensor* moment1_in = const_cast<phi::DenseTensor*>(&moment1);
  phi::DenseTensor* moment2_in = const_cast<phi::DenseTensor*>(&moment2);

  // init beta_pow_out in case beta_pow_out is NULL when use_global_beta_pow is
  // true
  float* b1_out = beta1_pow->data<MPDType>();
  float* b2_out = beta2_pow->data<MPDType>();
  if (!use_global_beta_pow) {
    device_scale += 2;
    if (!beta1_pow_out->IsSharedWith(beta1_pow_in)) {
      if (device_scale & 1) {
        dev_ctx.template Alloc<MPDType>(beta1_pow_out);
        dev_ctx.template Alloc<MPDType>(beta2_pow_out);
      } else {
        dev_ctx.template HostAlloc<MPDType>(beta1_pow_out);
        dev_ctx.template HostAlloc<MPDType>(beta2_pow_out);
      }
    }
    b1_out = beta1_pow_out->data<float>();
    b2_out = beta2_pow_out->data<float>();
  }
  void* A[4] = {
      grad_in->data(), param_in.data(), moment1_in->data(), moment2_in->data()};
  phi::DenseTensor param_out_;
  if (multi_precision) {
    param_out_ = *master_param_out;
  } else {
    param_out_ = *param_out;
  }
  void* B[3] = {param_out_.data(), moment1_out->data(), moment2_out->data()};
  sdaaStream_t custom_stream = GetStreamFromCTX(dev_ctx);
  DataTypes_t dtypes[3] = {sdaa_ops::ToExtendDataType(param_in.dtype()),
                           sdaa_ops::ToExtendDataType(grad_in->dtype()),
                           sdaa_ops::ToExtendDataType(moment1_in->dtype())};
  TCUS_CHECK(sdcops::pd_adam_out(n_total,
                                 A,
                                 B,
                                 beta1,
                                 beta2,
                                 beta1_pow->data<float>(),
                                 beta2_pow->data<float>(),
                                 epsilon,
                                 lr->data<float>(),
                                 device_scale,
                                 b1_out,
                                 b2_out,
                                 dtypes,
                                 custom_stream));
  if (multi_precision) {
    sdaa_ops::doCastTensor(dev_ctx, param_out_, param_out);
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
  VLOG(4) << "call sdaa AdamwKernel";
  if (isEnvEnable("HIGH_PERFORMANCE_CONV") &&
      grad.storage_properties_initialized()) {
    VLOG(1) << "AdamW begin to update conv's filter at " << &grad;
    // update conv_filter
    SDAAStorageProperties grad_properties =
        grad.storage_properties<SDAAStorageProperties>();
    PADDLE_ENFORCE_EQ(
        &param,
        param_out,
        phi::errors::InvalidArgument(
            "HIGH_PERFORMANCE_CONV only support param inplace update"));
    PADDLE_ENFORCE_EQ(
        &moment1,
        moment1_out,
        phi::errors::InvalidArgument(
            "HIGH_PERFORMANCE_CONV only support moment1 inplace update"));
    PADDLE_ENFORCE_EQ(
        &moment2,
        moment2_out,
        phi::errors::InvalidArgument(
            "HIGH_PERFORMANCE_CONV only support moment2 inplace update"));

    if (!moment1.storage_properties_initialized()) {
      sdaa_ops::swapTensorData(dev_ctx, moment1, grad_properties);
    }
    if (!moment2.storage_properties_initialized()) {
      sdaa_ops::swapTensorData(dev_ctx, moment2, grad_properties);
    }
    if (!param.storage_properties_initialized()) {
      sdaa_ops::swapTensorData(dev_ctx, param, grad_properties);
    }
  }

  using MPDType = typename MPTypeTrait<T>::Type;
  std::vector<bool> cpu_if_skip = {false};

  if (skip_update.is_initialized()) {
    const phi::DenseTensor& skip_update_tmp =
        static_cast<const phi::DenseTensor&>(*skip_update);
    TensorToVector(dev_ctx, skip_update_tmp, dev_ctx, &cpu_if_skip);
    PADDLE_ENFORCE_EQ(skip_update->numel(),
                      1,
                      phi::errors::InvalidArgument(
                          "Input(SkipUpdate) size must be 1, but get %d",
                          skip_update->numel()));
  }

  if (static_cast<bool>(cpu_if_skip[0])) {
    dev_ctx.template Alloc<float>(beta1_pow_out);
    dev_ctx.template Alloc<float>(beta2_pow_out);
    VLOG(4) << "Adamw skip update";
    TensorCopy(dev_ctx, param, false, param_out);
    TensorCopy(dev_ctx, moment1, false, moment1_out);
    TensorCopy(dev_ctx, moment2, false, moment2_out);
    TensorCopy(dev_ctx, beta1_pow, false, beta1_pow_out);
    TensorCopy(dev_ctx, beta2_pow, false, beta2_pow_out);
    return;
  }
  dev_ctx.template Alloc<T>(param_out);
  dev_ctx.template Alloc<MPDType>(moment1_out);
  dev_ctx.template Alloc<MPDType>(moment2_out);
  VLOG(4) << "Adamw not skip update";
  VLOG(4) << "use_global_beta_pow:" << use_global_beta_pow;
  PADDLE_ENFORCE_EQ(
      moment1.dims(),
      grad.dims(),
      phi::errors::InvalidArgument("moment1 and grad should have same dims"));

  int device_scale = 0;
  if (beta1_pow.place().GetType() == phi::AllocationType::CPU) {
    VLOG(6) << "beta_pow_in is in CPU!";
  } else {
    VLOG(6) << "beta_pow_in is in SDAA!";
    device_scale += 1;
  }
  float beta1_ = beta1.to<float>();  // cpu
  float beta2_ = beta2.to<float>();  // cpu
  float epsilon_ = epsilon.to<float>();
  phi::DenseTensor* beta1_pow_ = const_cast<phi::DenseTensor*>(&beta1_pow);
  phi::DenseTensor* beta2_pow_ = const_cast<phi::DenseTensor*>(&beta2_pow);
  phi::DenseTensor* lr = const_cast<phi::DenseTensor*>(&learning_rate);
  phi::DenseTensor* grad_in = const_cast<phi::DenseTensor*>(&grad);
  phi::DenseTensor param_in = multi_precision ? master_param.get() : param;
  phi::DenseTensor* moment1_in = const_cast<phi::DenseTensor*>(&moment1);
  phi::DenseTensor* moment2_in = const_cast<phi::DenseTensor*>(&moment2);
  int n_total = static_cast<int>(param.numel());

  float* b1_out = beta1_pow_->data<MPDType>();
  float* b2_out = beta2_pow_->data<MPDType>();
  if (!use_global_beta_pow) {
    device_scale += 2;
    if (!beta1_pow_out->IsSharedWith(beta1_pow)) {
      if (device_scale & 1) {
        dev_ctx.template Alloc<MPDType>(beta1_pow_out);
        dev_ctx.template Alloc<MPDType>(beta2_pow_out);
      } else {
        dev_ctx.template HostAlloc<MPDType>(beta1_pow_out);
        dev_ctx.template HostAlloc<MPDType>(beta2_pow_out);
      }
    }
    b1_out = beta1_pow_out->data<float>();
    b2_out = beta2_pow_out->data<float>();
  }
  void* A[4] = {
      grad_in->data(), param_in.data(), moment1_in->data(), moment2_in->data()};
  phi::DenseTensor param_out_;
  if (multi_precision) {
    param_out_ = *master_param_outs;
  } else {
    param_out_ = *param_out;
  }
  void* B[3] = {param_out_.data(), moment1_out->data(), moment2_out->data()};
  sdaaStream_t custom_stream = GetStreamFromCTX(dev_ctx);
  DataTypes_t dtypes[3] = {sdaa_ops::ToExtendDataType(param_in.dtype()),
                           sdaa_ops::ToExtendDataType(grad_in->dtype()),
                           sdaa_ops::ToExtendDataType(moment1_in->dtype())};
  TCUS_CHECK(sdcops::pd_adamw_out(n_total,
                                  A,
                                  B,
                                  beta1_,
                                  beta2_,
                                  beta1_pow_->data<float>(),
                                  beta2_pow_->data<float>(),
                                  epsilon_,
                                  lr->data<float>(),
                                  device_scale,
                                  lr_ratio,
                                  coeff,
                                  b1_out,
                                  b2_out,
                                  dtypes,
                                  custom_stream));
  if (multi_precision) {
    sdaa_ops::doCastTensor(dev_ctx, param_out_, param_out);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(adam,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::AdamKernel,
                          float,
                          phi::dtype::float16) {
  // Skip beta1_pow, beta2_pow, skip_update data transform
  kernel->InputAt(5).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(6).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(8).SetBackend(phi::Backend::ALL_BACKEND);
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(4).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(5).SetDataType(phi::DataType::FLOAT32);
  }
  kernel->OutputAt(3).SetBackend(phi::Backend::UNDEFINED);
  kernel->OutputAt(4).SetBackend(phi::Backend::UNDEFINED);
}

PD_REGISTER_PLUGIN_KERNEL(adamw,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::AdamwKernel,
                          float,
                          phi::dtype::float16) {
  // Skip beta1_pow, beta2_pow, skip_update data transform
  kernel->InputAt(5).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(6).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(8).SetBackend(phi::Backend::ALL_BACKEND);
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(4).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(5).SetDataType(phi::DataType::FLOAT32);
  }
  kernel->OutputAt(3).SetBackend(phi::Backend::UNDEFINED);
  kernel->OutputAt(4).SetBackend(phi::Backend::UNDEFINED);
}
