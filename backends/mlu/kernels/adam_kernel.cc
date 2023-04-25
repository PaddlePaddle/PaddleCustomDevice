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

#include "kernels/funcs/mlu_baseop.h"
#include "kernels/funcs/mlu_funcs.h"

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
    TensorToVector(dev_ctx, *skip_update, dev_ctx, &skip_update_vec);
    dev_ctx.Wait();
    skip_update_ = skip_update_vec[0];
  }

  // skip_update_=true, just copy input to output asynchronously on the device,
  // and TensorCopy will call mutable_data
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

  *param_out = param;
  *moment1_out = moment1;
  *moment2_out = moment2;

  phi::DenseTensor beta1_pow_tmp;
  phi::DenseTensor beta2_pow_tmp;
  if (beta1_pow->place().GetType() == phi::AllocationType::CPU) {
    T beta1 = *beta1_pow->data<T>();
    beta1_pow_tmp.Resize({1});
    dev_ctx.template Alloc<T>(&beta1_pow_tmp);
    FillMLUTensorWithHostValue(dev_ctx, beta1, &beta1_pow_tmp);
    beta1_pow = &beta1_pow_tmp;
  }
  if (beta2_pow->place().GetType() == phi::AllocationType::CPU) {
    T beta2 = *beta2_pow->data<T>();
    beta2_pow_tmp.Resize({1});
    dev_ctx.template Alloc<T>(&beta2_pow_tmp);
    FillMLUTensorWithHostValue(dev_ctx, beta2, &beta2_pow_tmp);
    beta2_pow = &beta2_pow_tmp;
  }

  VLOG(3) << "beta1_pow->numel() : " << beta1_pow->numel()
          << "beta2_pow->numel() : " << beta2_pow->numel();
  VLOG(3) << "param.numel(): " << param.numel();

  PADDLE_ENFORCE_EQ(beta1_pow_out->numel(),
                    1,
                    phi::errors::InvalidArgument(
                        "beta1 pow output size should be 1, but received "
                        "value is:%d.",
                        beta1_pow_out->numel()));

  PADDLE_ENFORCE_EQ(beta2_pow_out->numel(),
                    1,
                    phi::errors::InvalidArgument(
                        "beta2 pow output size should be 1, but received "
                        "value is:%d.",
                        beta2_pow_out->numel()));

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
  FillMLUTensorWithHostValue<T>(dev_ctx, beta1, &beta1_tmp);
  beta1_tensor = &beta1_tmp;

  T beta2 = beta2_in.to<T>();
  dev_ctx.template Alloc<T>(&beta2_tmp);
  FillMLUTensorWithHostValue<T>(dev_ctx, beta2, &beta2_tmp);
  beta2_tensor = &beta2_tmp;

  T epsilon = epsilon_in.to<T>();
  dev_ctx.template Alloc<T>(&epsilon_tmp);
  FillMLUTensorWithHostValue<T>(dev_ctx, epsilon, &epsilon_tmp);
  epsilon_tensor = &epsilon_tmp;

  MLUCnnlTensorDesc param_desc(param);
  MLUCnnlTensorDesc mom1_desc(moment1);
  MLUCnnlTensorDesc mom2_desc(moment2);
  MLUCnnlTensorDesc grad_desc(grad);
  MLUCnnl::ApplyAdam(dev_ctx,
                     param_desc.get(),
                     GetBasePtr(param_out),
                     mom1_desc.get(),
                     GetBasePtr(moment1_out),
                     mom2_desc.get(),
                     GetBasePtr(moment2_out),
                     grad_desc.get(),
                     GetBasePtr(&grad),
                     GetBasePtr(&learning_rate),
                     GetBasePtr(beta1_tensor),
                     GetBasePtr(beta2_tensor),
                     GetBasePtr(beta1_pow),
                     GetBasePtr(beta2_pow),
                     GetBasePtr(epsilon_tensor),
                     /*use_nesterov*/ false);

  if (!use_global_beta_pow) {
    dev_ctx.template Alloc<T>(beta1_pow_out);
    dev_ctx.template Alloc<T>(beta2_pow_out);

    MLUCnnlTensorDesc beta1_desc(*beta1_tensor);
    MLUCnnlOpTensorDesc mul_op_desc(
        CNNL_OP_TENSOR_MUL, ToCnnlDataType<T>(), CNNL_NOT_PROPAGATE_NAN);

    MLUCnnl::OpTensor(dev_ctx,
                      mul_op_desc.get(),
                      beta1_desc.get(),
                      GetBasePtr(beta1_pow),
                      beta1_desc.get(),
                      GetBasePtr(beta1_tensor),
                      beta1_desc.get(),
                      GetBasePtr(beta1_pow_out),
                      ToCnnlDataType<T>());

    MLUCnnl::OpTensor(dev_ctx,
                      mul_op_desc.get(),
                      beta1_desc.get(),
                      GetBasePtr(beta2_pow),
                      beta1_desc.get(),
                      GetBasePtr(beta2_tensor),
                      beta1_desc.get(),
                      GetBasePtr(beta2_pow_out),
                      ToCnnlDataType<T>());
  }
}

template <typename T, typename Context>
void AdamWKernel(const Context& dev_ctx,
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
    TensorToVector(dev_ctx, *skip_update, dev_ctx, &skip_update_vec);
    dev_ctx.Wait();
    skip_update_ = skip_update_vec[0];
  }

  VLOG(3) << "Skip update" << skip_update_ << ", With decay: " << with_decay;

  if (!skip_update_ && with_decay) {
    if (master_param.is_initialized()) {
      PADDLE_THROW(
          phi::errors::Unimplemented("Master Param is not supported on MLU"));
    } else {
      // update param with decay coeff: mul(-1 * lr, coeff * param) + param
      MLUCnnlTensorDesc lr_desc(learning_rate);
      MLUCnnlTensorDesc param_desc(param);
      MLUCnnlOpTensorDesc mul_op_desc(
          CNNL_OP_TENSOR_MUL, ToCnnlDataType<T>(), CNNL_NOT_PROPAGATE_NAN);

      MLUCnnl::OpTensor(dev_ctx,
                        mul_op_desc.get(),
                        lr_desc.get(),
                        GetBasePtr(&learning_rate),
                        param_desc.get(),
                        GetBasePtr(&param),
                        param_desc.get(),
                        const_cast<void*>(GetBasePtr(&param)),
                        ToCnnlDataType<T>(),
                        /*alpha1*/ -1.f,
                        /*alpha2*/ coeff,
                        /*beta*/ 1.f);
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

template <typename T, typename Context>
void MergedAdamKernel(
    const Context& dev_ctx,
    const std::vector<const phi::DenseTensor*>& param,
    const std::vector<const phi::DenseTensor*>& grad,
    const std::vector<const phi::DenseTensor*>& learning_rate,
    const std::vector<const phi::DenseTensor*>& moment1,
    const std::vector<const phi::DenseTensor*>& moment2,
    const std::vector<const phi::DenseTensor*>& beta1_pow,
    const std::vector<const phi::DenseTensor*>& beta2_pow,
    const paddle::optional<std::vector<const phi::DenseTensor*>>& master_param,
    const phi::Scalar& beta1,
    const phi::Scalar& beta2,
    const phi::Scalar& epsilon,
    bool multi_precision,
    bool use_global_beta_pow,
    std::vector<phi::DenseTensor*> param_out,
    std::vector<phi::DenseTensor*> moment1_out,
    std::vector<phi::DenseTensor*> moment2_out,
    std::vector<phi::DenseTensor*> beta1_pow_out,
    std::vector<phi::DenseTensor*> beta2_pow_out,
    std::vector<phi::DenseTensor*> master_param_out) {
  size_t param_num = param.size();
  PADDLE_ENFORCE_EQ(param_num,
                    grad.size(),
                    phi::errors::InvalidArgument(
                        "The size of Input(grad) must be equal to "
                        "Input(param), but got the size of Input(grad) "
                        "is %d, the size of Input(param) is %d.",
                        grad.size(),
                        param_num));
  PADDLE_ENFORCE_EQ(
      param_num,
      learning_rate.size(),
      phi::errors::InvalidArgument(
          "The size of Input(learning_rate) must be equal to "
          "Input(param), but got the size of Input(learning_rate) "
          "is %d, the size of Input(param) is %d.",
          learning_rate.size(),
          param_num));
  PADDLE_ENFORCE_EQ(param_num,
                    moment1.size(),
                    phi::errors::InvalidArgument(
                        "The size of Input(moment1) must be equal to "
                        "Input(param), but got the size of Input(moment1) "
                        "is %d, the size of Input(param) is %d.",
                        moment1.size(),
                        param_num));
  PADDLE_ENFORCE_EQ(param_num,
                    moment2.size(),
                    phi::errors::InvalidArgument(
                        "The size of Input(moment2) must be equal to "
                        "Input(param), but got the size of Input(moment2) "
                        "is %d, the size of Input(param) is %d.",
                        moment2.size(),
                        param_num));
  PADDLE_ENFORCE_EQ(param_num,
                    beta1_pow.size(),
                    phi::errors::InvalidArgument(
                        "The size of Input(beta1_pow) must be equal to "
                        "Input(param), but got the size of Input(beta1_pow) "
                        "is %d, the size of Input(param) is %d.",
                        beta1_pow.size(),
                        param_num));
  PADDLE_ENFORCE_EQ(param_num,
                    beta2_pow.size(),
                    phi::errors::InvalidArgument(
                        "The size of Input(beta2_pow) must be equal to "
                        "Input(param), but got the size of Input(beta2_pow) "
                        "is %d, the size of Input(param) is %d.",
                        beta2_pow.size(),
                        param_num));

  VLOG(4) << "use_global_beta_pow:" << use_global_beta_pow;

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

  T beta1_ = beta1.to<T>();
  dev_ctx.template Alloc<T>(&beta1_tmp);
  FillMLUTensorWithHostValue<T>(dev_ctx, beta1_, &beta1_tmp);
  beta1_tensor = &beta1_tmp;

  T beta2_ = beta2.to<T>();
  dev_ctx.template Alloc<T>(&beta2_tmp);
  FillMLUTensorWithHostValue<T>(dev_ctx, beta2_, &beta2_tmp);
  beta2_tensor = &beta2_tmp;

  T epsilon_ = epsilon.to<T>();
  dev_ctx.template Alloc<T>(&epsilon_tmp);
  FillMLUTensorWithHostValue<T>(dev_ctx, epsilon_, &epsilon_tmp);
  epsilon_tensor = &epsilon_tmp;

  for (size_t idx = 0; idx < param_num; idx++) {
    VLOG(4) << "[MergedAdam] loop: " << idx;
    *param_out[idx] = *param[idx];
    *moment1_out[idx] = *moment1[idx];
    *moment2_out[idx] = *moment2[idx];

    phi::DenseTensor* beta1_pow_tensor =
        const_cast<phi::DenseTensor*>(beta1_pow[idx]);
    phi::DenseTensor* beta2_pow_tensor =
        const_cast<phi::DenseTensor*>(beta2_pow[idx]);
    phi::DenseTensor beta1_pow_tmp;
    phi::DenseTensor beta2_pow_tmp;
    if (beta1_pow_tensor->place().GetType() == phi::AllocationType::CPU) {
      T beta1_pow_ = *beta1_pow_tensor->data<T>();
      beta1_pow_tmp.Resize({1});
      dev_ctx.template Alloc<T>(&beta1_pow_tmp);
      FillMLUTensorWithHostValue(dev_ctx, beta1_pow_, &beta1_pow_tmp);
      beta1_pow_tensor = &beta1_pow_tmp;
    }
    if (beta2_pow_tensor->place().GetType() == phi::AllocationType::CPU) {
      T beta2_pow_ = *beta2_pow_tensor->data<T>();
      beta2_pow_tmp.Resize({1});
      dev_ctx.template Alloc<T>(&beta2_pow_tmp);
      FillMLUTensorWithHostValue(dev_ctx, beta2_pow_, &beta2_pow_tmp);
      beta2_pow_tensor = &beta2_pow_tmp;
    }

    VLOG(3) << "beta1_pow_tensor.numel() : " << beta1_pow_tensor->numel()
            << "beta2_pow_tensor.numel() : " << beta2_pow_tensor->numel();
    VLOG(3) << "param.numel(): " << param[idx]->numel();
    PADDLE_ENFORCE_EQ(beta1_pow_out[idx]->numel(),
                      1,
                      phi::errors::InvalidArgument(
                          "beta1 pow output size should be 1, but received "
                          "value is:%d.",
                          beta1_pow_out[idx]->numel()));

    PADDLE_ENFORCE_EQ(beta2_pow_out[idx]->numel(),
                      1,
                      phi::errors::InvalidArgument(
                          "beta2 pow output size should be 1, but received "
                          "value is:%d.",
                          beta2_pow_out[idx]->numel()));
    MLUCnnlTensorDesc param_desc(*param[idx]);
    MLUCnnlTensorDesc mom1_desc(*moment1[idx]);
    MLUCnnlTensorDesc mom2_desc(*moment2[idx]);
    MLUCnnlTensorDesc grad_desc(*grad[idx]);
    MLUCnnl::ApplyAdam(dev_ctx,
                       param_desc.get(),
                       GetBasePtr(param_out[idx]),
                       mom1_desc.get(),
                       GetBasePtr(moment1_out[idx]),
                       mom2_desc.get(),
                       GetBasePtr(moment2_out[idx]),
                       grad_desc.get(),
                       GetBasePtr(grad[idx]),
                       GetBasePtr(learning_rate[idx]),
                       GetBasePtr(beta1_tensor),
                       GetBasePtr(beta2_tensor),
                       GetBasePtr(beta1_pow_tensor),
                       GetBasePtr(beta2_pow_tensor),
                       GetBasePtr(epsilon_tensor),
                       /*use_nesterov*/ false);
    if (!use_global_beta_pow) {
      dev_ctx.template Alloc<T>(beta1_pow_out[idx]);
      dev_ctx.template Alloc<T>(beta2_pow_out[idx]);

      MLUCnnlTensorDesc beta1_desc(*beta1_tensor);
      MLUCnnlOpTensorDesc mul_op_desc(
          CNNL_OP_TENSOR_MUL, ToCnnlDataType<T>(), CNNL_NOT_PROPAGATE_NAN);

      MLUCnnl::OpTensor(dev_ctx,
                        mul_op_desc.get(),
                        beta1_desc.get(),
                        GetBasePtr(beta1_pow_tensor),
                        beta1_desc.get(),
                        GetBasePtr(beta1_tensor),
                        beta1_desc.get(),
                        GetBasePtr(beta1_pow_out[idx]),
                        ToCnnlDataType<T>());

      MLUCnnl::OpTensor(dev_ctx,
                        mul_op_desc.get(),
                        beta1_desc.get(),
                        GetBasePtr(beta2_pow_tensor),
                        beta1_desc.get(),
                        GetBasePtr(beta2_tensor),
                        beta1_desc.get(),
                        GetBasePtr(beta2_pow_out[idx]),
                        ToCnnlDataType<T>());
    }
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(adam,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::AdamKernel,
                          phi::dtype::float16,
                          float) {
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
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::AdamWKernel,
                          phi::dtype::float16,
                          float) {
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

PD_REGISTER_PLUGIN_KERNEL(merged_adam,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::MergedAdamKernel,
                          phi::dtype::float16,
                          float) {}
