// BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
// reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
// OF SUCH DAMAGE.

#include <cmath>
#include <iostream>

#include "kernels/funcs/sdaa_baseop.h"
#include "sdcops.h"   // NOLINT
#include "tecodnn.h"  // NOLINT
namespace custom_kernel {

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
  VLOG(4) << "call sdaa MergedAdamKernel";
  if (beta1_pow[0]->place().GetType() == phi::AllocationType::CPU) {
    VLOG(4) << "beta1_pow place is cpu!";
  }
  if (beta1_pow_out[0]->place().GetType() == phi::AllocationType::CPU) {
    VLOG(4) << "beta1_pow_out place is cpu!";
  }
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
  phi::DenseTensor lr, b1_pow, b2_pow;
  const int M = param_num;
  std::vector<int> dims = {M};
  phi::DDim dim = phi::make_ddim(dims);
  phi::DenseTensorMeta meta = {learning_rate[0]->dtype(), dim};
  lr.set_meta(meta);
  b1_pow.set_meta(meta);
  b2_pow.set_meta(meta);
  TensorFromVectorTensor<T>(dev_ctx, learning_rate, &lr);
  TensorFromVectorTensor<T>(dev_ctx, beta1_pow, &b1_pow);
  TensorFromVectorTensor<T>(dev_ctx, beta2_pow, &b2_pow);

  int input_num = 4;
  void* data[input_num][M];
  std::vector<phi::DenseTensor*> grad_in;
  for (int i = 0; i < param_num; ++i) {
    TensorCopy(dev_ctx, *param[i], false, param_out[i]);
    TensorCopy(dev_ctx, *moment1[i], false, moment1_out[i]);
    TensorCopy(dev_ctx, *moment2[i], false, moment2_out[i]);
    grad_in.push_back(const_cast<phi::DenseTensor*>(grad[i]));
  }
  for (int i = 0; i < M; i++) {
    data[0][i] = grad_in[i]->data();
    data[1][i] = param_out[i]->data();
    data[2][i] = moment1_out[i]->data();
    data[3][i] = moment2_out[i]->data();
  }

  void* pointer[input_num];
  std::vector<phi::DenseTensor> pointer_data(input_num);
  int64_t pointer_bytes = M * sizeof(void*);
  for (int i = 0; i < input_num; ++i) {
    pointer_data[i].Resize({pointer_bytes});
    dev_ctx.template Alloc<uint8_t>(&pointer_data[i]);
    AsyncMemCpyH2D(nullptr,
                   static_cast<C_Stream>(dev_ctx.stream()),
                   pointer_data[i].data(),
                   data[i],
                   pointer_bytes);
    pointer[i] = pointer_data[i].data();
  }

  float beta1_ = beta1.to<float>();  // cpu
  float beta2_ = beta2.to<float>();  // cpu
  float epsilon_ = epsilon.to<float>();

  std::vector<int64_t> len;
  for (int i = 0; i < M; i++) {
    int64_t num = param_out[i]->numel();
    len.push_back(num);
  }
  phi::DenseTensor n_total;
  phi::DenseTensorMeta meta1 = {phi::DataType::INT64, dim};
  n_total.set_meta(meta1);
  TensorFromVector(dev_ctx, len, dev_ctx, &n_total);
  sdaaStream_t custom_stream = GetStreamFromCTX(dev_ctx);
  TCUS_CHECK(sdcops::merged_adam_ops(M,
                                     n_total.data<int64_t>(),
                                     pointer,
                                     lr.data<T>(),
                                     b1_pow.data<T>(),
                                     b2_pow.data<T>(),
                                     beta1_,
                                     beta2_,
                                     epsilon_,
                                     0,
                                     false,
                                     custom_stream));
  if (!use_global_beta_pow) {
    sdaa_ops::doScaleTensor(dev_ctx, b1_pow, beta1_, 0, true, false, &b1_pow);
    sdaa_ops::doScaleTensor(dev_ctx, b2_pow, beta2_, 0, true, false, &b2_pow);

    for (int i = 0; i < M; ++i) {
      dev_ctx.template Alloc<T>(beta1_pow_out[i]);
      AsyncMemCpyD2D(nullptr,
                     static_cast<C_Stream>(dev_ctx.stream()),
                     beta1_pow_out[i]->data(),
                     reinterpret_cast<void*>(b1_pow.data<T>() + i),
                     sizeof(T));
      dev_ctx.template Alloc<T>(beta2_pow_out[i]);
      AsyncMemCpyD2D(nullptr,
                     static_cast<C_Stream>(dev_ctx.stream()),
                     beta2_pow_out[i]->data(),
                     reinterpret_cast<void*>(b2_pow.data<T>() + i),
                     sizeof(T));
    }
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    merged_adam, sdaa, ALL_LAYOUT, custom_kernel::MergedAdamKernel, float) {
  // Skip beta1_pow, beta2_pow, skip_update data transform
  kernel->InputAt(5).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(6).SetBackend(phi::Backend::ALL_BACKEND);
}
