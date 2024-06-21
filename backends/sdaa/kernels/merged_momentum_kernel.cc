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

#include <iostream>

#include "kernels/funcs/sdaa_baseop.h"
#include "kernels/funcs/sdaa_funcs.h"
#include "paddle/phi/extension.h"
#include "sdcops.h"  //NOLINT

namespace custom_kernel {

void CheckInputs(
    const std::vector<const phi::DenseTensor*>& param,
    const std::vector<const phi::DenseTensor*>& grad,
    const std::vector<const phi::DenseTensor*>& velocity,
    const std::vector<const phi::DenseTensor*>& learning_rate,
    const paddle::optional<std::vector<const phi::DenseTensor*>>& master_param,
    float mu,
    bool use_nesterov,
    const std::vector<std::string>& regularization_method,
    const std::vector<float>& regularization_coeff,
    bool multi_precision,
    float rescale_grad,
    std::vector<phi::DenseTensor*> param_out,
    std::vector<phi::DenseTensor*> velocity_out,
    std::vector<phi::DenseTensor*> master_param_out) {
  size_t n = param.size();
  PADDLE_ENFORCE_EQ(n,
                    param_out.size(),
                    phi::errors::InvalidArgument(
                        "The size of Output(ParamOut) must be equal to "
                        "Input(Param), but got the size of Output(ParamOut) "
                        "is %d, the size of Input(Param) is %d.",
                        param_out.size(),
                        n));
  for (size_t i = 0; i < n; ++i) {
    PADDLE_ENFORCE_EQ(param[i],
                      param_out[i],
                      phi::errors::InvalidArgument(
                          "The size of Input(Param) and Output(ParamOut) "
                          "must be the same Tensors."));
  }
  PADDLE_ENFORCE_EQ(
      n,
      grad.size(),
      phi::errors::InvalidArgument(
          "The size of Input(Grad) must be equal to Input(Param), but got "
          "the size of Input(Grad) is %d, the size of Input(Param) is %d.",
          grad.size(),
          n));
  PADDLE_ENFORCE_EQ(n,
                    velocity.size(),
                    phi::errors::InvalidArgument(
                        "The size of Input(Velocity) must be equal to "
                        "Input(Param), but got the size of Input(Velocity) "
                        "is %d, the size of Input(Param) is %d.",
                        velocity.size(),
                        n));
  PADDLE_ENFORCE_EQ(
      n,
      velocity_out.size(),
      phi::errors::InvalidArgument(
          "The size of Output(VelocityOut) must be "
          "equal to Input(Param), but got the size of Output(VelocityOut) is "
          "%d, the size of Input(Param) is %d.",
          velocity_out.size(),
          n));
  for (size_t i = 0; i < n; ++i) {
    PADDLE_ENFORCE_EQ(velocity[i],
                      velocity_out[i],
                      phi::errors::InvalidArgument(
                          "Input(Velocity) and Output(VelocityOut) must be "
                          "the same Tensors."));
  }
  if (learning_rate.size() != 1) {
    PADDLE_ENFORCE_EQ(
        n,
        learning_rate.size(),
        phi::errors::InvalidArgument(
            "If the size of Input(LearningRate) is not 1, the size of "
            "Input(LearningRate) must be "
            "equal to Input(Param), but got the size of Input(LearningRate) "
            "is %d, the size of Input(Param) is %d.",
            learning_rate.size(),
            n));
  }
  if (regularization_method.size() != 0) {
    PADDLE_ENFORCE_EQ(
        n,
        regularization_method.size(),
        phi::errors::InvalidArgument(
            "The size of Attr(regularization_method) must be equal "
            "to Input(Param), but got the size of "
            "Attr(regularization_method) is %d, the size of Input(Param) is "
            "%d.",
            regularization_method.size(),
            n));
    PADDLE_ENFORCE_EQ(
        n,
        regularization_coeff.size(),
        phi::errors::InvalidArgument(
            "The size of Attr(regularization_coeff) must be equal "
            "to Input(Param), but got the size of Attr(regularization_coeff) "
            "is %d, the size of Input(Param) is %d.",
            regularization_coeff.size(),
            n));
  }
}

template <typename T, typename Context>
void MergedMomentumKernel(
    const Context& dev_ctx,
    const std::vector<const phi::DenseTensor*>& param,
    const std::vector<const phi::DenseTensor*>& grad,
    const std::vector<const phi::DenseTensor*>& velocity,
    const std::vector<const phi::DenseTensor*>& learning_rate,
    const paddle::optional<std::vector<const phi::DenseTensor*>>& master_param,
    float mu,
    bool use_nesterov,
    const std::vector<std::string>& regularization_method,
    const std::vector<float>& regularization_coeff,
    bool multi_precision,
    float rescale_grad,
    std::vector<phi::DenseTensor*> param_out,
    std::vector<phi::DenseTensor*> velocity_out,
    std::vector<phi::DenseTensor*> master_param_out) {
  size_t param_num = param.size();
  const int M = param_num;
  static bool is_first_time = true;
  if (isEnvEnable("HIGH_PERFORMANCE_CONV") && is_first_time) {
    for (int i = 0; i < M; i++) {
      if (!param[i]->storage_properties_initialized() &&
          grad[i]->storage_properties_initialized()) {
        auto grad_properties =
            grad[i]->storage_properties<SDAAStorageProperties>();
        sdaa_ops::swapTensorData(dev_ctx, *param[i], grad_properties);
        sdaa_ops::swapTensorData(dev_ctx, *velocity[i], grad_properties);
      }
    }
    is_first_time = false;
  }
  VLOG(4) << "Call SDAA MergedMomentumKernel";
  CheckInputs(param,
              grad,
              velocity,
              learning_rate,
              master_param,
              mu,
              use_nesterov,
              regularization_method,
              regularization_coeff,
              multi_precision,
              rescale_grad,
              param_out,
              velocity_out,
              std::move(master_param_out));
  phi::DenseTensor lr, coeff, l2_decay;
  phi::DDim dim{M};
  phi::DenseTensorMeta meta(learning_rate[0]->dtype(), dim);
  lr.set_meta(meta);
  coeff.set_meta(meta);

  phi::DenseTensorMeta l2_meta(phi::DataType::INT32, dim);
  l2_decay.set_meta(l2_meta);
  if (learning_rate.size() != 1) {
    TensorFromVectorTensor<T>(dev_ctx, learning_rate, &lr);
  } else {
    std::vector<const phi::DenseTensor*> lr_vec;
    for (int i = 0; i < M; ++i) {
      lr_vec.push_back(learning_rate[0]);
    }
    TensorFromVectorTensor<T>(dev_ctx, lr_vec, &lr);
  }
  if (regularization_coeff.size()) {
    TensorFromVector(dev_ctx, regularization_coeff, dev_ctx, &coeff);
  } else {
    dev_ctx.template Alloc<T>(&coeff);
    sdaa_ops::doFillTensor<T>(
        dev_ctx, static_cast<T>(0.f), param[0]->dtype(), &coeff);
  }
  std::vector<int> l2_decay_vec(M, 0);
  for (size_t i = 0; i < regularization_method.size(); ++i) {
    auto& t = regularization_method[i];
    if (t == "l2_decay") {
      l2_decay_vec[i] = 1;
    }
  }
  TensorFromVector(dev_ctx, l2_decay_vec, dev_ctx, &l2_decay);
  int input_num = 3;
  void* data[input_num][M];
  std::vector<phi::DenseTensor*> grad_in;
  for (int i = 0; i < param_num; ++i) {
    grad_in.push_back(const_cast<phi::DenseTensor*>(grad[i]));
  }
  for (int i = 0; i < M; i++) {
    data[0][i] = param_out[i]->data();
    data[1][i] = grad_in[i]->data();
    data[2][i] = velocity_out[i]->data();
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

  std::vector<int64_t> len;
  for (int i = 0; i < M; i++) {
    len.push_back(param_out[i]->numel());
  }
  phi::DenseTensor n_total;
  phi::DenseTensorMeta meta1(phi::DataType::INT64, dim);
  n_total.set_meta(meta1);
  TensorFromVector(dev_ctx, len, dev_ctx, &n_total);
  sdaaStream_t custom_stream = GetStreamFromCTX(dev_ctx);
  TCUS_CHECK(sdcops::merged_momentum(M,
                                     n_total.data<int64_t>(),
                                     pointer,
                                     lr.data<float>(),
                                     coeff.data<float>(),
                                     mu,
                                     l2_decay.data<int>(),
                                     use_nesterov,
                                     custom_stream));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(merged_momentum,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::MergedMomentumKernel,
                          float,
                          phi::dtype::float16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
  }
}
