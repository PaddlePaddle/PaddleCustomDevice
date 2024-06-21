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
#include "sdcops.h"  // NOLINT
namespace custom_kernel {

template <typename T, typename Context>
void MomentumKernel(const Context& dev_ctx,
                    const phi::DenseTensor& param,
                    const phi::DenseTensor& grad,
                    const phi::DenseTensor& velocity,
                    const phi::DenseTensor& learning_rate,
                    const paddle::optional<phi::DenseTensor>& master_param,
                    float mu_f,
                    bool use_nesterov,
                    const std::string& regularization_method,
                    float regularization_coeff,
                    bool multi_precision,
                    float rescale_grad,
                    phi::DenseTensor* param_out,
                    phi::DenseTensor* velocity_out,
                    phi::DenseTensor* master_param_out) {
  if (isEnvEnable("HIGH_PERFORMANCE_CONV") &&
      (!param.storage_properties_initialized()) &&
      (grad.storage_properties_initialized())) {
    SDAAStorageProperties grad_properties =
        grad.storage_properties<SDAAStorageProperties>();

    sdaa_ops::swapTensorData(dev_ctx, param, grad_properties);
    sdaa_ops::swapTensorData(dev_ctx, velocity, grad_properties);
    if (&param != param_out) {
      sdaa_ops::doAddStorageProperties(dev_ctx, param_out, grad_properties);
    }

    if (&velocity != velocity_out) {
      sdaa_ops::doAddStorageProperties(dev_ctx, velocity_out, grad_properties);
    }
  }
  VLOG(4) << "Call SDAA MomentumKernel";
  TensorCopy(dev_ctx, param, false, param_out);
  TensorCopy(dev_ctx, velocity, false, velocity_out);
  phi::DenseTensor* grad_in = const_cast<phi::DenseTensor*>(&grad);

  bool l2_decay = false;
  if (regularization_method == "l2_decay") {
    l2_decay = true;
  }
  int n_total = static_cast<int>(param.numel());
  void* A[3] = {param_out->data(), grad_in->data(), velocity_out->data()};
  sdaaStream_t custom_stream = GetStreamFromCTX(dev_ctx);
  TCUS_CHECK(sdcops::multi_t_ops_t_momentum(n_total,
                                            A,
                                            learning_rate.data<float>(),
                                            mu_f,
                                            regularization_coeff,
                                            use_nesterov,
                                            l2_decay,
                                            custom_stream));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(momentum,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::MomentumKernel,
                          float,
                          phi::dtype::float16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
  }
}
