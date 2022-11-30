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
#include "kernels/funcs/op_command.h"

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
  dev_ctx.template Alloc<T>(param_out);
  dev_ctx.template Alloc<T>(velocity_out);

  phi::DenseTensor host_mu;
  experimental::OpCommandHelper::ScalarToHostTensor(
      dev_ctx, static_cast<T>(mu_f), &host_mu);

  phi::DenseTensor regularized_grad;
  if (regularization_method == "l2_decay") {
    regularized_grad.Resize(grad.dims());
    dev_ctx.template Alloc<T>(&regularized_grad);
    experimental::OpCommand("Axpy")
        .Input(grad)
        .Input(param)
        .Output(regularized_grad)
        .Attr("alpha", regularization_coeff)
        .Run(dev_ctx);
  } else {
    regularized_grad = grad;
  }
  // NOTE: ApplyMomentum will change the input
  experimental::OpCommandHelper::MarkAsParameter(param_out);
  experimental::OpCommandHelper::MarkAsParameter(velocity_out);
  phi::DenseTensor tmp_out;
  tmp_out.Resize(param_out->dims());
  dev_ctx.template Alloc<T>(&tmp_out);
  experimental::OpCommand("ApplyMomentum")
      .Input(*param_out,
             experimental::TensorDescMaker("var")
                 .FromTensor(*param_out)
                 .SetDataLayout(phi::DataLayout::ANY))
      .Input(*velocity_out,
             experimental::TensorDescMaker("accum")
                 .FromTensor(*velocity_out)
                 .SetDataLayout(phi::DataLayout::ANY))
      .Input(learning_rate)
      .Input(regularized_grad)
      .ScalarInput(host_mu)
      .Attr("use_nesterov", use_nesterov)
      .Output(tmp_out)
      .Run(dev_ctx);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(momentum,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::MomentumKernel,
                          phi::dtype::float16,
                          float,
                          double) {}
