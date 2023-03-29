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
void SGDKernel(const Context& dev_ctx,
               const phi::DenseTensor& param_var,
               const phi::DenseTensor& learning_rate,
               const phi::DenseTensor& grad_var,
               const paddle::optional<phi::DenseTensor>& master_param,
               bool multi_precision,
               phi::DenseTensor* param_out,
               phi::DenseTensor* master_param_out) {
  dev_ctx.template Alloc<T>(param_out);
  MLUCnnlTensorDesc grad_desc(grad_var);

  // NOTE: if param and param_out is not same, we need to do copy first.
  if (param_out->data<T>() != param_var.data<T>()) {
    TensorCopy(dev_ctx, param_var, false, param_out);
  }
  MLUCnnlTensorDesc param_desc(*param_out);
  MLUCnnl::SGD(dev_ctx,
               grad_desc.get(),
               GetBasePtr(&grad_var),
               GetBasePtr(&learning_rate),
               param_desc.get(),
               GetBasePtr(param_out));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(sgd,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::SGDKernel,
                          phi::dtype::float16,
                          float) {}
