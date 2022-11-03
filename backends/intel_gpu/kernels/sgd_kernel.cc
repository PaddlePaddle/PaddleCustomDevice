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
#include "dnn_support.hpp"
#include "paddle/phi/capi/all.h"

namespace custom_kernel {

template <typename T>
void sgd_dense_param_dense_grad_impl(sycl::queue* q, const phi::DenseTensor& param,
                                     const phi::DenseTensor& learning_rate,
                                     const phi::DenseTensor& grad,
                                     phi::DenseTensor* param_out) {
  const auto sz = param_out->numel();
  const T* lr = learning_rate.data<T>();
  const T* param_data = param.data<T>();
  const T* grad_data = grad.data<T>();
  T* out_data = param_out->data<T>();

  // for (auto i = 0; i < sz; ++i) {
  //   out_data[i] = param_data[i] - *lr * grad_data[i];
  // }

   q->parallel_for(sz, [=](auto& i){

          out_data[i] = param_data[i] - *lr * grad_data[i];
   });

  q->wait();

}

template <typename T>
void SGDDenseKernel(const phi::Context& dev_ctx,
                    const phi::DenseTensor& param,
                    const phi::DenseTensor& learning_rate,
                    const phi::DenseTensor& grad,
                    const paddle::optional<phi::DenseTensor>& master_param,
                    bool multi_precision,
                    phi::DenseTensor* param_out,
                    phi::DenseTensor* master_param_out) {
  show_kernel("sgd " << dnn_support::type2String<T>::name() );

  auto* q = static_cast<sycl::queue*>(const_cast<void*>(dev_ctx.stream()));

  if (!q) {
  }

  dev_ctx.template Alloc<T>(param_out);
  sgd_dense_param_dense_grad_impl<T>(q,param, learning_rate, grad, param_out);
}
}  // namespace custom_kernel

PD_BUILD_PHI_KERNEL(
    sgd, intel_gpu, ALL_LAYOUT, custom_kernel::SGDDenseKernel, float, double) {
}
