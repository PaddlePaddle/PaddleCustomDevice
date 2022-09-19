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

#include "paddle/phi/capi/all.h"
#include "phi_funcs.h"
#include <CL/sycl.hpp>
#include "dnn_support.hpp"
namespace custom_kernel {

template <typename T>
void MeanAllKernel(const phi::Context& dev_ctx,
                   const phi::DenseTensor& x,
                   phi::DenseTensor* out) {
  auto out_data = dev_ctx.template Alloc<T>(out);
  auto x_data = x.data<T>();
  auto numel = x.numel();

  show_kernel("Mean-Sycl");
  auto* q = static_cast<sycl::queue*>(dev_ctx.stream());

  show_debug("Mean numel="<< numel );

  auto e1 = q->fill(out_data, static_cast<T>(0), 1);
  q->single_task(e1, [=](){
    for (auto i = 0; i < numel; ++i) {
      *out_data += x_data[i];
    };
    *out_data /= static_cast<T>(numel);
  });

  q->wait();
}

template <typename T>
void MeanAllGradKernel(const phi::Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& out_grad,
                       phi::DenseTensor* x_grad) {
  PD_CHECK(out_grad.numel() == 1UL,
           "Mean Gradient should be scalar. But received "
           "Out@Grad's elements num is %d.",
           out_grad.numel());
  auto x_grad_data = dev_ctx.template Alloc<T>(x_grad);
  auto out_grad_data = out_grad.data<T>();
  auto numel = x_grad->numel();
  auto* q = static_cast<sycl::queue*>(dev_ctx.stream());

  show_kernel("MeanAllGrad-Sycl");

  q->submit([&](sycl::handler& h) {
    h.parallel_for(numel, [=](auto& i){
        x_grad_data[i] = *out_grad_data / static_cast<T>(numel);
    });
  });
  q->wait();
}

}  // namespace custom_kernel

PD_BUILD_PHI_KERNEL(mean_all,
                    intel_gpu,
                    ALL_LAYOUT,
                    custom_kernel::MeanAllKernel,
                    float,
                    double) {}

PD_BUILD_PHI_KERNEL(mean_all_grad,
                    intel_gpu,
                    ALL_LAYOUT,
                    custom_kernel::MeanAllGradKernel,
                    float,
                    double) {}
