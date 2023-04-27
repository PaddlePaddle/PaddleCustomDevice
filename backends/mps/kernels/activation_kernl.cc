// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "kernels/activation_impl.h"
#include "kernels/phi_funcs.h"
#include "paddle/phi/capi/all.h"

namespace custom_kernel {

template <typename T>

void PowKernel(const phi::Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::Scalar& factor_scalar,
               phi::DenseTensor* out) {
  auto factor = factor_scalar.to<float>();
  T* out_data = dev_ctx.template Alloc<T>(out);
  mps_kernel::Pow(x.data<T>(), out->data<T>(), x.dims(), factor);
}

template <typename T>
void ExpKernel(const phi::Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  T* out_data = dev_ctx.template Alloc<T>(out);
  mps_kernel::Activation(
      x.data<T>(), out->data<T>(), x.dims(), mps_kernel::ActivationOP::EXP);
}

template <typename T>
void SigmoidKernel(const phi::Context& dev_ctx,
                   const phi::DenseTensor& x,
                   phi::DenseTensor* out) {
  T* out_data = dev_ctx.template Alloc<T>(out);
  mps_kernel::Activation(
      x.data<T>(), out->data<T>(), x.dims(), mps_kernel::ActivationOP::SIGMOID);
}

template <typename T>
void SinKernel(const phi::Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  T* out_data = dev_ctx.template Alloc<T>(out);
  mps_kernel::Activation(
      x.data<T>(), out->data<T>(), x.dims(), mps_kernel::ActivationOP::SIN);
}

template <typename T>
void CosKernel(const phi::Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  T* out_data = dev_ctx.template Alloc<T>(out);
  mps_kernel::Activation(
      x.data<T>(), out->data<T>(), x.dims(), mps_kernel::ActivationOP::COS);
}

}  // namespace custom_kernel

PD_BUILD_PHI_KERNEL(pow, mps, ALL_LAYOUT, custom_kernel::PowKernel, float) {}

PD_BUILD_PHI_KERNEL(exp, mps, ALL_LAYOUT, custom_kernel::ExpKernel, float) {}

PD_BUILD_PHI_KERNEL(
    sigmoid, mps, ALL_LAYOUT, custom_kernel::SigmoidKernel, float) {}

PD_BUILD_PHI_KERNEL(sin, mps, ALL_LAYOUT, custom_kernel::SinKernel, float) {}

PD_BUILD_PHI_KERNEL(cos, mps, ALL_LAYOUT, custom_kernel::CosKernel, float) {}
