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
#include "kernels/funcs/range_op.h"
#include "paddle/phi/common/scalar.h"

namespace custom_kernel {

using phi::Scalar;

template <typename T, typename Context>
void ArangeKernel(const Context& dev_ctx,
                  const phi::Scalar& start,
                  const phi::Scalar& end,
                  const phi::Scalar& step,
                  phi::DenseTensor* out) {
  T start_value = start.to<T>();
  T end_value = end.to<T>();
  T step_value = step.to<T>();

  ArangeRawKernel<T>(dev_ctx, start_value, end_value, step_value, out);
}

template <typename T, typename Context>
void ArangeTensorKernel(const Context& dev_ctx,
                        const phi::DenseTensor& start_t,
                        const phi::DenseTensor& end_t,
                        const phi::DenseTensor& step_t,
                        phi::DenseTensor* out) {
  custom_kernel::ArangeKernel<T, Context>(
      dev_ctx, Scalar(start_t), Scalar(end_t), Scalar(step_t), out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(arange_tensor,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::ArangeTensorKernel,
                          int,
                          int64_t,
                          float,
                          double) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(1).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
}

PD_REGISTER_PLUGIN_KERNEL(arange,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::ArangeKernel,
                          int,
                          int64_t,
                          float,
                          double) {}
