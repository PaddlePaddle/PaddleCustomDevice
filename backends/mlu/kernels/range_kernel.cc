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

#include "kernels/funcs/range_op.h"

namespace custom_kernel {

template <typename T, typename Context>
void ArangeTensorKernel(const Context& dev_ctx,
                        const phi::DenseTensor& start_t,
                        const phi::DenseTensor& end_t,
                        const phi::DenseTensor& step_t,
                        phi::DenseTensor* out) {
  T* h_start_ptr = nullptr;
  T* h_end_ptr = nullptr;
  T* h_step_ptr = nullptr;

  if (start_t.place().GetType() == phi::AllocationType::CPU) {  // tensor at CPU
    h_start_ptr = reinterpret_cast<T*>(const_cast<void*>(GetBasePtr(&start_t)));
    h_end_ptr = reinterpret_cast<T*>(const_cast<void*>(GetBasePtr(&end_t)));
    h_step_ptr = reinterpret_cast<T*>(const_cast<void*>(GetBasePtr(&step_t)));
  } else {
    phi::DenseTensor n;
    n.Resize(start_t.dims());
    T* n_data = dev_ctx.template HostAlloc<T>(&n);
    TensorCopy(dev_ctx, start_t, true, &n, phi::CPUPlace());
    h_start_ptr = new T(n_data[0]);
    TensorCopy(dev_ctx, end_t, true, &n, phi::CPUPlace());
    h_end_ptr = new T(n_data[0]);
    TensorCopy(dev_ctx, step_t, true, &n, phi::CPUPlace());
    h_step_ptr = new T(n_data[0]);
  }

  T start_value = h_start_ptr[0];
  T end_value = h_end_ptr[0];
  T step_value = h_step_ptr[0];

  ArangeRawKernel<T>(dev_ctx, start_value, end_value, step_value, out);
  if (start_t.place().GetType() != phi::AllocationType::CPU) {
    delete h_start_ptr;
    delete h_end_ptr;
    delete h_step_ptr;
  }
}

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
