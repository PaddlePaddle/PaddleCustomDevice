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
void ArangeKernel(const Context& dev_ctx,
                  const phi::DenseTensor& start_t,
                  const phi::DenseTensor& end_t,
                  const phi::DenseTensor& step_t,
                  phi::DenseTensor* out) {
  phi::DenseTensor n;
  n.Resize(start_t.dims());
  T* n_data = dev_ctx.template HostAlloc<T>(&n);

  TensorCopy(dev_ctx, start_t, true, &n, phi::CPUPlace());
  T start = n_data[0];

  TensorCopy(dev_ctx, end_t, true, &n, phi::CPUPlace());
  T end = n_data[0];

  TensorCopy(dev_ctx, step_t, true, &n, phi::CPUPlace());
  T step = n_data[0];

  int64_t size = 0;
  GetSize(start, end, step, &size);

  out->Resize(phi::make_ddim({size}));
  dev_ctx.template Alloc<T>(out);

  std::vector<T> odata;
  T value = start;
  for (int64_t i = 0; i < size; ++i) {
    odata.push_back(value);
    value += step;
  }

  TensorFromVector(dev_ctx, odata, dev_ctx, out);
  dev_ctx.Wait();
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(arange,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::ArangeKernel,
                          int,
                          int64_t,
                          float,
                          double) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(1).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
}
