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

#include "kernels/dnn_support.hpp"
#include "kernels/phi_funcs.h"
#include "paddle/phi/capi/all.h"

namespace custom_kernel {

namespace gpu {

template <typename T>
void ArgsortKernel(const phi::Context& dev_ctx,
                   const phi::DenseTensor& input,
                   int axis,
                   bool descending,
                   phi::DenseTensor* output,
                   phi::DenseTensor* indices) {
  auto in_dims = input.dims();
  axis = (axis < 0) ? (in_dims.size() + axis) : axis;
  T* out_data = dev_ctx.template Alloc<T>(output);

  int64_t* ids_data = dev_ctx.template Alloc<int64_t>(indices);
  show_kernel("argsort in_dims=" << in_dims << " axis=" << axis << " type="
                                 << dnn_support::type2String<T>::name()
                                 << " desc=" << descending);

  PD_CHECK(in_dims.size() < 3, "PoC Lenet/Mnist use case only");

  namespace adpl_e = ::oneapi::dpl::execution;
  namespace adpl = ::oneapi::dpl;

  auto* q = static_cast<sycl::queue*>(const_cast<void*>(dev_ctx.stream()));
  auto policy_e = adpl_e::make_device_policy(*q);

  size_t n = 1;
  size_t m = in_dims[0];

  if (in_dims.size() == 2) {
    n = in_dims[0];
    m = in_dims[1];
  }

  auto input_data = input.data<T>();
  q->memcpy(out_data, input_data, input.memory_size());
  q->wait();

  for (size_t i = 0; i < n; i++) {
    q->parallel_for(m,
                    [p_data = ids_data + i * m, m](auto& i) { p_data[i] = i; });

    q->wait();

    sycl::buffer<int64_t> keys_buf{reinterpret_cast<int64_t*>(ids_data + i * m),
                                   sycl::range<1>(m)};
    sycl::buffer<T> vals_buf{reinterpret_cast<T*>(out_data + i * m),
                             sycl::range<1>(m)};

    auto keys_begin = adpl::begin(keys_buf);
    auto vals_begin = adpl::begin(vals_buf);
    auto zipped_begin = adpl::make_zip_iterator(keys_begin, vals_begin);

    // gpu sort
    std::stable_sort(policy_e,
                     zipped_begin,
                     zipped_begin + m,
                     [descending](auto lhs, auto rhs) {
                       return (descending)
                                  ? (adpl::get<1>(lhs) > adpl::get<1>(rhs))
                                  : (adpl::get<1>(lhs) < adpl::get<1>(rhs));
                     });
  }
}  // ArgsortKernel

}  // namespace gpu

}  // namespace custom_kernel

PD_BUILD_PHI_KERNEL(argsort,
                    intel_gpu,
                    ALL_LAYOUT,
                    custom_kernel::gpu::ArgsortKernel,
                    float,
                    double,
                    int,
                    int64_t) {}
