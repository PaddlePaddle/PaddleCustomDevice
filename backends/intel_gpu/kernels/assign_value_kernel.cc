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

#include <cmath>

#include "paddle/phi/capi/all.h"
#include "phi_funcs.h"
#include <CL/sycl.hpp>
#include "dnn_support.hpp"

namespace custom_kernel {

template <typename T>
void AssignValueKernel(const phi::Context& dev_ctx,
                       const std::vector<int>& shape,
                       phi::DataType dtype,
                       const std::vector<phi::Scalar>& values,
                       phi::DenseTensor* out) {
  show_kernel("AssignValue-SYCL");
  auto template_dtype = phi::capi::CppTypeToPDType<T>::Type();
  PD_CHECK(dtype == template_dtype,
           "Argument dtype mismatch for kernel dtype, "
           "argument dtype is %s, kernel dtype is %s.",
           dtype,
           template_dtype);
  auto out_size = values.size();
  out->Resize({static_cast<int64_t>(out_size)});
  auto out_data = dev_ctx.template Alloc<T>(out);

  auto* q = static_cast<sycl::queue*>(dev_ctx.stream());

  std::vector<T> assign_values;
  assign_values.reserve(values.size());
  for (const auto& val : values) {
      assign_values.emplace_back( val.to<T>());
  }
  {
    sycl::buffer<T> assign_value_buf(assign_values);
    q->submit([&](sycl::handler& h) {
      auto a_in = assign_value_buf.template get_access<sycl::access::mode::read>(h);
      h.copy(a_in, out_data);
    });
  }
  // q-> wait();  needed, runtime does sync ??
  out->Resize(std::vector<int64_t>(shape.cbegin(), shape.cend()));
}

template <typename T>
void AssignKernel(const phi::Context& dev_ctx,
                  const phi::DenseTensor& x,
                  phi::DenseTensor* out) {
  auto out_data = dev_ctx.template Alloc<T>(out);
  auto x_data = x.data<T>();
  std::cout << "***** Tutaj assign Kernel" << std::endl;
  std::memcpy(out_data, x_data, sizeof(T) * x.numel());
}

}  // namespace custom_kernel

PD_BUILD_PHI_KERNEL(assign_value,
                    intel_gpu,
                    ALL_LAYOUT,
                    custom_kernel::AssignValueKernel,
                    // bool,
                    int,
                    int64_t,
                    float,
                    double
                    ) {}

PD_BUILD_PHI_KERNEL(assign,
                    intel_gpu,
                    ALL_LAYOUT,
                    custom_kernel::AssignKernel,
                    int,
                    int64_t,
                    float,
                    double) {}
