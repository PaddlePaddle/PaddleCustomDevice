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
#include <CL/sycl.hpp>
namespace custom_kernel {

template <typename T, typename VType>
void FullValue(const phi::Context& dev_ctx,
               phi::DenseTensor* tensor,
               VType val) {


  auto t = dev_ctx.template Alloc<T>(tensor);
  auto* q = static_cast<sycl::queue*>(dev_ctx.stream());
  auto num = tensor->numel();
  std::cout << "*** FullValue *** size="<< num << " sizeof(T)="<< sizeof(T) << std::endl;
  q->submit([&](sycl::handler& h) {
    h.fill(t, val, num );
  });
  q-> wait();

  // q->submit([&](sycl::handler& h) {
  //   h.parallel_for(ile, [=](auto& i){
  //       t[i] = val;
  //   });
  // });
  // q-> wait();

/*
  T* gpu_mem = sycl::malloc_device<T>(ile, *q);

  q->submit([&](sycl::handler& h) {
    h.parallel_for(ile, [=](auto& i){
        gpu_mem[i] = val;
    });
  });
  q-> wait();

  q->submit([&](sycl::handler &h){
    h.memcpy(t, gpu_mem, sizeof(T)*ile);
  });
  q->wait();
  sycl::free(gpu_mem, *q);

*/


}

template <typename T>
void FullKernel(const phi::Context& dev_ctx,
                const phi::IntArray& shape,
                const phi::Scalar& val,
                phi::DataType dtype,
                phi::DenseTensor* out) {
  auto int_shape = shape.GetData();
  out->Resize(std::vector<int64_t>(int_shape.cbegin(), int_shape.cend()));
  FullValue<T>(dev_ctx, out, val.to<T>());
}
}  // namespace custom_kernel

PD_BUILD_PHI_KERNEL(full,
                    intel_gpu,
                    ALL_LAYOUT,
                    custom_kernel::FullKernel,
                    float,
                    double,
                    uint8_t,
                    int16_t,
                    int32_t,
                    int64_t,
                    bool) {}
