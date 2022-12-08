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
#include "phi_funcs.h"
namespace custom_kernel {

template <typename T>
void MemcpyD2HKernel(const phi::Context& dev_ctx,
                     const phi::DenseTensor& x,
                     int dst_place_type,
                     phi::DenseTensor* out) {
  show_kernel("memcpy_d2h");
  auto out_data = dev_ctx.HostAlloc<T>(out);
  auto x_data = x.data<T>();
  void* stream = const_cast<void*>(dev_ctx.stream());
  auto* q = static_cast<sycl::queue*>(stream);
  show_debug("memcpy_d2h -> memcpy(to=" << std::hex << out_data << ", from="
                                        << x_data << ", size=" << std::dec
                                        << x.memory_size() << ")");
   q->memcpy(out_data, x_data, x.memory_size());
}

template <typename T>
void MemcpyH2DKernel(const phi::Context& dev_ctx,
                     const phi::DenseTensor& x,
                     int dst_place_type,
                     phi::DenseTensor* out) {
  show_kernel("memcpy_h2d");
  auto out_data = dev_ctx.Alloc<T>(out);
  auto x_data = x.data<T>();

  void* stream = const_cast<void*>(dev_ctx.stream());
  auto* q = static_cast<sycl::queue*>(stream);
  show_debug("memcpy_h2d -> memcpy(to="<< std::hex<< out_data << ", from="<< x_data << ", size="<< std::dec << x.memory_size()<<")");
  q->memcpy(out_data, x_data, x.memory_size());
}

}  // namespace custom_kernel

PD_BUILD_PHI_KERNEL(memcpy_d2h,
                    intel_gpu,
                    ALL_LAYOUT,
                    custom_kernel::MemcpyD2HKernel,
                    float,
                    double,
                    int32_t,
                    int64_t,
                    bool) {}

PD_BUILD_PHI_KERNEL(memcpy_h2d,
                    intel_gpu,
                    ALL_LAYOUT,
                    custom_kernel::MemcpyH2DKernel,
                    float,
                    double,
                    int32_t,
                    int64_t,
                    bool) {}
