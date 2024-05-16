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

#include "common/gcu_op_runner.h"
#include "kernels/funcs/gcu_kernel_funcs.h"

namespace custom_kernel {

template <typename T, typename Context>
void MemcpyKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  int dst_place_type,
                  phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("memcpy");
  if (!x.initialized()) {
    return;
  }
  // The dst_place_type is defined in paddle/fluid/operators/memcpy_op.h:
  //     enum DeviceType {
  //       CPU = 0,
  //       CUDA = 1,
  //       CUDA_PINNED = 2,
  //       XPU = 3,
  //       NPU = 4,
  //       NPU_PINNED = 5,
  //       CUSTOM_DEVICE = 6,
  //     };
  if (dst_place_type == 0) {  // CPU
    TensorCopy(dev_ctx, x, false, out, phi::CPUPlace());
  } else if (dst_place_type == 6) {  // custom_device
    TensorCopy(dev_ctx, x, false, out, dev_ctx.GetPlace());
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "memcpy dst_place_type: %d is not supported yet.", dst_place_type));
  }
  dev_ctx.Wait();
}

template <typename T, typename Context>
void MemcpyH2DKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     int dst_place_type,
                     phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("memcpy_h2d");
  TensorCopy(dev_ctx, x, false, out, dev_ctx.GetPlace());
  dev_ctx.Wait();
}

// used in new executor, for memory copy from device to host
template <typename T, typename Context>
void MemcpyD2HKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     int dst_place_type,
                     phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("memcpy_d2h");
  if (x.storage_properties_initialized()) {
    PADDLE_THROW(
        phi::errors::Unimplemented("storage_properties is not supported yet."));
  } else {
    TensorCopy(dev_ctx, x, false, out, phi::CPUPlace());
  }
  dev_ctx.Wait();
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(memcpy_h2d,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::MemcpyH2DKernel,
                          phi::dtype::float16,
                          float,
                          double,
                          int8_t,
                          uint8_t,
                          int,
                          int64_t,
                          bool,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>,
                          int16_t) {}

PD_REGISTER_PLUGIN_KERNEL(memcpy_d2h,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::MemcpyD2HKernel,
                          phi::dtype::float16,
                          float,
                          double,
                          int8_t,
                          uint8_t,
                          int,
                          int64_t,
                          bool,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>,
                          int16_t) {}

PD_REGISTER_PLUGIN_KERNEL(memcpy,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::MemcpyKernel,
                          phi::dtype::float16,
                          float,
                          double,
                          int8_t,
                          uint8_t,
                          int,
                          int64_t,
                          bool,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>,
                          int16_t) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}
