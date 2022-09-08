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

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void MemcpyKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  int dst_place_type,
                  phi::DenseTensor* out) {
  if (!x.initialized()) {
    return;
  }
  dev_ctx.template Alloc<T>(out);
  if (dst_place_type == 0) {  // CPU
    TensorCopy(dev_ctx, x, false, out, phi::CPUPlace());
  } else if (dst_place_type == 4) {  // NPU
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
  dev_ctx.template Alloc<T>(out);
  TensorCopy(dev_ctx, x, false, out, dev_ctx.GetPlace());
  dev_ctx.Wait();
}

// used in new executor, for memory copy from device to host
template <typename T, typename Context>
void MemcpyD2HKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     int dst_place_type,
                     phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  TensorCopy(dev_ctx, x, false, out, phi::CPUPlace());
  dev_ctx.Wait();
}

template <typename T, typename Context>
void MemcpyD2HMultiIOKernel(const Context& dev_ctx,
                            const std::vector<const phi::DenseTensor*>& array,
                            int dst_place_type,
                            std::vector<phi::DenseTensor*> out_array) {
  PADDLE_ENFORCE_EQ(
      array.size(),
      out_array.size(),
      phi::errors::PreconditionNotMet(
          "input size %d != output size %d", array.size(), out_array.size()));
  for (size_t i = 0; i < array.size(); i++) {
    PADDLE_ENFORCE_NOT_NULL(array[i],
                            phi::errors::PreconditionNotMet(
                                "input tesnor %d should not be nullptr", i));
    PADDLE_ENFORCE_NOT_NULL(out_array[i],
                            phi::errors::PreconditionNotMet(
                                "output tesnor %d should not be nullptr", i));

    const auto& x = *(array[i]);
    MemcpyD2HKernel<T, Context>(dev_ctx, x, dst_place_type, out_array[i]);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(memcpy_h2d,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::MemcpyH2DKernel,
                          phi::dtype::float16,
                          float,
                          double,
                          int,
                          int64_t,
                          bool) {}

PD_REGISTER_PLUGIN_KERNEL(memcpy_d2h,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::MemcpyD2HKernel,
                          phi::dtype::float16,
                          float,
                          double,
                          int,
                          int64_t,
                          bool) {}

PD_REGISTER_PLUGIN_KERNEL(memcpy_d2h_multi_io,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::MemcpyD2HMultiIOKernel,
                          phi::dtype::float16,
                          float,
                          double,
                          int,
                          int64_t,
                          bool) {}

PD_REGISTER_PLUGIN_KERNEL(memcpy,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::MemcpyKernel,
                          phi::dtype::float16,
                          float,
                          double,
                          int,
                          int64_t,
                          bool) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}
