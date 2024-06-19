// BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
// reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
// OF SUCH DAMAGE.

#include "kernels/funcs/sdaa_funcs.h"

namespace custom_kernel {

template <typename T, typename Context>
void MemcpyKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  int dst_place_type,
                  phi::DenseTensor* out) {
  if (!x.initialized()) {
    return;
  }
  // The dst_place_type is defined in paddle/fluid/operators/memcpy.h:
  // CPU = 0, CUDA = 1, CUDA_PINNED = 2,
  // XPU = 3, NPU = 4, NPU_PINNED = 5,
  // CUSTOM_DEVICE = 6
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
  TensorCopy(dev_ctx, x, false, out, dev_ctx.GetPlace());
  dev_ctx.Wait();
}

// used in new executor, for memory copy from device to host
template <typename T, typename Context>
void MemcpyD2HKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     int dst_place_type,
                     phi::DenseTensor* out) {
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
                          sdaa,
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
                          sdaa,
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

PD_REGISTER_PLUGIN_KERNEL(memcpy_d2h_multi_io,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::MemcpyD2HMultiIOKernel,
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
                          sdaa,
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
