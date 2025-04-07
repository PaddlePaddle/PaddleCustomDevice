// BSD 3- Clause License Copyright (c) 2024, Tecorigin Co., Ltd. All rights
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

#include "kernels/funcs/sdaa_baseop.h"
#include "kernels/funcs/strided_copy_utils.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

template <typename T, typename Context>
void ContiguousKernel(const Context& dev_ctx,
                      const phi::DenseTensor& input,
                      phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA ContiguousKernel.";

  phi::DenseTensorMeta meta = input.meta();
  meta.strides = meta.calc_strides(meta.dims);
  meta.offset = 0;
  out->set_meta(meta);
  dev_ctx.template Alloc<T>(out);

  bool optimizer_exec = sdaa_copy::strided_copy(dev_ctx, input, out);

  if (!optimizer_exec) {
    VLOG(1) << "A point-to-point data handling was used as all optimization "
               "methods failed in contiguous kernel.";
    const T* input_data = input.data<T>();
    auto* output_data = out->data<T>();
    int rank = input.dims().size();
    int64_t numel = input.numel();
    auto dims = input.dims();
    auto input_stride = input.strides();

    for (int64_t i = 0; i < numel; i++) {
      int64_t input_offset = 0;
      int64_t index_tmp = i;
      for (int dim = rank - 1; dim >= 0; --dim) {
        int64_t mod = index_tmp % dims[dim];
        index_tmp = index_tmp / dims[dim];
        input_offset += mod * input_stride[dim];
      }

      auto* dst_ptr = &output_data[i];
      auto* src_ptr = &input_data[input_offset];
      AsyncMemCpyD2D(nullptr,
                     static_cast<C_Stream>(dev_ctx.stream()),
                     dst_ptr,
                     src_ptr,
                     phi::SizeOf(input.dtype()));
    }
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(contiguous,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::ContiguousKernel,
                          double,
                          float,
                          int,
                          int64_t,
                          int16_t,
                          int8_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          bool) {}
