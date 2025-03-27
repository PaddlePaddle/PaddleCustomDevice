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
void StridedCopyKernel(const Context& dev_ctx,
                       const phi::DenseTensor& input,
                       const std::vector<int64_t>& dims,
                       const std::vector<int64_t>& out_stride,
                       int64_t offset,
                       phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA StridedCopyKernel.";

  phi::DenseTensorMeta meta = input.meta();
  meta.strides = phi::make_ddim(out_stride);
  meta.dims = phi::make_ddim(dims);
  meta.offset = offset;
  out->set_meta(meta);

  PADDLE_ENFORCE_EQ(input.dims(),
                    out->dims(),
                    phi::errors::InvalidArgument(
                        "Input shape(%s) must be equal with out shape(%s).",
                        input.dims(),
                        out->dims()));

  PADDLE_ENFORCE_EQ(input.numel(),
                    out->numel(),
                    phi::errors::InvalidArgument(
                        "Input numel(%d) must be equal with out numel(%d).",
                        input.numel(),
                        out->numel()));

  if (input.numel() <= 0) {
    return;
  }

  T* output_data = out->data<T>();
  PADDLE_ENFORCE_NOT_NULL(output_data,
                          phi::errors::InvalidArgument(
                              "StridedCopyKernel's out tensor must complete "
                              "mutable data before call kernel."));

  bool optimizer_exec = sdaa_copy::strided_copy(dev_ctx, input, out);

  if (!optimizer_exec) {
    VLOG(1) << "A point-to-point data handling was used as all optimization "
               "methods failed in strided_copy kernel.";
    const T* input_data = input.data<T>();
    const int64_t* input_dims = input.dims().Get();
    const int64_t* input_stride = input.strides().Get();
    int64_t numel = input.numel();
    int input_rank = input.dims().size();
    int output_rank = meta.dims.size();
    const int64_t* output_dims = meta.dims.Get();
    const int64_t* output_stride = meta.strides.Get();

    for (int64_t i = 0; i < numel; i++) {
      int64_t input_offset = 0;
      int64_t index_tmp = i;
      for (int dim = input_rank - 1; dim >= 0; --dim) {
        input_offset += (index_tmp % input_dims[dim]) * input_stride[dim];
        index_tmp = index_tmp / input_dims[dim];
      }
      int64_t output_offset = 0;
      index_tmp = i;
      for (int dim = output_rank - 1; dim >= 0; --dim) {
        output_offset += (index_tmp % output_dims[dim]) * output_stride[dim];
        index_tmp = index_tmp / output_dims[dim];
      }

      auto* dst_ptr = &output_data[output_offset];
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

PD_REGISTER_PLUGIN_KERNEL(strided_copy,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::StridedCopyKernel,
                          double,
                          float,
                          int,
                          int64_t,
                          int16_t,
                          int8_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          bool) {}
