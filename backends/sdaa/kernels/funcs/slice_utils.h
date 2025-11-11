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

#pragma once

#include "paddle/phi/extension.h"

namespace custom_kernel {

inline phi::DenseTensor Slice(const phi::DenseTensor& src,
                              int64_t begin_index,
                              int64_t end_index) {
  auto meta = src.meta();
  PADDLE_ENFORCE_GE(
      begin_index,
      0,
      phi::errors::OutOfRange("The start row index must be greater than 0."
                              "But received the start index is d%.",
                              begin_index));
  PADDLE_ENFORCE_LE(
      end_index,
      meta.dims[0],
      phi::errors::OutOfRange("The end row index is out of bound."));
  PADDLE_ENFORCE_LT(
      begin_index,
      end_index,
      phi::errors::InvalidArgument(
          "The start row index must be less than the end row index."
          "But received the start index = %d, the end index = %d.",
          begin_index,
          end_index));

  if (meta.dims[0] == 1) {
    return src;
  } else {
    size_t base = src.numel() / meta.dims[0];
    phi::DenseTensor dst(src);
    phi::DDim dst_dims = meta.dims;
    dst_dims[0] = end_index - begin_index;
    size_t dst_offset =
        meta.offset + begin_index * base * phi::SizeOf(meta.dtype);
    phi::DenseTensorMeta dst_meta = {
        meta.dtype, dst_dims, meta.layout, dst_offset};
    dst.set_meta(dst_meta);
    return dst;
  }
}
}  // namespace custom_kernel
