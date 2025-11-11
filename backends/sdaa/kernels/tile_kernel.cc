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

#include <iostream>
#include <vector>

#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

template <typename T, typename Context>
void TileKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::IntArray& repeat_times,
                phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA TileKernel";
  auto rank = x.dims().size();
  auto& repeat_times_data = repeat_times.GetData();
  int repeat_times_size = repeat_times_data.size();
  rank = std::max(rank, repeat_times_size);

  dev_ctx.template Alloc<T>(out);
  if (rank == 0) {
    phi::Copy(dev_ctx, x, x.place(), false, out);
    return;
  }
  for (size_t i = 0; i < repeat_times.size(); ++i) {
    PADDLE_ENFORCE_GT(
        repeat_times[i],
        0,
        phi::errors::InvalidArgument(
            "All elements of the input 'repeat_times' for tile op must "
            "be positive integers, but the value received is %d.",
            repeat_times[i]));
  }

  std::vector<int> x_dims = phi::vectorize<int>(x.dims());
  std::vector<int> out_dims = phi::vectorize<int>(out->dims());
  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);

  tecodnnTensorDescriptor_t x_Desc = sdaa_ops::GetTecodnnTensorDesc(
      x_dims, x.dtype(), TensorFormat::Undefined);
  tecodnnTensorDescriptor_t out_Desc = sdaa_ops::GetTecodnnTensorDesc(
      out_dims, out->dtype(), TensorFormat::Undefined);
  TECODNN_CHECK(tecodnnTile(tecodnnHandle,
                            repeat_times.size(),
                            repeat_times_data.data(),
                            x_Desc,
                            x.data(),
                            out_Desc,
                            out->data()));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(out_Desc));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(tile,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::TileKernel,
                          bool,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::dtype::float16) {}
