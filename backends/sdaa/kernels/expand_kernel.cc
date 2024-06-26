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

#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

template <typename T, typename Context>
void ExpandKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::IntArray& shape,
                  phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA ExpandKernel";

  auto expand_shape = shape.GetData();
  std::vector<int> x_dims = phi::vectorize<int>(x.dims());

  int diff = expand_shape.size() - x_dims.size();
  x_dims.insert(x_dims.begin(), diff, 1);

  std::vector<int> final_expand_shape(x_dims.size());
  for (size_t i = 0; i < x_dims.size(); i++) {
    PADDLE_ENFORCE_NE(
        expand_shape[i],
        0,
        phi::errors::InvalidArgument("The expanded size cannot be zero."));

    if (i < diff) {  // expand_shape = [3, 4, -1, -1], x = [10, 2] -->
                     // final_expand_shape = [3, 4, 10, 2]
      PADDLE_ENFORCE_GT(expand_shape[i],
                        0,
                        phi::errors::InvalidArgument(
                            "The expanded size (%d) for non-existing "
                            "dimensions must be positive for expand_v2 op",
                            expand_shape[i]));

      final_expand_shape[i] = expand_shape[i];
    } else if (expand_shape[i] >
               0) {  // expand_shape = [3, 4, 10, 2], x = [10, 1] -->
                     // final_expand_shape = [3, 4, 10, 2]
      if (x_dims[i] != 1) {
        PADDLE_ENFORCE_EQ(x_dims[i],
                          expand_shape[i],
                          phi::errors::InvalidArgument(
                              "The value (%d) of the non-singleton dimensions "
                              "does not much the corresponding value (%d) in "
                              "shape for expand_v2 op.",
                              x_dims[i],
                              expand_shape[i]));

        final_expand_shape[i] = expand_shape[i];
      } else {
        final_expand_shape[i] = expand_shape[i];
      }
    } else {  // expand_shape = [3, 4, -1, -1], x = [10, 2] -->
              // final_expand_shape = [3, 4, 10, 2]
      PADDLE_ENFORCE_EQ(
          expand_shape[i],
          -1,
          phi::errors::InvalidArgument(
              "When the value in shape is negative for expand_v2 op, "
              "only -1 is supported, but the value received is %d.",
              expand_shape[i]));

      final_expand_shape[i] = x_dims[i];
    }
  }

  auto rank = x.dims().size();
  PADDLE_ENFORCE_GE(
      rank,
      0,
      phi::errors::InvalidArgument(
          "The rank of the input 'x' for expand_v2 op must be positive, "
          "but the value received is %d.",
          rank));

  auto shape_size = final_expand_shape.size();
  PADDLE_ENFORCE_GE(
      shape_size,
      rank,
      phi::errors::InvalidArgument(
          "The number (%d) of elements of 'shape' for expand_v2 op must "
          "be "
          "greater than or equal to the rank (%d) of the input 'x'.",
          shape_size,
          rank));

  out->Resize(phi::make_ddim(final_expand_shape));
  dev_ctx.template Alloc<T>(out);

  sdaa_ops::doExpandTensor(dev_ctx, x, out);
}

template <typename T, typename Context>
void ExpandGradKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& out_grad,
                      const phi::IntArray& shape,
                      phi::DenseTensor* x_grad) {
  VLOG(4) << "CALL SDAA ExpandGradKernel";

  dev_ctx.template Alloc<T>(x_grad);
  if (x_grad->dims() == out_grad.dims()) {
    phi::Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);
  } else {
    std::vector<int64_t> reduce_dims =
        sdaa_ops::GetReduceDimAxis(x_grad->dims(), out_grad.dims(), -1);
    sdaa_ops::doSumTensor(dev_ctx, out_grad, reduce_dims, x_grad);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(expand,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::ExpandKernel,
                          float,
                          phi::dtype::float16,
                          double,
                          int32_t,
                          int64_t,
                          bool) {}

PD_REGISTER_PLUGIN_KERNEL(expand_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::ExpandGradKernel,
                          float,
                          double,
                          phi::dtype::float16,
                          int,
                          int64_t) {}
