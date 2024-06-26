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

#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

static inline int ComputeAxis(int axis, int rank) {
  PADDLE_ENFORCE_EQ(
      axis >= -rank & axis < rank,
      true,
      phi::errors::InvalidArgument(
          "The axis is expected to be in range of [%d, %d), but got %d.",
          -rank,
          rank,
          axis));
  if (axis < 0) {
    axis = axis + rank;
  }
  return axis > 0 ? axis : 0;
}

template <typename T, typename Context>
void SplitKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const phi::IntArray& num_or_sections,
                 const phi::Scalar& axis_scalar,
                 std::vector<phi::DenseTensor*> outs) {
  VLOG(4) << "Call SDAA SplitKernel";

  int axis = axis_scalar.to<int>();

  axis = ComputeAxis(axis, static_cast<int>(x.dims().size()));

  for (size_t j = 0; j < outs.size(); j++) {
    dev_ctx.template Alloc<T>(outs[j]);
  }

  sdaa_ops::doSplitTensor(dev_ctx, x, axis, outs);
}

template <typename T, typename Context>
void SplitWithNumKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        int num,
                        const phi::Scalar& axis_scalar,
                        std::vector<phi::DenseTensor*> outs) {
  VLOG(4) << "Call SDAA SplitWithNumKernel";

  int axis_value = axis_scalar.to<int>();
  auto input_axis_dim = x.dims().at(axis_value);
  std::vector<int64_t> sections_vec;
  for (int i = 0; i < num; ++i) {
    sections_vec.push_back(input_axis_dim / num);
  }
  phi::IntArray sections(sections_vec);
  custom_kernel::SplitKernel<T, Context>(
      dev_ctx, x, sections, axis_scalar, std::move(outs));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(split,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::SplitKernel,
                          float,
                          phi::dtype::float16,
                          int64_t,
                          int,
                          uint8_t,
                          bool) {}

PD_REGISTER_PLUGIN_KERNEL(split_with_num,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::SplitWithNumKernel,
                          float,
                          phi::dtype::float16,
                          int64_t,
                          int,
                          uint8_t,
                          bool) {}
