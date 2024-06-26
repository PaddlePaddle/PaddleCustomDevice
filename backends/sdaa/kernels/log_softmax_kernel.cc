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
template <typename T, typename Context>
void LogSoftmaxKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      int axis,
                      phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA LogSoftmaxKernel";
  const int rank = x.dims().size();

  dev_ctx.template Alloc<T>(out);
  // For 0D Tensor
  if (rank == 0) {
    sdaa_ops::doFillTensor<T>(dev_ctx, static_cast<T>(0.0), out->dtype(), out);
    return;
  }

  if (axis < 0) {
    axis += x.dims().size();
  }
  if (axis != x.dims().size() - 1) {
    phi::DenseTensor x_temp;
    phi::DenseTensor out_temp;
    std::vector<int> x_dims = phi::vectorize<int>(x.dims());
    std::vector<int> axis_vec(x.dims().size());
    std::iota(axis_vec.begin(), axis_vec.end(), 0);
    // transpose axis_dim to last_dim
    int axis_dim = x_dims[axis];
    for (int i = axis; i < x.dims().size() - 1; i++) {
      x_dims[i] = x_dims[i + 1];
      axis_vec[i] = i + 1;
    }
    x_dims[x.dims().size() - 1] = axis_dim;
    axis_vec[x.dims().size() - 1] = axis;

    x_temp.Resize(phi::make_ddim(x_dims));
    out_temp.Resize(phi::make_ddim(x_dims));
    dev_ctx.Alloc(&x_temp, x.dtype());
    dev_ctx.Alloc(&out_temp, x.dtype());
    sdaa_ops::doTransposeTensor(dev_ctx, x, axis_vec, &x_temp);

    sdaa_ops::doLogSoftmaxForward(dev_ctx, x_temp, -1, &out_temp);

    for (int i = axis + 1; i < x.dims().size(); i++) {
      axis_vec[i] = i - 1;
    }
    axis_vec[axis] = x_dims.size() - 1;
    sdaa_ops::doTransposeTensor(dev_ctx, out_temp, axis_vec, out);
  } else {
    sdaa_ops::doLogSoftmaxForward(dev_ctx, x, axis, out);
  }
}

template <typename T, typename Context>
void LogSoftmaxGradKernel(const Context& dev_ctx,
                          const phi::DenseTensor& out,
                          const phi::DenseTensor& out_grad,
                          int axis,
                          phi::DenseTensor* x_grad) {
  VLOG(4) << "Call SDAA LogSoftmaxGradKernel";
  const int rank = out.dims().size();
  dev_ctx.template Alloc<T>(x_grad);
  // For 0D Tensor
  if (rank == 0) {
    sdaa_ops::doFillTensor<T>(
        dev_ctx, static_cast<T>(0.0), x_grad->dtype(), x_grad);
    return;
  }

  if (axis < 0) {
    axis += out.dims().size();
  }
  if (axis != out.dims().size() - 1) {
    phi::DenseTensor out_temp;
    phi::DenseTensor out_grad_temp;
    phi::DenseTensor x_grad_temp;

    std::vector<int> out_dims = phi::vectorize<int>(out.dims());
    std::vector<int> axis_vec(out.dims().size());
    std::iota(axis_vec.begin(), axis_vec.end(), 0);
    // transpose axis_dim to last_dim
    int axis_dim = out_dims[axis];
    for (int i = axis; i < out.dims().size() - 1; i++) {
      out_dims[i] = out_dims[i + 1];
      axis_vec[i] = i + 1;
    }
    out_dims[out.dims().size() - 1] = axis_dim;
    axis_vec[out.dims().size() - 1] = axis;

    out_temp.Resize(phi::make_ddim(out_dims));
    dev_ctx.Alloc(&out_temp, out.dtype());

    out_grad_temp.Resize(phi::make_ddim(out_dims));
    dev_ctx.Alloc(&out_grad_temp, out.dtype());

    x_grad_temp.Resize(phi::make_ddim(out_dims));
    dev_ctx.Alloc(&x_grad_temp, out.dtype());

    sdaa_ops::doTransposeTensor(dev_ctx, out, axis_vec, &out_temp);
    sdaa_ops::doTransposeTensor(dev_ctx, out_grad, axis_vec, &out_grad_temp);
    sdaa_ops::doLogSoftmaxBackward(
        dev_ctx, out_temp, out_grad_temp, -1, &x_grad_temp);

    for (int i = axis + 1; i < out.dims().size(); i++) {
      axis_vec[i] = i - 1;
    }
    axis_vec[axis] = out_dims.size() - 1;
    sdaa_ops::doTransposeTensor(dev_ctx, x_grad_temp, axis_vec, x_grad);
  } else {
    sdaa_ops::doLogSoftmaxBackward(dev_ctx, out, out_grad, axis, x_grad);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(log_softmax,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::LogSoftmaxKernel,
                          float,
                          phi::dtype::float16) {}
PD_REGISTER_PLUGIN_KERNEL(log_softmax_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::LogSoftmaxGradKernel,
                          float,
                          phi::dtype::float16) {}
