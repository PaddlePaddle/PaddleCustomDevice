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

#include <numeric>

#include "kernels/funcs/high_precision_op_list.h"
#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"
namespace custom_kernel {

template <typename T, typename Context>
void SoftmaxKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   int axis,
                   phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA SoftmaxKernel";

  dev_ctx.template Alloc<T>(out);
  // For 0-Sized Tensor
  if (out->numel() == 0) {
    return;
  }
  // For 0D Tensor
  if (out->dims().size() == 0) {
    auto out_dims = out->dims();
    sdaa_ops::doFillTensor<T>(dev_ctx, static_cast<T>(0), out->dtype(), out);
    out->Resize(out_dims);
    return;
  }

  bool high_precision = false;
  if (is_in_high_precision_op_list("softmax")) high_precision = true;

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

    sdaa_ops::doSoftmaxForward(dev_ctx, x_temp, -1, high_precision, &out_temp);

    for (int i = axis + 1; i < x.dims().size(); i++) {
      axis_vec[i] = i - 1;
    }
    axis_vec[axis] = x_dims.size() - 1;
    sdaa_ops::doTransposeTensor(dev_ctx, out_temp, axis_vec, out);
  } else {
    sdaa_ops::doSoftmaxForward(dev_ctx, x, axis, high_precision, out);
  }
}

template <typename T, typename Context>
void SoftmaxGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& out,
                       const phi::DenseTensor& out_grad,
                       int axis,
                       phi::DenseTensor* x_grad) {
  VLOG(4) << "Call SDAA SoftmaxGradKernel";

  dev_ctx.template Alloc<T>(x_grad);

  // For 0-Sized Tensor
  if (x_grad->numel() == 0) {
    return;
  }
  // For 0D Tensor
  if (x_grad->dims().size() == 0) {
    auto x_grad_dims = x_grad->dims();
    sdaa_ops::doFillTensor<T>(
        dev_ctx, static_cast<T>(0), x_grad->dtype(), x_grad);
    x_grad->Resize(x_grad_dims);
    return;
  }

  if (axis < 0) {
    axis += out.dims().size();
  }

  bool high_precision = false;
  if (is_in_high_precision_op_list("softmax_grad")) high_precision = true;

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
    sdaa_ops::doSoftmaxBackward(
        dev_ctx, out_temp, out_grad_temp, -1, high_precision, &x_grad_temp);

    for (int i = axis + 1; i < out.dims().size(); i++) {
      axis_vec[i] = i - 1;
    }
    axis_vec[axis] = out_dims.size() - 1;
    sdaa_ops::doTransposeTensor(dev_ctx, x_grad_temp, axis_vec, x_grad);
  } else {
    sdaa_ops::doSoftmaxBackward(
        dev_ctx, out, out_grad, axis, high_precision, x_grad);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(softmax,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::SoftmaxKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(softmax_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::SoftmaxGradKernel,
                          float,
                          double,
                          phi::dtype::float16) {}
