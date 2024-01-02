// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "kernels/common_ops/common_ops.h"
#include "kernels/funcs/gcu_kernel_funcs.h"
#include "kernels/funcs/gcu_op_runner.h"

namespace custom_kernel {

inline phi::DDim GetUnsqueezeShape(const std::vector<int64_t> unsqz_dims,
                                   const phi::DDim& in_dims) {
  int output_size = in_dims.size() + static_cast<int>(unsqz_dims.size());
  int cur_output_size = in_dims.size();
  std::vector<int64_t> output_shape(output_size, 0);

  // Validity Check: rank range.
  PADDLE_ENFORCE_LE(
      output_size,
      6,
      phi::errors::InvalidArgument("The output "
                                   "tensor's rank should be less than 6."));

  for (int axis : unsqz_dims) {
    int cur = axis < 0 ? axis + cur_output_size + 1 : axis;
    // Vaildity Check: the axis bound
    PADDLE_ENFORCE_GE(
        cur,
        0,
        phi::errors::InvalidArgument("The insert dimension value should "
                                     "not be less than 0"));
    PADDLE_ENFORCE_LE(cur,
                      cur_output_size,
                      phi::errors::InvalidArgument(
                          "The insert dimension value shoule not be larger "
                          "than the dimension size of input tensor"));
    // Move old axis, and insert new axis
    for (int i = cur_output_size; i >= cur; --i) {
      if (output_shape[i] == 1) {
        // Move axis
        output_shape[i + 1] = 1;
        output_shape[i] = 0;
      }
    }
    output_shape[cur] = 1;
    // Add the output size.
    cur_output_size++;
  }

  // Make output shape
  for (int in_idx = 0, out_idx = 0; out_idx < output_size; ++out_idx) {
    if (output_shape[out_idx] == 0) {
      output_shape[out_idx] = in_dims[in_idx++];
    }
  }

  return phi::make_ddim(output_shape);
}

template <typename T, typename Context>
void UnsqueezeInferKernel(const Context& dev_ctx,
                          const phi::DenseTensor& x,
                          const phi::IntArray& axes,
                          phi::DenseTensor* out) {
  auto x_dims = x.dims();
  auto out_dims = out->dims();

  if (axes.FromTensor()) {
    out_dims = GetUnsqueezeShape(axes.GetData(), x_dims);
  }
  out->Resize(out_dims);
  dev_ctx.template Alloc<T>(out);
  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "unsqueeze_infer", unsqueeze_infer);
    if (out->data() == x.data()) {
      auto tmp = EmptyTensor(dev_ctx, out->meta());
      dev_ctx.template Alloc(&tmp, tmp.dtype());
      *out = tmp;
    }
    reshape(dev_ctx, x, *out);
    PADDLE_GCU_KERNEL_END("unsqueeze_infer", unsqueeze_infer);
  } else {
    custom_kernel::TensorCopy(dev_ctx, x, false, out);
    out->Resize(out_dims);  // copy will reset the dims.
  }
}

template <typename T, typename Context>
void UnsqueezeKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::IntArray& axes,
                     phi::DenseTensor* out,
                     phi::DenseTensor* xshape) {
  custom_kernel::UnsqueezeInferKernel<T, Context>(dev_ctx, x, axes, out);
}

template <typename T, typename Context>
void UnsqueezeGradKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x_shape,
                         const phi::DenseTensor& dout,
                         phi::DenseTensor* dx) {
  auto xshape_dims = x_shape.dims();
  auto x_dims = phi::slice_ddim(xshape_dims, 1, xshape_dims.size());

  dev_ctx.template Alloc<T>(dx);
  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "unsqueeze_grad", unsqueeze_grad);
    if (dx->data() == dout.data()) {
      auto tmp = EmptyTensor(dev_ctx, dx->meta());
      dev_ctx.template Alloc(&tmp, tmp.dtype());
      *dx = tmp;
    }
    reshape(dev_ctx, dout, *dx);
    PADDLE_GCU_KERNEL_END("unsqueeze_grad", unsqueeze_grad);
  } else {
    custom_kernel::TensorCopy(dev_ctx, dout, false, dx);
    dx->Resize(x_dims);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(unsqueeze_infer,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::UnsqueezeInferKernel,
                          float,
                          double,
                          phi::dtype::bfloat16,
                          bool,
                          int,
                          int16_t,
                          uint8_t,
                          int8_t,
                          int64_t,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}

PD_REGISTER_PLUGIN_KERNEL(unsqueeze,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::UnsqueezeKernel,
                          float,
                          double,
                          phi::dtype::bfloat16,
                          bool,
                          int,
                          int16_t,
                          uint8_t,
                          int8_t,
                          int64_t,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(unsqueeze_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::UnsqueezeGradKernel,
                          phi::dtype::bfloat16,
                          bool,
                          int,
                          int16_t,
                          uint8_t,
                          int8_t,
                          int64_t,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>,
                          float,
                          double,
                          phi::dtype::float16) {}
