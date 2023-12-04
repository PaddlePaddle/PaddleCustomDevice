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

#include "common/common.h"
#include "common/utils.h"
#include "kernels/common_ops/common_ops.h"
#include "kernels/funcs/gcu_kernel_funcs.h"
#include "kernels/funcs/gcu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void SqueezeInferKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::IntArray& axes_int_array,
                        phi::DenseTensor* out) {
  auto out_dims = out->dims();
  out->Resize(out_dims);
  dev_ctx.template Alloc<T>(out);
  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "squeeze_infer", squeeze_infer);
    if (out->data() == x.data()) {
      auto tmp = EmptyTensor(dev_ctx, out->meta());
      dev_ctx.template Alloc(&tmp, tmp.dtype());
      *out = tmp;
    }
    reshape(dev_ctx, x, *out);
    PADDLE_GCU_KERNEL_END("squeeze_infer", squeeze_infer);
  } else {
    TensorCopy(dev_ctx, x, false, out);
    out->Resize(out_dims);
  }
}

template <typename T, typename Context>
void SqueezeKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::IntArray& axes_int_array,
                   phi::DenseTensor* out,
                   phi::DenseTensor* xshape) {
  custom_kernel::SqueezeInferKernel<T, Context>(
      dev_ctx, x, axes_int_array, out);
}

template <typename T, typename Context>
void SqueezeGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& xshape,
                       const phi::DenseTensor& dout,
                       const phi::IntArray& axes_int_array,
                       phi::DenseTensor* dx) {
  auto xshape_dims = xshape.dims();
  auto x_dims = phi::slice_ddim(xshape_dims, 1, xshape_dims.size());
  dx->Resize(x_dims);
  dev_ctx.template Alloc<T>(dx);

  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "squeeze_grad", unsqueeze_grad);
    if (dx->data() == dout.data()) {
      auto tmp = EmptyTensor(dev_ctx, dx->meta());
      dev_ctx.template Alloc(&tmp, tmp.dtype());
      *dx = tmp;
    }
    reshape(dev_ctx, dout, *dx);
    PADDLE_GCU_KERNEL_END("squeeze_grad", unsqueeze_grad);
  } else {
    TensorCopy(dev_ctx, dout, false, dx);
    dx->Resize(x_dims);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(squeeze_infer,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::SqueezeInferKernel,
                          bool,
                          int,
                          uint8_t,
                          int8_t,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(squeeze,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::SqueezeKernel,
                          bool,
                          int,
                          uint8_t,
                          int8_t,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(squeeze_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::SqueezeGradKernel,
                          bool,
                          int,
                          uint8_t,
                          int8_t,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          double) {}
