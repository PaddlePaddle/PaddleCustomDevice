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

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void ConcatKernel(const Context& dev_ctx,
                  const std::vector<const phi::DenseTensor*>& x,
                  const phi::Scalar& axis_scalar,
                  phi::DenseTensor* out);
template <typename T, typename Context>
void SplitKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const phi::IntArray& num_or_sections,
                 const phi::Scalar& axis_scalar,
                 std::vector<phi::DenseTensor*> outs);

template <typename T, typename Context>
void SwiGluKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const paddle::optional<phi::DenseTensor>& y,
                  phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  const auto& dims = x.dims();
  int64_t axis = -1;

  if (y) {
    const auto& y_tensor = y.get();
    const auto& y_dims = y_tensor.dims();
    PADDLE_ENFORCE_EQ(
        y_dims,
        dims,
        phi::errors::InvalidArgument("The shape of Input(Y):[%s] must be equal "
                                     "to the shape of Input(X):[%s].",
                                     y_dims,
                                     dims));
    // Concatenate x and y in the - 1 dimension
    phi::DenseTensor concat_xy;
    auto dst_dims = y_dims;
    phi::Scalar concat_dim = -1;
    dst_dims[y_dims.size() - 1] = y_dims[y_dims.size() - 1] * 2;
    concat_xy.Resize(dst_dims);
    dev_ctx.template Alloc<T>(&concat_xy);

    std::vector<const phi::DenseTensor*> in_tensors{&x, &y_tensor};
    custom_kernel::ConcatKernel<T, Context>(
        dev_ctx, in_tensors, &concat_dim, &concat_xy);
    EXEC_NPU_CMD(aclnnSwiGlu, dev_ctx, concat_xy, axis, *out);

  } else {
    // check dims, divide x into two equal parts
    auto dims_2d = flatten_to_2d(dims, dims.size() - 1);
    int64_t n = dims_2d[1];
    PADDLE_ENFORCE_EQ(n % 2,
                      0,
                      phi::errors::InvalidArgument(
                          "The last dim of Input(X) should be exactly divided "
                          "by 2 when Input(Y) is None, but got %d",
                          n));
    EXEC_NPU_CMD(aclnnSwiGlu, dev_ctx, x, axis, *out);
  }
}

template <typename T, typename Context>
void SwiGluGradKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const paddle::optional<phi::DenseTensor>& y,
                      const phi::DenseTensor& dout,
                      phi::DenseTensor* dx,
                      phi::DenseTensor* dy) {
  const auto& dims = x.dims();
  int64_t axis = -1;
  if (y) {
    if (dx || dy) {
      const auto& y_tensor = y.get();
      const auto& y_dims = y_tensor.dims();
      PADDLE_ENFORCE_EQ(y_dims,
                        dims,
                        phi::errors::InvalidArgument(
                            "The shape of Input(Y):[%s] must be equal "
                            "to the shape of Input(X):[%s].",
                            y_dims,
                            dims));
      // Concatenate x and y in the - 1 dimension
      phi::DenseTensor concat_xy;
      auto dst_dims = y_dims;
      phi::Scalar concat_dim = -1;
      dst_dims[y_dims.size() - 1] = y_dims[y_dims.size() - 1] * 2;
      concat_xy.Resize(dst_dims);
      dev_ctx.template Alloc<T>(&concat_xy);

      phi::DenseTensor dx_temp;
      dx_temp.Resize(concat_xy.dims());
      dev_ctx.template Alloc<T>(&dx_temp);

      std::vector<const phi::DenseTensor*> in_tensors{&x, &y_tensor};
      custom_kernel::ConcatKernel<T, Context>(
          dev_ctx, in_tensors, &concat_dim, &concat_xy);
      EXEC_NPU_CMD(aclnnSwiGluGrad, dev_ctx, dout, concat_xy, axis, dx_temp);
      auto num_or_sections = phi::IntArray({2});
      auto axis_scalar = phi::Scalar(-1);
      if (dx && dy) {
        dev_ctx.template Alloc<T>(dx);
        dev_ctx.template Alloc<T>(dy);
        std::vector<phi::DenseTensor*> outs_d = {dx, dy};
        custom_kernel::SplitKernel<T, Context>(
            dev_ctx, dx_temp, num_or_sections, axis_scalar, outs_d);
      } else if (dx) {
        dev_ctx.template Alloc<T>(dx);
        phi::DenseTensor dx_fill;
        dx_fill.Resize(dx->dims());
        dev_ctx.template Alloc<T>(&dx_fill);
        std::vector<phi::DenseTensor*> outs_d = {dx, &dx_fill};
        custom_kernel::SplitKernel<T, Context>(
            dev_ctx, dx_temp, num_or_sections, axis_scalar, outs_d);
      } else if (dy) {
        dev_ctx.template Alloc<T>(dy);
        phi::DenseTensor dy_fill;
        dy_fill.Resize(dy->dims());
        dev_ctx.template Alloc<T>(&dy_fill);
        std::vector<phi::DenseTensor*> outs_d = {&dy_fill, dy};
        custom_kernel::SplitKernel<T, Context>(
            dev_ctx, dx_temp, num_or_sections, axis_scalar, outs_d);
      }
    }
    return;
  }
  if (dx) {
    dev_ctx.template Alloc<T>(dx);
    auto dims_2d = flatten_to_2d(dims, dims.size() - 1);
    int64_t n = dims_2d[1];
    PADDLE_ENFORCE_EQ(n % 2,
                      0,
                      phi::errors::InvalidArgument(
                          "The last dim of Input(X) should be exactly divided "
                          "by 2 when Input(Y) is None, but got %d",
                          n));
    EXEC_NPU_CMD(aclnnSwiGluGrad, dev_ctx, dout, x, axis, *dx);
  }
}

}  // namespace custom_kernel
PD_REGISTER_PLUGIN_KERNEL(swiglu,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SwiGluKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(swiglu_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SwiGluGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
