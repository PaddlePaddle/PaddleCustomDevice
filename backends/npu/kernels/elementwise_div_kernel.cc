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
void DivideRawKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& y,
                     int axis,
                     phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  const auto& runner = NpuOpRunner("Div", {x, y}, {*out}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void DivideKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& y,
                  phi::DenseTensor* out) {
  int axis = -1;
  custom_kernel::DivideRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void DivideGradKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& y,
                      const phi::DenseTensor& out,
                      const phi::DenseTensor& dout,
                      int axis,
                      phi::DenseTensor* dx,
                      phi::DenseTensor* dy) {
  auto stream = dev_ctx.stream();

  if (dx) {
    dev_ctx.template Alloc<T>(dx);

    phi::DenseTensor tensor_one;
    phi::DenseTensorMeta tensor_one_meta = {y.dtype(), phi::make_ddim({1})};
    tensor_one.set_meta(tensor_one_meta);
    dev_ctx.template Alloc<T>(&tensor_one);
    FillNpuTensorWithConstant<float>(
        &tensor_one, dev_ctx, static_cast<float>(1.0));

    // Use `Div` CANN OP to achieve `1/y` instead of `Power` CANN OP.
    // Because `Power` will cause precision overflow, that is, `float_status`
    // will be set to 1.
    phi::DenseTensor y_div;
    phi::DenseTensorMeta y_div_meta = {y.dtype(), y.dims()};
    y_div.set_meta(y_div_meta);
    dev_ctx.template Alloc<T>(&y_div);
    const auto& runner_one_div_y =
        NpuOpRunner("Div", {tensor_one, y}, {y_div}, {});
    runner_one_div_y.Run(stream);

    phi::DenseTensor tensor_zeros;
    phi::DenseTensorMeta tensor_zeros_meta = {x.dtype(), x.dims()};
    tensor_zeros.set_meta(tensor_zeros_meta);
    dev_ctx.template Alloc<T>(&tensor_zeros);
    const auto& runner_tensor_zeros =
        NpuOpRunner("ZerosLike", {x}, {tensor_zeros}, {});
    runner_tensor_zeros.Run(stream);

    phi::DenseTensor x_zero;
    phi::DenseTensorMeta x_zero_meta = {paddle::experimental::DataType::BOOL,
                                        x.dims()};
    x_zero.set_meta(x_zero_meta);
    dev_ctx.template Alloc<bool>(&x_zero);
    const auto& runner_x_zero =
        NpuOpRunner("Equal", {x, tensor_zeros}, {x_zero}, {});
    runner_x_zero.Run(stream);

    phi::DenseTensor x_nozero;
    phi::DenseTensorMeta x_nozero_meta = {paddle::experimental::DataType::BOOL,
                                          x.dims()};
    x_nozero.set_meta(x_zero_meta);
    dev_ctx.template Alloc<bool>(&x_nozero);
    const auto& runner_x_nonzero =
        NpuOpRunner("LogicalNot", {x_zero}, {x_nozero}, {});
    runner_x_nonzero.Run(stream);

    phi::DenseTensor x_nozero_f;
    phi::DenseTensorMeta x_nozero_f_meta = {x.dtype(), x.dims()};
    x_nozero_f.set_meta(x_nozero_f_meta);
    dev_ctx.template Alloc<T>(&x_nozero_f);
    const auto& runner_x_nonzero_f =
        NpuOpRunner("Cast",
                    {x_nozero},
                    {x_nozero_f},
                    {{"dst_type", static_cast<int32_t>(0)}});
    runner_x_nonzero_f.Run(stream);

    phi::DenseTensor x_grad_w;
    phi::DenseTensorMeta x_grad_w_meta = {x.dtype(), x.dims()};
    x_grad_w.set_meta(x_grad_w_meta);
    dev_ctx.template Alloc<T>(&x_grad_w);
    const auto& runner_x_grad_w =
        NpuOpRunner("Mul", {x_nozero_f, y_div}, {x_grad_w}, {});
    runner_x_grad_w.Run(stream);

    const auto& runner_x_grad = NpuOpRunner("Mul", {x_grad_w, dout}, {*dx}, {});
    runner_x_grad.Run(stream);
  }

  if (dy) {
    dev_ctx.template Alloc<T>(dy);

    phi::DenseTensor neg_out;
    phi::DenseTensorMeta neg_out_meta = {out.dtype(), out.dims()};
    neg_out.set_meta(neg_out_meta);
    dev_ctx.template Alloc<T>(&neg_out);
    const auto& runner_neg_out = NpuOpRunner("Neg", {out}, {neg_out}, {});
    runner_neg_out.Run(stream);

    phi::DenseTensor tmp_mul;
    phi::DenseTensorMeta tmp_mul_meta = {out.dtype(), out.dims()};
    tmp_mul.set_meta(tmp_mul_meta);
    dev_ctx.template Alloc<T>(&tmp_mul);
    const auto& runner_mul = NpuOpRunner("Mul", {neg_out, dout}, {tmp_mul}, {});
    runner_mul.Run(stream);

    if (dy->dims() != dout.dims()) {
      phi::DenseTensor reduced_tmp_mul;
      phi::DenseTensorMeta reduced_tmp_mul_meta = {y.dtype(), y.dims()};
      reduced_tmp_mul.set_meta(reduced_tmp_mul_meta);
      dev_ctx.template Alloc<T>(&reduced_tmp_mul);

      std::vector<int64_t> axes;
      int64_t diff = dout.dims().size() - dy->dims().size();
      for (int64_t i = 0; i < dout.dims().size(); ++i) {
        if (i < diff) {
          axes.push_back(i);
          continue;
        }
        if (dout.dims()[i] > dy->dims()[i - diff]) {
          axes.push_back(i);
        }
      }
      const auto& runner_reduce =
          NpuOpRunner("ReduceSumD",
                      {tmp_mul},
                      {reduced_tmp_mul},
                      {{"axes", axes}, {"keep_dims", false}});
      runner_reduce.Run(stream);

      const auto& runner_y_grad =
          NpuOpRunner("Div", {reduced_tmp_mul, y}, {*dy}, {});
      runner_y_grad.Run(stream);
    } else {
      const auto& runner_y_grad = NpuOpRunner("Div", {tmp_mul, y}, {*dy}, {});
      runner_y_grad.Run(stream);
    }
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(divide_raw,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::DivideRawKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(divide,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::DivideKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(divide_grad,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::DivideGradKernel,
                          float,
                          phi::dtype::float16,
                          double) {}
