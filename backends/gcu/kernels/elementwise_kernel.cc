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
#include "kernels/common_ops/elementwise_ops.h"
#include "kernels/common_ops/reduce_x_ops.h"
#include "kernels/funcs/gcu_kernel_funcs.h"
#include "kernels/funcs/gcu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void ElementBaseKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& y,
                       int axis,
                       phi::DenseTensor* out,
                       const std::string& op_type) {
  dev_ctx.template Alloc<T>(out);

  TensorNameMap input_names;
  input_names["X"] = {"x"};
  input_names["Y"] = {"y"};

  TensorValueMap inputs;
  inputs["X"] = {const_cast<DenseTensor*>(&x)};
  inputs["Y"] = {const_cast<DenseTensor*>(&y)};

  TensorNameMap output_names;
  output_names["Out"] = {"out"};

  TensorValueMap outputs;
  outputs["Out"] = {out};

  GcuAttributeMap attrs;
  attrs["axis"] = axis;

  GcuRunner(
      input_names, inputs, output_names, outputs, attrs, op_type, dev_ctx);
}

template <typename T, typename Context>
void ElementBaseGradKernel(const Context& dev_ctx,
                           const phi::DenseTensor& x,
                           const phi::DenseTensor& y,
                           const phi::DenseTensor& dout,
                           int axis,
                           phi::DenseTensor* dx,
                           phi::DenseTensor* dy,
                           const std::string& op_type) {
  TensorNameMap input_names;
  input_names["X"] = {"x"};
  input_names["Y"] = {"y"};
  input_names[GradVarName("Out")] = {"dout"};

  VLOG(6) << op_type << " input x shape: " << x.dims().to_str()
          << " initialized: " << x.initialized();

  VLOG(6) << op_type << " input y shape: " << y.dims().to_str()
          << " initialized: " << y.initialized();

  phi::DenseTensor input_x_tmp;
  phi::DenseTensor input_y_tmp;

  phi::DenseTensor* input_x = const_cast<phi::DenseTensor*>(&x);
  phi::DenseTensor* input_y = const_cast<phi::DenseTensor*>(&y);
  if (!x.initialized()) {
    input_x_tmp.set_meta(x.meta());
    dev_ctx.template Alloc<T>(&input_x_tmp);
    input_x = &input_x_tmp;
  }
  if (!y.initialized()) {
    input_y_tmp.set_meta(y.meta());
    dev_ctx.template Alloc<T>(&input_y_tmp);
    input_y = &input_y_tmp;
  }

  TensorValueMap inputs;
  inputs["X"] = {input_x};
  inputs["Y"] = {input_y};
  inputs[GradVarName("Out")] = {const_cast<DenseTensor*>(&dout)};

  TensorNameMap output_names;
  TensorValueMap outputs;
  if (dx) {
    dev_ctx.template Alloc<T>(dx);
    output_names[GradVarName("X")] = {"dx"};
    outputs[GradVarName("X")] = {dx};
  }
  if (dy) {
    dev_ctx.template Alloc<T>(dy);
    output_names[GradVarName("Y")] = {"dy"};
    outputs[GradVarName("Y")] = {dy};
  }

  GcuAttributeMap attrs;
  attrs["axis"] = axis;

  GcuRunner(
      input_names, inputs, output_names, outputs, attrs, op_type, dev_ctx);
}

template <typename T, typename Context>
void AddKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::DenseTensor& y,
               phi::DenseTensor* out) {
  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "add", add);
    dev_ctx.template Alloc<T>(out);
    phi::DenseTensor tmp_x = x;
    phi::DenseTensor tmp_y = y;
    if (x.dtype() == phi::DataType::INT64) {
      tmp_x = cast(dev_ctx, x, phi::DataType::INT32);
    }
    if (y.dtype() == phi::DataType::INT64) {
      tmp_y = cast(dev_ctx, y, phi::DataType::INT32);
    }
    phi::DenseTensor tmp_out = *out;
    if (tmp_out.dtype() == phi::DataType::INT64) {
      auto tmp = EmptyTensor(dev_ctx, phi::DataType::INT32, tmp_out.dims());
      dev_ctx.template Alloc(&tmp, tmp.dtype());
      tmp_out = tmp;
    }
    add_compute(static_cast<const phi::CustomContext&>(dev_ctx),
                tmp_x,
                tmp_y,
                &tmp_out);
    if (out->dtype() == phi::DataType::INT64) {
      cast(dev_ctx, tmp_out, phi::DataType::INT64, out);
    }
    PADDLE_GCU_KERNEL_END("add", add);
  } else {
    ElementBaseKernel<T, Context>(dev_ctx, x, y, -1, out, "elementwise_add");
  }
}

template <typename T, typename Context>
void AddGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& y,
                   const phi::DenseTensor& dout,
                   int axis,
                   phi::DenseTensor* dx,
                   phi::DenseTensor* dy) {
  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "add_grad", add_grad);
    if (axis == -1) {
      int32_t lhs_rank = x.dims().size();
      int32_t rhs_rank = y.dims().size();
      axis = std::abs(lhs_rank - rhs_rank);
    }

    if (dx) {
      dev_ctx.template Alloc<T>(dx);
      if (x.dims() != dout.dims()) {
        std::vector<int64_t> reduce_axes;
        std::vector<int64_t> dst_shapes;
        std::vector<int64_t> dy_shapes = phi::vectorize(x.dims());
        std::vector<int64_t> dout_shapes = phi::vectorize(dout.dims());
        int32_t dy_axis = dy_shapes.size() < dout_shapes.size() ? axis : 0;
        int32_t dy_rank = dy_shapes.size();
        int32_t dout_rank = dout_shapes.size();
        for (int32_t i = 0; i < dout_rank; ++i) {
          if ((i < dy_axis || i >= dy_axis + dy_rank) ||
              (dout_shapes[i] > 1 && dy_shapes[i - dy_axis] == 1)) {
            reduce_axes.push_back(i);
          } else {
            dst_shapes.push_back(dout_shapes[i]);
          }
        }
        if (dst_shapes.size() == 0) {
          dst_shapes.push_back(1);
        }
        *dx = reduce_sum_compute(dev_ctx, dout, false, reduce_axes);

        if (x.dims() != dx->dims()) {
          *dx = reshape(dev_ctx, *dx, dy_shapes);
        }
      } else {
        *dx = dout;
      }
    }

    if (dy) {
      dev_ctx.template Alloc<T>(dy);
      if (y.dims() != dout.dims()) {
        std::vector<int64_t> reduce_axes;
        std::vector<int64_t> dst_shapes;
        std::vector<int64_t> dy_shapes = phi::vectorize(y.dims());
        std::vector<int64_t> dout_shapes = phi::vectorize(dout.dims());
        int32_t dy_axis = dy_shapes.size() < dout_shapes.size() ? axis : 0;
        int32_t dy_rank = dy_shapes.size();
        int32_t dout_rank = dout_shapes.size();
        for (int32_t i = 0; i < dout_rank; ++i) {
          if ((i < dy_axis || i >= dy_axis + dy_rank) ||
              (dout_shapes[i] > 1 && dy_shapes[i - dy_axis] == 1)) {
            reduce_axes.push_back(i);
          } else {
            dst_shapes.push_back(dout_shapes[i]);
          }
        }
        if (dst_shapes.size() == 0) {
          dst_shapes.push_back(1);
        }
        *dy = reduce_sum_compute(dev_ctx, dout, false, reduce_axes);

        if (y.dims() != dy->dims()) {
          *dy = reshape(dev_ctx, *dy, dy_shapes);
        }
      } else {
        *dy = dout;
      }
    }
    PADDLE_GCU_KERNEL_END("add_grad", add_grad);
  } else {
    ElementBaseGradKernel<T, Context>(
        dev_ctx, x, y, dout, axis, dx, dy, "elementwise_add_grad");
  }
}

template <typename T, typename Context>
void DivideKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& y,
                  phi::DenseTensor* out) {
  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "divide", divide);
    dev_ctx.template Alloc<T>(out);
    phi::DenseTensor tmp_x = x;
    phi::DenseTensor tmp_y = y;
    if (x.dtype() == phi::DataType::INT64) {
      tmp_x = cast(dev_ctx, x, phi::DataType::INT32);
    }
    if (y.dtype() == phi::DataType::INT64) {
      tmp_y = cast(dev_ctx, y, phi::DataType::INT32);
    }
    phi::DenseTensor tmp_out = *out;
    if (tmp_out.dtype() == phi::DataType::INT64) {
      auto tmp = EmptyTensor(dev_ctx, phi::DataType::INT32, tmp_out.dims());
      dev_ctx.template Alloc(&tmp, tmp.dtype());
      tmp_out = tmp;
    }
    div_compute(static_cast<const phi::CustomContext&>(dev_ctx),
                tmp_x,
                tmp_y,
                &tmp_out);
    if (out->dtype() == phi::DataType::INT64) {
      cast(dev_ctx, tmp_out, phi::DataType::INT64, out);
    }
    PADDLE_GCU_KERNEL_END("divide", divide);
  } else {
    ElementBaseKernel<T, Context>(dev_ctx, x, y, -1, out, "elementwise_div");
  }
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
  ElementBaseGradKernel<T, Context>(
      dev_ctx, x, y, dout, axis, dx, dy, "elementwise_div_grad");
}

template <typename T, typename Context>
void SubtractKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    phi::DenseTensor* out) {
  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "subtract", subtract);
    dev_ctx.template Alloc<T>(out);
    phi::DenseTensor tmp_x = x;
    phi::DenseTensor tmp_y = y;
    if (x.dtype() == phi::DataType::INT64) {
      tmp_x = cast(dev_ctx, x, phi::DataType::INT32);
    }
    if (y.dtype() == phi::DataType::INT64) {
      tmp_y = cast(dev_ctx, y, phi::DataType::INT32);
    }
    phi::DenseTensor tmp_out = *out;
    if (tmp_out.dtype() == phi::DataType::INT64) {
      auto tmp = EmptyTensor(dev_ctx, phi::DataType::INT32, tmp_out.dims());
      dev_ctx.template Alloc(&tmp, tmp.dtype());
      tmp_out = tmp;
    }
    sub_compute(static_cast<const phi::CustomContext&>(dev_ctx),
                tmp_x,
                tmp_y,
                &tmp_out);
    if (out->dtype() == phi::DataType::INT64) {
      cast(dev_ctx, tmp_out, phi::DataType::INT64, out);
    }
    PADDLE_GCU_KERNEL_END("subtract", subtract);
  } else {
    ElementBaseKernel<T, Context>(dev_ctx, x, y, -1, out, "elementwise_sub");
  }
}

template <typename T, typename Context>
void SubtractGradKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& y,
                        const phi::DenseTensor& dout,
                        int axis,
                        phi::DenseTensor* dx,
                        phi::DenseTensor* dy) {
  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "subtract_grad", subtract_grad);
    if (axis == -1) {
      int32_t lhs_rank = x.dims().size();
      int32_t rhs_rank = y.dims().size();
      axis = std::abs(lhs_rank - rhs_rank);
    }
    if (dx) {
      if (x.dims() != dout.dims()) {
        std::vector<int64_t> reduce_axes;
        std::vector<int64_t> dst_shapes;
        std::vector<int64_t> dx_shapes = phi::vectorize(x.dims());
        std::vector<int64_t> dout_shapes = phi::vectorize(dout.dims());
        int32_t dx_axis = dx_shapes.size() < dout_shapes.size() ? axis : 0;
        int32_t dx_rank = dx_shapes.size();
        int32_t dout_rank = dout_shapes.size();
        for (int32_t i = 0; i < dout_rank; ++i) {
          if (i < dx_axis || i >= dx_axis + dx_rank ||
              (dout_shapes[i] > 1 && dx_shapes[i - dx_axis] == 1)) {
            reduce_axes.push_back(i);
          } else {
            dst_shapes.push_back(dout_shapes[i]);
          }
        }
        if (dst_shapes.size() == 0) {
          dst_shapes.push_back(1);
        }
        *dx = reduce_sum_compute(dev_ctx, dout, false, reduce_axes);
        if (dx_shapes != phi::vectorize(dx->dims())) {
          *dx = reshape(dev_ctx, *dx, dx_shapes);
        }
      } else {
        *dx = dout;
      }
    }
    if (dy) {
      auto tmp_out_grad = neg_compute(dev_ctx, dout);
      if (y.dims() != dout.dims()) {
        std::vector<int64_t> reduce_axes;
        std::vector<int64_t> dst_shapes;
        std::vector<int64_t> dy_shapes = phi::vectorize(y.dims());
        std::vector<int64_t> dout_shapes = phi::vectorize(dout.dims());
        int32_t dy_axis = dy_shapes.size() < dout_shapes.size() ? axis : 0;
        int32_t dy_rank = dy_shapes.size();
        int32_t dout_rank = dout_shapes.size();
        for (int32_t i = 0; i < dout_rank; ++i) {
          if ((i < dy_axis || i >= dy_axis + dy_rank) ||
              (dout_shapes[i] > 1 && dy_shapes[i - dy_axis] == 1)) {
            reduce_axes.push_back(i);
          } else {
            dst_shapes.push_back(dout_shapes[i]);
          }
        }
        if (dst_shapes.size() == 0) {
          dst_shapes.push_back(1);
        }
        *dy = reduce_sum_compute(dev_ctx, tmp_out_grad, false, reduce_axes);
        if (dy_shapes != phi::vectorize(dy->dims())) {
          *dy = reshape(dev_ctx, *dy, dy_shapes);
        }
      } else {
        *dy = tmp_out_grad;
      }
    }
    PADDLE_GCU_KERNEL_END("subtract_grad", subtract_grad);
  } else {
    ElementBaseGradKernel<T, Context>(
        dev_ctx, x, y, dout, axis, dx, dy, "elementwise_sub_grad");
  }
}

template <typename T, typename Context>
void MultiplyKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    phi::DenseTensor* out) {
  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "multiply", multiply);
    dev_ctx.template Alloc<T>(out);
    phi::DenseTensor tmp_x = x;
    phi::DenseTensor tmp_y = y;
    if (x.dtype() == phi::DataType::INT64) {
      tmp_x = cast(dev_ctx, x, phi::DataType::INT32);
    }
    if (y.dtype() == phi::DataType::INT64) {
      tmp_y = cast(dev_ctx, y, phi::DataType::INT32);
    }
    phi::DenseTensor tmp_out = *out;
    if (tmp_out.dtype() == phi::DataType::INT64) {
      auto tmp = EmptyTensor(dev_ctx, phi::DataType::INT32, tmp_out.dims());
      dev_ctx.template Alloc(&tmp, tmp.dtype());
      tmp_out = tmp;
    }
    mul_compute(static_cast<const phi::CustomContext&>(dev_ctx),
                tmp_x,
                tmp_y,
                &tmp_out);
    if (out->dtype() == phi::DataType::INT64) {
      cast(dev_ctx, tmp_out, phi::DataType::INT64, out);
    }
    PADDLE_GCU_KERNEL_END("multiply", multiply);
  } else {
    ElementBaseKernel<T, Context>(dev_ctx, x, y, -1, out, "elementwise_mul");
  }
}

template <typename T, typename Context>
void MultiplyGradKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& y,
                        const phi::DenseTensor& dout,
                        int axis,
                        phi::DenseTensor* dx,
                        phi::DenseTensor* dy) {
  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "multiply_grad", multiply_grad);
    phi::DenseTensor tmp_dout = dout;

    int32_t lhs_rank = x.dims().size();
    int32_t rhs_rank = y.dims().size();
    if (axis == -1) {
      axis = std::abs(lhs_rank - rhs_rank);
    }

    if (dx) {
      std::vector<int64_t> rb_dims(rhs_rank);
      std::iota(rb_dims.begin(), rb_dims.begin() + rhs_rank, axis);
      auto tmp_y = y;
      if (rhs_rank != 0 && rhs_rank != lhs_rank) {
        tmp_y = broadcast_in_dim(
            dev_ctx, y, phi::vectorize(tmp_dout.dims()), rb_dims);
      }
      *dx = mul_compute(dev_ctx, tmp_dout, tmp_y);
    }

    if (dy) {
      auto tmp_out_grad = mul_compute(dev_ctx, tmp_dout, x);
      if (y.dims() != tmp_dout.dims()) {
        std::vector<int64_t> reduce_axes;
        std::vector<int64_t> dst_shapes;
        std::vector<int64_t> dy_shapes = phi::vectorize(y.dims());
        std::vector<int64_t> dout_shapes = phi::vectorize(tmp_dout.dims());
        int32_t dy_axis = dy_shapes.size() < dout_shapes.size() ? axis : 0;
        int32_t dy_rank = dy_shapes.size();
        int32_t dout_rank = dout_shapes.size();
        for (int32_t i = 0; i < dout_rank; ++i) {
          if ((i < dy_axis || i >= dy_axis + dy_rank) ||
              (dout_shapes[i] > 1 && dy_shapes[i - dy_axis] == 1)) {
            reduce_axes.push_back(i);
          } else {
            dst_shapes.push_back(dout_shapes[i]);
          }
        }
        if (dst_shapes.size() == 0) {
          dst_shapes.push_back(1);
        }
        *dy = reduce_sum_compute(dev_ctx, tmp_out_grad, false, reduce_axes);
        if (dy_shapes != phi::vectorize(dy->dims())) {
          *dy = reshape(dev_ctx, *dy, dy_shapes);
        }
      } else {
        *dy = tmp_out_grad;
      }
    }

    PADDLE_GCU_KERNEL_END("multiply_grad", multiply_grad);
  } else {
    ElementBaseGradKernel<T, Context>(
        dev_ctx, x, y, dout, axis, dx, dy, "elementwise_mul_grad");
  }
}

template <typename T, typename Context>
void MinimumKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& y,
                   phi::DenseTensor* out) {
  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "minimum", minimum);
    dev_ctx.template Alloc<T>(out);
    phi::DenseTensor tmp_x = x;
    phi::DenseTensor tmp_y = y;
    if (x.dtype() == phi::DataType::INT64) {
      tmp_x = cast(dev_ctx, x, phi::DataType::INT32);
    }
    if (y.dtype() == phi::DataType::INT64) {
      tmp_y = cast(dev_ctx, y, phi::DataType::INT32);
    }
    phi::DenseTensor tmp_out = *out;
    if (tmp_out.dtype() == phi::DataType::INT64) {
      auto tmp = EmptyTensor(dev_ctx, phi::DataType::INT32, tmp_out.dims());
      dev_ctx.template Alloc(&tmp, tmp.dtype());
      tmp_out = tmp;
    }
    minimum_compute(static_cast<const phi::CustomContext&>(dev_ctx),
                    tmp_x,
                    tmp_y,
                    &tmp_out);
    if (out->dtype() == phi::DataType::INT64) {
      cast(dev_ctx, tmp_out, phi::DataType::INT64, out);
    }
    PADDLE_GCU_KERNEL_END("minimum", minimum);
  } else {
    ElementBaseKernel<T, Context>(dev_ctx, x, y, -1, out, "elementwise_min");
  }
}

template <typename T, typename Context>
void MinimumGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& y,
                       const phi::DenseTensor& dout,
                       phi::DenseTensor* dx,
                       phi::DenseTensor* dy) {
  ElementBaseGradKernel<T, Context>(
      dev_ctx, x, y, dout, -1, dx, dy, "elementwise_min_grad");
}

template <typename T, typename Context>
void MaximumKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& y,
                   phi::DenseTensor* out) {
  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "maximum", maximum);
    dev_ctx.template Alloc<T>(out);
    phi::DenseTensor tmp_x = x;
    phi::DenseTensor tmp_y = y;
    if (x.dtype() == phi::DataType::INT64) {
      tmp_x = cast(dev_ctx, x, phi::DataType::INT32);
    }
    if (y.dtype() == phi::DataType::INT64) {
      tmp_y = cast(dev_ctx, y, phi::DataType::INT32);
    }
    phi::DenseTensor tmp_out = *out;
    if (tmp_out.dtype() == phi::DataType::INT64) {
      auto tmp = EmptyTensor(dev_ctx, phi::DataType::INT32, tmp_out.dims());
      dev_ctx.template Alloc(&tmp, tmp.dtype());
      tmp_out = tmp;
    }
    maximum_compute(static_cast<const phi::CustomContext&>(dev_ctx),
                    tmp_x,
                    tmp_y,
                    &tmp_out);
    if (out->dtype() == phi::DataType::INT64) {
      cast(dev_ctx, tmp_out, phi::DataType::INT64, out);
    }
    PADDLE_GCU_KERNEL_END("maximum", maximum);
  } else {
    ElementBaseKernel<T, Context>(dev_ctx, x, y, -1, out, "elementwise_max");
  }
}

template <typename T, typename Context>
void MaximumGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& y,
                       const phi::DenseTensor& dout,
                       phi::DenseTensor* dx,
                       phi::DenseTensor* dy) {
  ElementBaseGradKernel<T, Context>(
      dev_ctx, x, y, dout, -1, dx, dy, "elementwise_max_grad");
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(add,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::AddKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(add_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::AddGradKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(divide,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::DivideKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(divide_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::DivideGradKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(multiply,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::MultiplyKernel,
                          int8_t,
                          int32_t,
                          int64_t,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(multiply_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::MultiplyGradKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(subtract,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::SubtractKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(subtract_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::SubtractGradKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(maximum,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::MaximumKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(maximum_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::MaximumGradKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(minimum,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::MinimumKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(minimum_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::MinimumGradKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          double) {}
