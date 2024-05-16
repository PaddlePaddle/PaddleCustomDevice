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

#include "common/gcu_op_runner.h"
#include "kernels/funcs/gcu_kernel_funcs.h"

namespace custom_kernel {
namespace {
inline phi::DDim GetDimsWithAxis(const phi::DDim& x_dims,
                                 const phi::DDim& y_dims,
                                 const int axis) {
  std::vector<int64_t> y_shape(x_dims.size(), 1);
  for (int i = 0; i < axis; ++i) {
    y_shape[i] = 1;
  }
  for (int i = 0; i < y_dims.size(); ++i) {
    bool is_common_boardcast = x_dims[i + axis] == y_dims[i]
                                   ? true
                                   : (y_dims[i] == 1 || x_dims[i + axis] == 1);
    PADDLE_ENFORCE_EQ(is_common_boardcast,
                      true,
                      phi::errors::InvalidArgument(
                          "Broadcast dimension mismatch. Operands "
                          "could not be broadcast together with the shape of "
                          "X = [%s] and the shape of Y = [%s]. Received [%d] "
                          "in X is not equal to [%d] in Y.",
                          x_dims,
                          y_dims,
                          x_dims[i + axis],
                          y_dims[i]));
    y_shape[i + axis] = y_dims[i];
  }
  for (int i = axis + y_dims.size(); i < x_dims.size(); ++i) {
    y_shape[i] = 1;
  }
  return phi::make_ddim(y_shape);
}
}  // namespace

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
  PADDLE_GCU_KERNEL_TRACE("add");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    auto scalar = phi::Scalar(1.0f);
    phi::DenseTensor input_x = MaybeCreateOrTrans64To32bits(dev_ctx, x);
    phi::DenseTensor input_y = MaybeCreateOrTrans64To32bits(dev_ctx, y);
    phi::DenseTensor output_z =
        MaybeCreateOrTrans64To32bits(dev_ctx, *out, false);
    LAUNCH_TOPSATENOP(topsatenAdd, dev_ctx, output_z, input_x, input_y, scalar);
    MaybeTransResult(dev_ctx, output_z, out);

  } else {  // kernel impl base on JIT
    ElementBaseKernel<T, Context>(dev_ctx, x, y, -1, out, "elementwise_add");
  }
}

template <typename T, typename Context>
void AddRawKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& y,
                  int axis,
                  phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("add_raw");
  if (axis == -1) {
    custom_kernel::AddKernel<T, Context>(dev_ctx, x, y, out);
    return;
  }
  if (LaunchAOTKernel()) {
    PADDLE_ENFORCE_GE(
        axis,
        0,
        phi::errors::InvalidArgument(
            "axis should greater or equal than 0, but get %d.", axis));
    dev_ctx.template Alloc<T>(out);
    auto x_dims = x.dims();
    auto y_dims = y.dims();
    phi::DenseTensor x_tensor(x);
    phi::DenseTensor y_tensor(y);
    if (x_dims.size() >= y_dims.size()) {
      y_tensor.Resize(GetDimsWithAxis(x_dims, y_dims, axis));
    } else {
      x_tensor.Resize(GetDimsWithAxis(y_dims, x_dims, axis));
    }
    auto scalar = phi::Scalar(1.0f);
    phi::DenseTensor input_x = MaybeCreateOrTrans64To32bits(dev_ctx, x_tensor);
    phi::DenseTensor input_y = MaybeCreateOrTrans64To32bits(dev_ctx, y_tensor);
    phi::DenseTensor output_z =
        MaybeCreateOrTrans64To32bits(dev_ctx, *out, false);
    LAUNCH_TOPSATENOP(topsatenAdd, dev_ctx, output_z, input_x, input_y, scalar);
    MaybeTransResult(dev_ctx, output_z, out);

  } else {  // kernel impl base on JIT
    ElementBaseKernel<T, Context>(dev_ctx, x, y, axis, out, "elementwise_add");
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
  PADDLE_GCU_KERNEL_TRACE("add_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    ElementBaseGradKernel<T, Context>(
        dev_ctx, x, y, dout, axis, dx, dy, "elementwise_add_grad");
  }
}

template <typename T, typename Context>
void SubtractKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("subtract");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    auto scalar = phi::Scalar(1.0f);
    LAUNCH_TOPSATENOP(topsatenSub, dev_ctx, *out, x, y, scalar);

  } else {  // kernel impl base on JIT
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
  PADDLE_GCU_KERNEL_TRACE("subtract_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    ElementBaseGradKernel<T, Context>(
        dev_ctx, x, y, dout, axis, dx, dy, "elementwise_sub_grad");
  }
}

template <typename T, typename Context>
void MultiplyKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("multiply");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    phi::DenseTensor input_x = MaybeCreateOrTrans64To32bits(dev_ctx, x);
    phi::DenseTensor input_y = MaybeCreateOrTrans64To32bits(dev_ctx, y);
    phi::DenseTensor output_z =
        MaybeCreateOrTrans64To32bits(dev_ctx, *out, false);
    LAUNCH_TOPSATENOP(topsatenMul, dev_ctx, output_z, input_x, input_y);
    MaybeTransResult(dev_ctx, output_z, out);

  } else {  // kernel impl base on JIT
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
  PADDLE_GCU_KERNEL_TRACE("multiply_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    ElementBaseGradKernel<T, Context>(
        dev_ctx, x, y, dout, axis, dx, dy, "elementwise_mul_grad");
  }
}

template <typename T, typename Context>
void DivideKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& y,
                  phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("divide");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    LAUNCH_TOPSATENOP(topsatenDiv, dev_ctx, *out, x, y);

  } else {  // kernel impl base on JIT
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
  PADDLE_GCU_KERNEL_TRACE("divide_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    ElementBaseGradKernel<T, Context>(
        dev_ctx, x, y, dout, axis, dx, dy, "elementwise_div_grad");
  }
}

template <typename T, typename Context>
void MinimumKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& y,
                   phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("minimum");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
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
  PADDLE_GCU_KERNEL_TRACE("minimum_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    ElementBaseGradKernel<T, Context>(
        dev_ctx, x, y, dout, -1, dx, dy, "elementwise_min_grad");
  }
}

template <typename T, typename Context>
void MaximumKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& y,
                   phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("maximum");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
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
  PADDLE_GCU_KERNEL_TRACE("maximum_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    ElementBaseGradKernel<T, Context>(
        dev_ctx, x, y, dout, -1, dx, dy, "elementwise_max_grad");
  }
}

template <typename T, typename Context>
void ElementwisePowKernel(const Context& dev_ctx,
                          const phi::DenseTensor& x,
                          const phi::DenseTensor& y,
                          phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("elementwise_pow");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    phi::DenseTensor input_x = MaybeCreateOrTrans64To32bits(dev_ctx, x);
    phi::DenseTensor input_y = MaybeCreateOrTrans64To32bits(dev_ctx, y);
    phi::DenseTensor output_z =
        MaybeCreateOrTrans64To32bits(dev_ctx, *out, false);
    LAUNCH_TOPSATENOP(topsatenPow, dev_ctx, output_z, input_x, input_y);

    //   auto alpha1 = phi::Scalar(1.0f);
    //   auto alpha2 = phi::Scalar(1.0f);
    //   auto beta = phi::Scalar(0.0f);
    //   LAUNCH_TOPSOP(topsopPower,
    //                 dev_ctx,
    //                 output_z, input_x, input_y, alpha1, alpha2, beta);

    MaybeTransResult(dev_ctx, output_z, out);

  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(add_raw,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::AddRawKernel,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(add,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::AddKernel,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(add_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::AddGradKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(subtract,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::SubtractKernel,
                          int,
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

PD_REGISTER_PLUGIN_KERNEL(multiply,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::MultiplyKernel,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(multiply_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::MultiplyGradKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(divide,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::DivideKernel,
                          int,
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

// PD_REGISTER_PLUGIN_KERNEL(elementwise_pow,
//                           gcu,
//                           ALL_LAYOUT,
//                           custom_kernel::ElementwisePowKernel,
//                           int,
//                           int64_t,
//                           float,
//                           double,
//                           phi::dtype::float16) {}
