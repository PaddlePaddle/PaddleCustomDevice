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
#include "paddle/phi/common/type_traits.h"

namespace custom_kernel {

template <typename T, typename Context>
void ActivationBaseKernel(const Context& dev_ctx,
                          const phi::DenseTensor& x,
                          const GcuAttributeMap& attrs,
                          phi::DenseTensor* out,
                          const std::string& op_type) {
  dev_ctx.template Alloc<T>(out);

  TensorNameMap input_names;
  input_names["X"] = {"x"};

  TensorValueMap inputs;
  inputs["X"] = {const_cast<DenseTensor*>(&x)};

  TensorNameMap output_names;
  output_names["Out"] = {"out"};

  TensorValueMap outputs;
  outputs["Out"] = {out};

  GcuRunner(
      input_names, inputs, output_names, outputs, attrs, op_type, dev_ctx);
}

template <typename T, typename Context>
void ActivationGradBaseKernel(const Context& dev_ctx,
                              const std::string& x_name,
                              const phi::DenseTensor& x,
                              const phi::DenseTensor& dout,
                              const GcuAttributeMap& attrs,
                              phi::DenseTensor* dx,
                              const std::string& op_type) {
  dev_ctx.template Alloc<T>(dx);

  TensorNameMap input_names;
  input_names[x_name] = {"x"};
  input_names[GradVarName("Out")] = {"dout"};

  TensorValueMap inputs;
  inputs[x_name] = {const_cast<DenseTensor*>(&x)};
  inputs[GradVarName("Out")] = {const_cast<DenseTensor*>(&dout)};

  TensorNameMap output_names;
  output_names[GradVarName("X")] = {"dx"};

  TensorValueMap outputs;
  outputs[GradVarName("X")] = {dx};

  GcuRunner(
      input_names, inputs, output_names, outputs, attrs, op_type, dev_ctx);
}

template <typename T, typename Context>
void AbsKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("abs");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    ActivationBaseKernel<T, Context>(dev_ctx, x, {}, out, "abs");
  }
}

template <typename T, typename Context>
void AbsGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& dout,
                   phi::DenseTensor* dx) {
  PADDLE_GCU_KERNEL_TRACE("abs_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    ActivationGradBaseKernel<T, Context>(
        dev_ctx, "X", x, dout, {}, dx, "abs_grad");
  }
}

template <typename T, typename Context>
void CosKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("cos");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    LAUNCH_TOPSATENOP(topsatenCos, dev_ctx, *out, x);

  } else {  // kernel impl base on JIT
    ActivationBaseKernel<T, Context>(dev_ctx, x, {}, out, "cos");
  }
}

template <typename T, typename Context>
void CosGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& dout,
                   phi::DenseTensor* dx) {
  PADDLE_GCU_KERNEL_TRACE("cos_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    ActivationGradBaseKernel<T, Context>(
        dev_ctx, "X", x, dout, {}, dx, "cos_grad");
  }
}

template <typename T, typename Context>
void SinKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("sin");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    LAUNCH_TOPSATENOP(topsatenSin, dev_ctx, *out, x);

  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}

template <typename T, typename Context>
void AtanKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("atan");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    ActivationBaseKernel<T, Context>(dev_ctx, x, {}, out, "atan");
  }
}

template <typename T, typename Context>
void AtanGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& dout,
                    phi::DenseTensor* dx) {
  PADDLE_GCU_KERNEL_TRACE("atan_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    ActivationGradBaseKernel<T, Context>(
        dev_ctx, "X", x, dout, {}, dx, "atan_grad");
  }
}

template <typename T, typename Context>
void ExpKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("exp");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    ActivationBaseKernel<T, Context>(dev_ctx, x, {}, out, "exp");
  }
}

template <typename T, typename Context>
void ExpGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& out,
                   const phi::DenseTensor& dout,
                   phi::DenseTensor* dx) {
  PADDLE_GCU_KERNEL_TRACE("exp_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    ActivationGradBaseKernel<T, Context>(
        dev_ctx, "Out", out, dout, {}, dx, "exp_grad");
  }
}

template <typename T, typename Context>
void FloorKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("floor");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    ActivationBaseKernel<T, Context>(dev_ctx, x, {}, out, "floor");
  }
}

template <typename T, typename Context>
void FloorGradKernel(const Context& dev_ctx,
                     const phi::DenseTensor& dout,
                     phi::DenseTensor* dx) {
  PADDLE_GCU_KERNEL_TRACE("floor_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    dev_ctx.template Alloc<T>(dx);

    TensorNameMap input_names;
    input_names[GradVarName("Out")] = {"dout"};

    TensorValueMap inputs;
    inputs[GradVarName("Out")] = {const_cast<DenseTensor*>(&dout)};

    TensorNameMap output_names;
    output_names[GradVarName("X")] = {"dx"};

    TensorValueMap outputs;
    outputs[GradVarName("X")] = {dx};

    GcuRunner(
        input_names, inputs, output_names, outputs, {}, "floor_grad", dev_ctx);
  }
}

template <typename T, typename Context>
void SwishKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("swish");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    LAUNCH_TOPSATENOP(topsatenSilu, dev_ctx, *out, x);
  } else {  // kernel impl base on JIT
    ActivationBaseKernel<T, Context>(dev_ctx, x, {}, out, "swish");
  }
}

template <typename T, typename Context>
void SwishGradKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& dout,
                     phi::DenseTensor* dx) {
  PADDLE_GCU_KERNEL_TRACE("swish_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    float beta = 1.0;
    GcuAttributeMap attrs;
    attrs["beta"] = beta;
    ActivationGradBaseKernel<T, Context>(
        dev_ctx, "X", x, dout, attrs, dx, "swish_grad");
  }
}

template <typename T, typename Context>
void ReluKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("relu");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    LAUNCH_TOPSATENOP(topsatenRelu, dev_ctx, *out, x);
  } else {  // kernel impl base on JIT
    ActivationBaseKernel<T, Context>(dev_ctx, x, {}, out, "relu");
  }
}

template <typename T, typename Context>
void ReluGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& out,
                    const phi::DenseTensor& dout,
                    phi::DenseTensor* dx) {
  PADDLE_GCU_KERNEL_TRACE("relu_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    ActivationGradBaseKernel<T, Context>(
        dev_ctx, "Out", out, dout, {}, dx, "relu_grad");
  }
}

template <typename T, typename Context>
void Relu6Kernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("relu6");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    ActivationBaseKernel<T, Context>(dev_ctx, x, {}, out, "relu6");
  }
}

template <typename T, typename Context>
void Relu6GradKernel(const Context& dev_ctx,
                     const phi::DenseTensor& out,
                     const phi::DenseTensor& dout,
                     phi::DenseTensor* dx) {
  PADDLE_GCU_KERNEL_TRACE("relu6_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    ActivationGradBaseKernel<T, Context>(
        dev_ctx, "Out", out, dout, {}, dx, "relu6_grad");
  }
}

template <typename T, typename Context>
void LeakyReluKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     float alpha,
                     phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("leaky_relu");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    GcuAttributeMap attrs;
    attrs["alpha"] = alpha;

    ActivationBaseKernel<T, Context>(dev_ctx, x, attrs, out, "leaky_relu");
  }
}

template <typename T, typename Context>
void LeakyReluGradKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const phi::DenseTensor& dout,
                         float alpha,
                         phi::DenseTensor* dx) {
  PADDLE_GCU_KERNEL_TRACE("leaky_relu_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    GcuAttributeMap attrs;
    attrs["alpha"] = alpha;

    ActivationGradBaseKernel<T, Context>(
        dev_ctx, "X", x, dout, attrs, dx, "leaky_relu_grad");
  }
}

template <typename T, typename Context>
void GeluKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                bool approximate,
                phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("gelu");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    GcuAttributeMap attrs;
    attrs["approximate"] = approximate;
    attrs["use_mkldnn"] = false;
    attrs["use_cudnn"] = false;

    ActivationBaseKernel<T, Context>(dev_ctx, x, attrs, out, "gelu");
  }
}

template <typename T, typename Context>
void GeluGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& dout,
                    bool approximate,
                    phi::DenseTensor* dx) {
  PADDLE_GCU_KERNEL_TRACE("gelu_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    GcuAttributeMap attrs;
    attrs["approximate"] = approximate;
    attrs["use_mkldnn"] = false;
    attrs["use_cudnn"] = false;

    ActivationGradBaseKernel<T, Context>(
        dev_ctx, "X", x, dout, attrs, dx, "gelu_grad");
  }
}

template <typename T, typename Context>
void TanhKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("tanh");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    ActivationBaseKernel<T, Context>(dev_ctx, x, {}, out, "tanh");
  }
}

template <typename T, typename Context>
void TanhGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& out,
                    const phi::DenseTensor& dout,
                    phi::DenseTensor* dx) {
  PADDLE_GCU_KERNEL_TRACE("tanh_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    ActivationGradBaseKernel<T, Context>(
        dev_ctx, "Out", out, dout, {}, dx, "tanh_grad");
  }
}

template <typename T, typename Context>
void SigmoidKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("sigmoid");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    LAUNCH_TOPSATENOP(topsatenSigmoid, dev_ctx, *out, x);
  } else {  // kernel impl base on JIT
    ActivationBaseKernel<T, Context>(dev_ctx, x, {}, out, "sigmoid");
  }
}

template <typename T, typename Context>
void SigmoidGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& out,
                       const phi::DenseTensor& dout,
                       phi::DenseTensor* dx) {
  PADDLE_GCU_KERNEL_TRACE("sigmoid_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    ActivationGradBaseKernel<T, Context>(
        dev_ctx, "Out", out, dout, {}, dx, "sigmoid_grad");
  }
}

template <typename T, typename Context>
void SqrtKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("sqrt");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    ActivationBaseKernel<T, Context>(dev_ctx, x, {}, out, "sqrt");
  }
}

template <typename T, typename Context>
void RsqrtKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("rsqrt");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    LAUNCH_TOPSATENOP(topsatenRsqrt, dev_ctx, *out, x);

  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}

template <typename T, typename Context>
void LogKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("log");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    LAUNCH_TOPSATENOP(topsatenLog, dev_ctx, *out, x);

  } else {  // kernel impl base on JIT
    ActivationBaseKernel<T, Context>(dev_ctx, x, {}, out, "log");
  }
}

template <typename T, typename Context>
void LogGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& dout,
                   phi::DenseTensor* dx) {
  PADDLE_GCU_KERNEL_TRACE("log_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    ActivationGradBaseKernel<T, Context>(
        dev_ctx, "X", x, dout, {}, dx, "log_grad");
  }
}

template <typename T, typename Context>
void PowKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::Scalar& factor_scalar,
               phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("pow");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    LAUNCH_TOPSATENOP(topsatenPow, dev_ctx, *out, x, factor_scalar);

  } else {  // kernel impl base on JIT
    GcuAttributeMap attrs;
    attrs["factor"] = factor_scalar.to<float>();
    ActivationBaseKernel<T, Context>(dev_ctx, x, attrs, out, "pow");
  }
}

template <typename T, typename Context>
void PowGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& dout,
                   const phi::Scalar& factor_scalar,
                   phi::DenseTensor* dx) {
  PADDLE_GCU_KERNEL_TRACE("pow_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    GcuAttributeMap attrs;
    attrs["factor"] = factor_scalar.to<float>();
    ActivationGradBaseKernel<T, Context>(
        dev_ctx, "X", x, dout, attrs, dx, "pow_grad");
  }
}

template <typename T, typename Context>
void SquareKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("square");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    ActivationBaseKernel<T, Context>(dev_ctx, x, {}, out, "square");
  }
}

template <typename T, typename Context>
void SquareGradKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& dout,
                      phi::DenseTensor* dx) {
  PADDLE_GCU_KERNEL_TRACE("square_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    ActivationGradBaseKernel<T, Context>(
        dev_ctx, "X", x, dout, {}, dx, "square_grad");
  }
}

template <typename T, typename Context>
void Hard_SigmoidKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        float slope,
                        float offset,
                        phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("hard_sigmoid");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    LAUNCH_TOPSATENOP(topsatenHardsigmoid, dev_ctx, *out, x);
  } else {  // kernel impl base on JIT
    GcuAttributeMap attrs;
    attrs["slope"] = slope;
    attrs["offset"] = offset;

    ActivationBaseKernel<T, Context>(dev_ctx, x, attrs, out, "hard_sigmoid");
  }
}

template <typename T, typename Context>
void HardSigmoidKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       float slope,
                       float offset,
                       phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("hardsigmoid");
  Hard_SigmoidKernel<T, Context>(dev_ctx, x, slope, offset, out);
}

template <typename T, typename Context>
void HardSigmoidGradKernel(const Context& dev_ctx,
                           const phi::DenseTensor& out,
                           const phi::DenseTensor& dout,
                           float slope,
                           float offset,
                           phi::DenseTensor* dx) {
  PADDLE_GCU_KERNEL_TRACE("hard_sigmoid_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    GcuAttributeMap attrs;
    attrs["slope"] = slope;
    attrs["offset"] = offset;

    ActivationGradBaseKernel<T, Context>(
        dev_ctx, "Out", out, dout, attrs, dx, "hard_sigmoid_grad");
  }
}

template <typename T, typename Context>
void HardSwishKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("hard_swish");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    LAUNCH_TOPSATENOP(topsatenHardswish, dev_ctx, *out, x);
  } else {  // kernel impl base on JIT
    ActivationBaseKernel<T, Context>(dev_ctx, x, {}, out, "hard_swish");
  }
}

template <typename T, typename Context>
void HardSwishGradKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const phi::DenseTensor& dout,
                         phi::DenseTensor* dx) {
  PADDLE_GCU_KERNEL_TRACE("hard_swish_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    float offset = 3;
    float threshold = 6;
    float scale = 6;
    GcuAttributeMap attrs;
    attrs["offset"] = offset;
    attrs["threshold"] = threshold;
    attrs["scale"] = scale;
    ActivationGradBaseKernel<T, Context>(
        dev_ctx, "X", x, dout, attrs, dx, "hard_swish_grad");
  }
}

// Silu = x * sigmoid(x)
template <typename T, typename Context>
void SiluKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("silu");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    LAUNCH_TOPSATENOP(topsatenSilu, dev_ctx, *out, x);

  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(abs,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::AbsKernel,
                          float,
                          double,
                          int64_t,
                          int32_t) {
  kernel->InputAt(0).SetDataType(phi::dtype::ToReal(kernel_key.dtype()));
}

PD_REGISTER_PLUGIN_KERNEL(abs_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::AbsGradKernel,
                          float,
                          double,
                          int64_t,
                          int32_t) {
  kernel->InputAt(1).SetDataType(phi::dtype::ToReal(kernel_key.dtype()));
}

PD_REGISTER_PLUGIN_KERNEL(cos,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::CosKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(cos_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::CosGradKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(sin,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::SinKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(atan,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::AtanKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(atan_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::AtanGradKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(exp,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::ExpKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(exp_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::ExpGradKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(swish,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::SwishKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(swish_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::SwishGradKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(relu,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::ReluKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(relu_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::ReluGradKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(relu6,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::Relu6Kernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(relu6_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::Relu6GradKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(leaky_relu,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::LeakyReluKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(leaky_relu_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::LeakyReluGradKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(pow,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::PowKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(pow_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::PowGradKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(log,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::LogKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(floor,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::FloorKernel,
                          double,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(floor_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::FloorGradKernel,
                          double,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(gelu,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::GeluKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(gelu_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::GeluGradKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(tanh,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::TanhKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(tanh_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::TanhGradKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(sigmoid,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::SigmoidKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(sigmoid_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::SigmoidGradKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(sqrt,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::SqrtKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(rsqrt,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::RsqrtKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(square,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::SquareKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(square_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::SquareGradKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(hard_sigmoid,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::Hard_SigmoidKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(hardsigmoid,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::HardSigmoidKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(hard_sigmoid_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::HardSigmoidGradKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(hardswish,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::HardSwishKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(hardswish_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::HardSwishGradKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(silu,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::SiluKernel,
                          float,
                          phi::dtype::float16) {}
