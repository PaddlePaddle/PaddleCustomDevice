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

#include "kernels/common_ops/unary_ops.h"
#include "kernels/funcs/gcu_kernel_funcs.h"
#include "kernels/funcs/gcu_op_runner.h"
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
  ActivationBaseKernel<T, Context>(dev_ctx, x, {}, out, "abs");
}

template <typename T, typename Context>
void AbsGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& dout,
                   phi::DenseTensor* dx) {
  ActivationGradBaseKernel<T, Context>(
      dev_ctx, "X", x, dout, {}, dx, "abs_grad");
}

template <typename T, typename Context>
void CosKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  ActivationBaseKernel<T, Context>(dev_ctx, x, {}, out, "cos");
}

template <typename T, typename Context>
void CosGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& dout,
                   phi::DenseTensor* dx) {
  ActivationGradBaseKernel<T, Context>(
      dev_ctx, "X", x, dout, {}, dx, "cos_grad");
}

template <typename T, typename Context>
void AtanKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* out) {
  ActivationBaseKernel<T, Context>(dev_ctx, x, {}, out, "atan");
}

template <typename T, typename Context>
void AtanGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& dout,
                    phi::DenseTensor* dx) {
  ActivationGradBaseKernel<T, Context>(
      dev_ctx, "X", x, dout, {}, dx, "atan_grad");
}

template <typename T, typename Context>
void ExpKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "exp", exp);
    dev_ctx.template Alloc<T>(out);
    exp_compute(static_cast<const phi::CustomContext&>(dev_ctx), x, out);
    PADDLE_GCU_KERNEL_END("exp", exp);
  } else {
    ActivationBaseKernel<T, Context>(dev_ctx, x, {}, out, "exp");
  }
}

template <typename T, typename Context>
void ExpGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& out,
                   const phi::DenseTensor& dout,
                   phi::DenseTensor* dx) {
  ActivationGradBaseKernel<T, Context>(
      dev_ctx, "Out", out, dout, {}, dx, "exp_grad");
}

template <typename T, typename Context>
void FloorKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 phi::DenseTensor* out) {
  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "floor", floor);
    dev_ctx.template Alloc<T>(out);
    floor_compute(static_cast<const phi::CustomContext&>(dev_ctx), x, out);
    PADDLE_GCU_KERNEL_END("floor", floor);
  } else {
    ActivationBaseKernel<T, Context>(dev_ctx, x, {}, out, "floor");
  }
}

template <typename T, typename Context>
void FloorGradKernel(const Context& dev_ctx,
                     const phi::DenseTensor& dout,
                     phi::DenseTensor* dx) {
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

// template <typename T, typename Context>
// void RsqrtKernel(const Context& dev_ctx,
//                  const phi::DenseTensor& x,
//                  phi::DenseTensor* out) {
// }

// template <typename T, typename Context>
// void RsqrtGradKernel(const Context& dev_ctx,
//                      const phi::DenseTensor& out,
//                      const phi::DenseTensor& dout,
//                      phi::DenseTensor* dx) {
// }

// template <typename T, typename Context>
// void SinKernel(const Context& dev_ctx,
//                const phi::DenseTensor& x,
//                phi::DenseTensor* out) {}

// // Swish = x * sigmoid(beta * x)
// template <typename T, typename Context>
// void SwishRawKernel(const Context& dev_ctx,
//                     const phi::DenseTensor& x,
//                     float beta,
//                     phi::DenseTensor* out) {}

template <typename T, typename Context>
void SwishKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 phi::DenseTensor* out) {
  ActivationBaseKernel<T, Context>(dev_ctx, x, {}, out, "swish");
}

template <typename T, typename Context>
void SwishGradKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& dout,
                     phi::DenseTensor* dx) {
  float beta = 1.0;
  GcuAttributeMap attrs;
  attrs["beta"] = beta;
  ActivationGradBaseKernel<T, Context>(
      dev_ctx, "X", x, dout, attrs, dx, "swish_grad");
}

template <typename T, typename Context>
void ReluKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* out) {
  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "relu", relu);
    dev_ctx.template Alloc<T>(out);
    relu_compute(static_cast<const phi::CustomContext&>(dev_ctx), x, out);
    PADDLE_GCU_KERNEL_END("relu", relu);
  } else {
    ActivationBaseKernel<T, Context>(dev_ctx, x, {}, out, "relu");
  }
}

template <typename T, typename Context>
void ReluGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& out,
                    const phi::DenseTensor& dout,
                    phi::DenseTensor* dx) {
  if (UseScatterMemory()) {
    dev_ctx.template Alloc<T>(dx);

    PADDLE_GCU_KERNEL_START(dev_ctx, "relu_grad", relu_grad);
    auto out_gcu = GetHlirTensor(out);
    auto dout_gcu = GetHlirTensor(dout);
    auto dx_gcu = GetHlirTensor(*dx);
    hlir::DispatchParam params;
    params.inputs = {dout_gcu, out_gcu};
    params.outputs = {dx_gcu};
    params.stream = static_cast<topsStream_t>(dev_ctx.stream());
    AOTOPS_DEBUG(kReluGrad, params);
    GCUOPS_TRACE_START(relu_grad);
    auto func_ptr = GetOpFuncPtr(kReluGrad, params);
    if (func_ptr) {
      auto pass = hlir::HlirDispatch::dispatch(func_ptr, params);
      PADDLE_ENFORCE(
          pass, phi::errors::InvalidArgument("dispatch %s failed!", kReluGrad));
    } else {
      PADDLE_ENFORCE(
          false,
          phi::errors::InvalidArgument("not find aot func for %s", kReluGrad));
    }
    FreeDispatchParam(params);
    GCUOPS_TRACE_END(relu_grad);
    GcuOpStreamSync(dev_ctx);

    PADDLE_GCU_KERNEL_END("relu_grad", relu_grad);
  } else {
    ActivationGradBaseKernel<T, Context>(
        dev_ctx, "Out", out, dout, {}, dx, "relu_grad");
  }
}

// template <typename T, typename Context>
// void Relu6RawKernel(const Context& dev_ctx,
//                     const phi::DenseTensor& x,
//                     float threshold,
//                     phi::DenseTensor* out) {
// }

template <typename T, typename Context>
void Relu6Kernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 phi::DenseTensor* out) {
  ActivationBaseKernel<T, Context>(dev_ctx, x, {}, out, "relu6");
}

template <typename T, typename Context>
void Relu6GradKernel(const Context& dev_ctx,
                     const phi::DenseTensor& out,
                     const phi::DenseTensor& dout,
                     phi::DenseTensor* dx) {
  ActivationGradBaseKernel<T, Context>(
      dev_ctx, "Out", out, dout, {}, dx, "relu6_grad");
}

template <typename T, typename Context>
void LeakyReluKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     float alpha,
                     phi::DenseTensor* out) {
  GcuAttributeMap attrs;
  attrs["alpha"] = alpha;

  ActivationBaseKernel<T, Context>(dev_ctx, x, attrs, out, "leaky_relu");
}

template <typename T, typename Context>
void LeakyReluGradKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const phi::DenseTensor& dout,
                         float alpha,
                         phi::DenseTensor* dx) {
  GcuAttributeMap attrs;
  attrs["alpha"] = alpha;

  ActivationGradBaseKernel<T, Context>(
      dev_ctx, "X", x, dout, attrs, dx, "leaky_relu_grad");
}

template <typename T, typename Context>
void GeluKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                bool approximate,
                phi::DenseTensor* out) {
  GcuAttributeMap attrs;
  attrs["approximate"] = approximate;
  attrs["use_mkldnn"] = false;
  attrs["use_cudnn"] = false;

  ActivationBaseKernel<T, Context>(dev_ctx, x, attrs, out, "gelu");
}

template <typename T, typename Context>
void GeluGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& dout,
                    bool approximate,
                    phi::DenseTensor* dx) {
  GcuAttributeMap attrs;
  attrs["approximate"] = approximate;
  attrs["use_mkldnn"] = false;
  attrs["use_cudnn"] = false;

  ActivationGradBaseKernel<T, Context>(
      dev_ctx, "X", x, dout, attrs, dx, "gelu_grad");
}

template <typename T, typename Context>
void TanhKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* out) {
  ActivationBaseKernel<T, Context>(dev_ctx, x, {}, out, "tanh");
}

template <typename T, typename Context>
void TanhGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& out,
                    const phi::DenseTensor& dout,
                    phi::DenseTensor* dx) {
  ActivationGradBaseKernel<T, Context>(
      dev_ctx, "Out", out, dout, {}, dx, "tanh_grad");
}

template <typename T, typename Context>
void SigmoidKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   phi::DenseTensor* out) {
  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "sigmoid", sigmoid);
    dev_ctx.template Alloc<T>(out);
    sigmoid_compute(static_cast<const phi::CustomContext&>(dev_ctx), x, out);
    PADDLE_GCU_KERNEL_END("sigmoid", sigmoid);
  } else {
    ActivationBaseKernel<T, Context>(dev_ctx, x, {}, out, "sigmoid");
  }
}

template <typename T, typename Context>
void SigmoidGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& out,
                       const phi::DenseTensor& dout,
                       phi::DenseTensor* dx) {
  ActivationGradBaseKernel<T, Context>(
      dev_ctx, "Out", out, dout, {}, dx, "sigmoid_grad");
}

// template <typename T, typename Context>
// void EluKernel(const Context& dev_ctx,
//                const phi::DenseTensor& x,
//                float alpha,
//                phi::DenseTensor* out) {
// }

// template <typename T, typename Context>
// void EluGradKernel(const Context& dev_ctx,
//                    const phi::DenseTensor& x,  // output
//                    const phi::DenseTensor& out,
//                    const phi::DenseTensor& dout,
//                    float alpha,
//                    phi::DenseTensor* dx) {
// }

// template <typename T, typename Context>
// void CeluKernel(const Context& dev_ctx,
//                 const phi::DenseTensor& x,
//                 float alpha,
//                 phi::DenseTensor* out) {
// }

// template <typename T, typename Context>
// void CeluGradKernel(const Context& dev_ctx,
//                     const phi::DenseTensor& x,
//                     const phi::DenseTensor& dout,
//                     float alpha,
//                     phi::DenseTensor* dx) {
// }

// template <typename T, typename Context>
// void SeluKernel(const Context& dev_ctx,
//                 const phi::DenseTensor& x,
//                 float scale,
//                 float alpha,
//                 phi::DenseTensor* out) {

// template <typename T, typename Context>
// void SeluGradKernel(const Context& dev_ctx,
//                     const phi::DenseTensor& out,
//                     const phi::DenseTensor& dout,
//                     float scale,
//                     float alpha,
//                     phi::DenseTensor* dx) {
// }

template <typename T, typename Context>
void SqrtKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* out) {
  ActivationBaseKernel<T, Context>(dev_ctx, x, {}, out, "sqrt");
}

// template <typename T, typename Context>
// void SqrtGradKernel(const Context& dev_ctx,
//                     const phi::DenseTensor& out,
//                     const phi::DenseTensor& dout,
//                     phi::DenseTensor* dx) {
// }

template <typename T, typename Context>
void LogKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "log", log);
    dev_ctx.template Alloc<T>(out);
    log_compute(static_cast<const phi::CustomContext&>(dev_ctx), x, out);
    PADDLE_GCU_KERNEL_END("log", log);
  } else {
    ActivationBaseKernel<T, Context>(dev_ctx, x, {}, out, "log");
  }
}

template <typename T, typename Context>
void LogGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& dout,
                   phi::DenseTensor* dx) {
  ActivationGradBaseKernel<T, Context>(
      dev_ctx, "X", x, dout, {}, dx, "log_grad");
}

template <typename T, typename Context>
void PowKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::Scalar& factor_scalar,
               phi::DenseTensor* out) {
  GcuAttributeMap attrs;
  attrs["factor"] = factor_scalar.to<float>();
  ActivationBaseKernel<T, Context>(dev_ctx, x, attrs, out, "pow");
}

template <typename T, typename Context>
void PowGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& dout,
                   const phi::Scalar& factor_scalar,
                   phi::DenseTensor* dx) {
  GcuAttributeMap attrs;
  attrs["factor"] = factor_scalar.to<float>();
  ActivationGradBaseKernel<T, Context>(
      dev_ctx, "X", x, dout, attrs, dx, "pow_grad");
}

template <typename T, typename Context>
void SquareKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  phi::DenseTensor* out) {
  ActivationBaseKernel<T, Context>(dev_ctx, x, {}, out, "square");
}

template <typename T, typename Context>
void SquareGradKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& dout,
                      phi::DenseTensor* dx) {
  ActivationGradBaseKernel<T, Context>(
      dev_ctx, "X", x, dout, {}, dx, "square_grad");
}

template <typename T, typename Context>
void HardSigmoidKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       float slope,
                       float offset,
                       phi::DenseTensor* out) {
  GcuAttributeMap attrs;
  attrs["slope"] = slope;
  attrs["offset"] = offset;

  ActivationBaseKernel<T, Context>(dev_ctx, x, attrs, out, "hard_sigmoid");
}

template <typename T, typename Context>
void HardSigmoidGradKernel(const Context& dev_ctx,
                           const phi::DenseTensor& out,
                           const phi::DenseTensor& dout,
                           float slope,
                           float offset,
                           phi::DenseTensor* dx) {
  GcuAttributeMap attrs;
  attrs["slope"] = slope;
  attrs["offset"] = offset;

  ActivationGradBaseKernel<T, Context>(
      dev_ctx, "Out", out, dout, attrs, dx, "hard_sigmoid_grad");
}

template <typename T, typename Context>
void HardSwishKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     phi::DenseTensor* out) {
  ActivationBaseKernel<T, Context>(dev_ctx, x, {}, out, "hard_swish");
}

template <typename T, typename Context>
void HardSwishGradKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const phi::DenseTensor& dout,
                         phi::DenseTensor* dx) {
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

// template <typename T, typename Context>
// void SoftplusKernel(const Context& dev_ctx,
//                     const phi::DenseTensor& x,
//                     const float beta,
//                     const float threshold,
//                     phi::DenseTensor* out) {
// }

// template <typename T, typename Context>
// void SoftplusGradKernel(const Context& dev_ctx,
//                         const phi::DenseTensor& a,
//                         const phi::DenseTensor& dout,
//                         const float beta,
//                         const float threshold,
//                         phi::DenseTensor* dx) {
// }

// template <typename T, typename Context>
// void SoftshrinkKernel(const Context& dev_ctx,
//                       const phi::DenseTensor& x,
//                       const float lambd,
//                       phi::DenseTensor* out) {
// }

// template <typename T, typename Context>
// void SoftshrinkGradKernel(const Context& dev_ctx,
//                           const phi::DenseTensor& a,
//                           const phi::DenseTensor& dout,
//                           const float lambd,
//                           phi::DenseTensor* dx) {
// }

// template <typename T, typename Context>
// void HardshrinkKernel(const Context& dev_ctx,
//                       const phi::DenseTensor& x,
//                       const float lambd,
//                       phi::DenseTensor* out) {
// }

// template <typename T, typename Context>
// void HardshrinkGradKernel(const Context& dev_ctx,
//                           const phi::DenseTensor& a,
//                           const phi::DenseTensor& dout,
//                           const float lambd,
//                           phi::DenseTensor* dx) {
// }

// template <typename T, typename Context>
// void ReciprocalKernel(const Context& dev_ctx,
//                       const phi::DenseTensor& x,
//                       phi::DenseTensor* out) {

// }

// template <typename T, typename Context>
// void ReciprocalGradKernel(const Context& dev_ctx,
//                           const phi::DenseTensor& out,
//                           const phi::DenseTensor& dout,
//                           phi::DenseTensor* dx) {

// }
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(cos,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::CosKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(cos_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::CosGradKernel,
                          float,
                          double,
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

// PD_REGISTER_PLUGIN_KERNEL(sin,
//                           gcu,
//                           ALL_LAYOUT,
//                           custom_kernel::SinKernel,
//                           float,
//                           double,
//                           phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(swish,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::SwishKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

// PD_REGISTER_PLUGIN_KERNEL(swish_raw,
//                           gcu,
//                           ALL_LAYOUT,
//                           custom_kernel::SwishRawKernel,
//                           float,
//                           phi::dtype::float16) {}

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

// PD_REGISTER_PLUGIN_KERNEL(relu6_raw,
//                           gcu,
//                           ALL_LAYOUT,
//                           custom_kernel::Relu6RawKernel,
//                           float,
//                           double,
//                           phi::dtype::float16) {}

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
                          double,
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
                          double,
                          phi::dtype::float16) {}

// PD_REGISTER_PLUGIN_KERNEL(log_grad,
//                           gcu,
//                           ALL_LAYOUT,
//                           custom_kernel::LogGradKernel,
//                           float,
//                           phi::dtype::float16) {}

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

// PD_REGISTER_PLUGIN_KERNEL(sqrt_grad,
//                           gcu,
//                           ALL_LAYOUT,
//                           custom_kernel::SqrtGradKernel,
//                           float,
//                           phi::dtype::float16,
//                           double) {}

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

// PD_REGISTER_PLUGIN_KERNEL(softplus,
//                           gcu,
//                           ALL_LAYOUT,
//                           custom_kernel::SoftplusKernel,
//                           float,
//                           phi::dtype::float16) {}

// PD_REGISTER_PLUGIN_KERNEL(softplus_grad,
//                           gcu,
//                           ALL_LAYOUT,
//                           custom_kernel::SoftplusGradKernel,
//                           float,
//                           phi::dtype::float16) {}

// PD_REGISTER_PLUGIN_KERNEL(softshrink,
//                           gcu,
//                           ALL_LAYOUT,
//                           custom_kernel::SoftshrinkKernel,
//                           float,
//                           phi::dtype::float16) {}

// PD_REGISTER_PLUGIN_KERNEL(softshrink_grad,
//                           gcu,
//                           ALL_LAYOUT,
//                           custom_kernel::SoftshrinkGradKernel,
//                           float,
//                           phi::dtype::float16) {}

// PD_REGISTER_PLUGIN_KERNEL(hard_shrink,
//                           gcu,
//                           ALL_LAYOUT,
//                           custom_kernel::HardshrinkKernel,
//                           float,
//                           phi::dtype::float16) {}

// PD_REGISTER_PLUGIN_KERNEL(hard_shrink_grad,
//                           gcu,
//                           ALL_LAYOUT,
//                           custom_kernel::HardshrinkGradKernel,
//                           float,
//                           phi::dtype::float16) {}

// PD_REGISTER_PLUGIN_KERNEL(reciprocal,
//                           gcu,
//                           ALL_LAYOUT,
//                           custom_kernel::ReciprocalKernel,
//                           float,
//                           double,
//                           phi::dtype::float16) {}

// PD_REGISTER_PLUGIN_KERNEL(reciprocal_grad,
//                           gcu,
//                           ALL_LAYOUT,
//                           custom_kernel::ReciprocalGradKernel,
//                           float,
//                           double,
//                           phi::dtype::float16) {}

// PD_REGISTER_PLUGIN_KERNEL(selu,
//                           gcu,
//                           ALL_LAYOUT,
//                           custom_kernel::SeluKernel,
//                           float,
//                           phi::dtype::float16) {}

// PD_REGISTER_PLUGIN_KERNEL(selu_grad,
//                           gcu,
//                           ALL_LAYOUT,
//                           custom_kernel::SeluGradKernel,
//                           float,
//                           phi::dtype::float16) {}

// PD_REGISTER_PLUGIN_KERNEL(
//     rsqrt, gcu, ALL_LAYOUT, custom_kernel::RsqrtKernel, float, double) {}

// PD_REGISTER_PLUGIN_KERNEL(rsqrt_grad,
//                           gcu,
//                           ALL_LAYOUT,
//                           custom_kernel::RsqrtGradKernel,
//                           float,
//                           double) {}

// PD_REGISTER_PLUGIN_KERNEL(elu,
//                           gcu,
//                           ALL_LAYOUT,
//                           custom_kernel::EluKernel,
//                           float,
//                           phi::dtype::float16) {}

// PD_REGISTER_PLUGIN_KERNEL(elu_grad,
//                           gcu,
//                           ALL_LAYOUT,
//                           custom_kernel::EluGradKernel,
//                           float,
//                           phi::dtype::float16) {}

// PD_REGISTER_PLUGIN_KERNEL(celu,
//                           gcu,
//                           ALL_LAYOUT,
//                           custom_kernel::CeluKernel,
//                           float,
//                           phi::dtype::float16) {}

// PD_REGISTER_PLUGIN_KERNEL(celu_grad,
//                           gcu,
//                           ALL_LAYOUT,
//                           custom_kernel::CeluGradKernel,
//                           float,
//                           phi::dtype::float16) {}
