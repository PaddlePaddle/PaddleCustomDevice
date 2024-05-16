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

template <typename T, typename Context>
void SoftmaxKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   int axis,
                   phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("softmax");
  const int rank = x.dims().size();
  if (rank == 0) {
    dev_ctx.template Alloc<T>(out);
    auto out_dim = out->dims();
    FillGcuTensorWithConstant<T>(out, dev_ctx, static_cast<T>(1));
    out->Resize(out_dim);
    // dev_ctx.Wait();
    return;
  }

  dev_ctx.template Alloc<T>(out);

  if (LaunchAOTKernel()) {
    if (axis < 0) {
      axis += x.dims().size();
    }
    LAUNCH_TOPSATENOP(topsatenSoftmaxForward, dev_ctx, *out, x, axis);

  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    input_names["X"] = {"x"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};

    TensorValueMap outputs;
    outputs["Out"] = {out};

    GcuAttributeMap attrs;
    attrs["axis"] = axis;

    GcuRunner(
        input_names, inputs, output_names, outputs, attrs, "softmax", dev_ctx);
  }
}

template <typename T, typename Context>
void SoftmaxGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& out,
                       const phi::DenseTensor& out_grad,
                       int axis,
                       phi::DenseTensor* x_grad) {
  PADDLE_GCU_KERNEL_TRACE("softmax_grad");
  auto dims = x_grad->dims();
  const int rank = dims.size();
  if (out.dims().size() == 0) {
    dev_ctx.template Alloc<T>(x_grad);
    auto x_grad_dim = x_grad->dims();
    FillGcuTensorWithConstant<T>(x_grad, dev_ctx, static_cast<T>(0));
    x_grad->Resize(x_grad_dim);
    return;
  }

  dev_ctx.template Alloc<T>(x_grad);
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    input_names["Out"] = {"out"};
    input_names[GradVarName("Out")] = {"out_grad"};

    TensorValueMap inputs;
    inputs["Out"] = {const_cast<DenseTensor*>(&out)};
    inputs[GradVarName("Out")] = {const_cast<DenseTensor*>(&out_grad)};

    TensorNameMap output_names;
    TensorValueMap outputs;

    output_names[GradVarName("X")] = {"x_grad"};
    outputs[GradVarName("X")] = {x_grad};

    GcuAttributeMap attrs;
    attrs["axis"] = axis;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "softmax_grad",
              dev_ctx);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(softmax,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::SoftmaxKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(softmax_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::SoftmaxGradKernel,
                          float,
                          double,
                          phi::dtype::float16) {}
