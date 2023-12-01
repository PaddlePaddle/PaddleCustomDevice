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
#include "kernels/common_ops/elementwise_ops.h"
#include "kernels/funcs/gcu_kernel_funcs.h"
#include "kernels/funcs/gcu_name_list.h"
#include "kernels/funcs/gcu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void SoftmaxKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   int axis,
                   phi::DenseTensor* out) {
  const int rank = x.dims().size();
  if (rank == 0) {
    dev_ctx.template Alloc<T>(out);
    auto out_dim = out->dims();
    FillGcuTensorWithConstant<T>(out, dev_ctx, static_cast<T>(1));
    out->Resize(out_dim);
    dev_ctx.Wait();
    return;
  }

  dev_ctx.template Alloc<T>(out);

  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "softmax", softmax);
    if (out->numel() > 0) {
      auto x_gcu = GetHlirTensor(x);
      auto out_gcu = GetHlirTensor(*out);
      hlir::DispatchParam params;
      params.inputs = {x_gcu};
      params.outputs = {out_gcu};
      params.metadata.setValue("axis", axis);
      params.stream = static_cast<topsStream_t>(dev_ctx.stream());
      AOTOPS_DEBUG(kSoftmax, params);
      GCUOPS_TRACE_START(softmax);
      auto func_ptr = GetOpFuncPtr(kSoftmax, params);
      if (func_ptr) {
        auto pass = hlir::HlirDispatch::dispatch(func_ptr, params);
        PADDLE_ENFORCE(
            pass,
            phi::errors::InvalidArgument("dispatch %s failed!", kSoftmax));
      } else {
        PADDLE_ENFORCE(
            false,
            phi::errors::InvalidArgument("not find aot func for %s", kSoftmax));
      }
      FreeDispatchParam(params);
      GCUOPS_TRACE_END(softmax);
      GcuOpStreamSync(dev_ctx);
    }
    PADDLE_GCU_KERNEL_END("softmax", softmax);
  } else {
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

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(softmax,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::SoftmaxKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(softmax_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::SoftmaxGradKernel,
                          float,
                          double,
                          phi::dtype::float16) {}
