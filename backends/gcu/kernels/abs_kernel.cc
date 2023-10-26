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
#include "kernels/funcs/gcu_name_list.h"
#include "kernels/funcs/gcu_op_runner.h"
#include "paddle/phi/common/type_traits.h"
namespace custom_kernel {
// export FLAGS_auto_growth_chunk_size_in_mb=512
//  ------ test_abs.py -------
// import paddle
// paddle.set_device('gcu')
// x = paddle.to_tensor([-0.3, -0.2, 0.1, 0.3])
// out = paddle.abs(x)
// print(out)

template <typename T, typename Context>
void AbsKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "abs", abs);
    abs_compute(static_cast<const phi::CustomContext&>(dev_ctx), x, out);
    PADDLE_GCU_KERNEL_END("abs", abs);
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

    GcuRunner(
        input_names, inputs, output_names, outputs, attrs, "abs", dev_ctx);
  }
}

template <typename T, typename Context>
void AbsGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& dout,
                   phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);

  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "abs_grad", abs_grad);
    //           /  1, x > 0
    // dy / dx = -  0, x = 0
    //           \ -1, x < 0
    auto zero = zeros_like(dev_ctx, x);
    auto pred_negative = less_than_compute(dev_ctx, x, zero);
    auto temp =
        select(dev_ctx, pred_negative, neg_compute(dev_ctx, dout), dout);
    auto pred_positive = equal_compute(dev_ctx, x, zero);
    *dx = select(dev_ctx, pred_positive, zero, temp);
    PADDLE_GCU_KERNEL_END("abs_grad", abs_grad);
  } else {
    TensorNameMap input_names;
    input_names["X"] = {"x"};
    input_names[GradVarName("Out")] = {"dout"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};
    inputs[GradVarName("Out")] = {const_cast<DenseTensor*>(&dout)};

    TensorNameMap output_names;
    output_names[GradVarName("X")] = {"dx"};

    TensorValueMap outputs;
    outputs[GradVarName("X")] = {dx};

    GcuAttributeMap attrs;

    GcuRunner(
        input_names, inputs, output_names, outputs, attrs, "abs_grad", dev_ctx);
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
