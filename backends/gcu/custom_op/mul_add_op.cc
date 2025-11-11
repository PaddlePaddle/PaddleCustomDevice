// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <iostream>
#include <vector>

#include "common/gcu_op_runner.h"
#include "kernels/funcs/gcu_kernel_funcs.h"
#include "paddle/extension.h"

std::vector<std::vector<int64_t>> MulAddOpInferShape(
    const std::vector<int64_t>& x_shape) {
  return {x_shape};
}

std::vector<paddle::Tensor> MulAddOp(const paddle::Tensor& x,
                                     const paddle::Tensor& y,
                                     int axis1,
                                     int axis2) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(x.place()));

  auto x_tensor = static_cast<const phi::DenseTensor*>(x.impl().get());
  auto y_tensor = static_cast<const phi::DenseTensor*>(y.impl().get());
  // Infershape
  auto out_shapes = MulAddOpInferShape(x.shape());
  paddle::Tensor out = paddle::empty(out_shapes[0], x.dtype(), x.place());

  custom_kernel::TensorNameMap input_names;
  custom_kernel::TensorValueMap inputs;
  input_names["X"] = {"x"};
  input_names["Y"] = {"y"};
  inputs["X"] = {const_cast<phi::DenseTensor*>(x_tensor)};
  inputs["Y"] = {const_cast<phi::DenseTensor*>(y_tensor)};

  custom_kernel::GcuAttributeMap attrs;
  attrs["axis1"] = axis1;
  attrs["axis2"] = axis2;

  custom_kernel::TensorNameMap output_names;
  custom_kernel::TensorValueMap outputs;
  output_names["Out"] = {"Out"};
  outputs["Out"] = {static_cast<phi::DenseTensor*>(out.impl().get())};

  custom_kernel::GcuRunner(
      input_names, inputs, output_names, outputs, attrs, "mul_add", *dev_ctx);

  return {out};
}

PD_BUILD_OP(GCUMulAdd)
    .Inputs({"X", "Y"})
    .Outputs({"Out"})
    .Attrs({"axis1: int", "axis2: int"})
    .SetKernelFn(PD_KERNEL(MulAddOp))
    .SetInferShapeFn(PD_INFER_SHAPE(MulAddOpInferShape));
