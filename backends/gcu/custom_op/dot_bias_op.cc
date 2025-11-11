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
// #include "common/phi_funcs.h"
#include "paddle/extension.h"
#include "paddle/phi/backends/custom/custom_context.h"
#include "paddle/phi/core/dense_tensor.h"

std::vector<std::vector<int64_t>> DotBiasOpInferShape(
    std::vector<int64_t> dims_x,
    std::vector<int64_t> dims_y,
    bool trans_x,
    bool trans_y) {
  VLOG(6) << "---------- here is DotBiasOpInferShape ------";
  auto ndims_x = dims_x.size();
  auto ndims_y = dims_y.size();
  PADDLE_ENFORCE_GT(ndims_x,
                    0UL,
                    phi::errors::InvalidArgument(
                        "The Input(x) dims size must be greater than 0,"
                        " but received dims size is 0. "));
  PADDLE_ENFORCE_GT(ndims_y,
                    0UL,
                    phi::errors::InvalidArgument(
                        "The Input(y) dims size must be greater than 0,"
                        " but received dims size is 0. "));

  bool x_broadcasted = false, y_broadcasted = false;
  if (ndims_x == 1) {
    dims_x.insert(dims_x.begin(), 1);
    ndims_x = 2;
    x_broadcasted = true;
  }

  if (ndims_y == 1) {
    dims_y.push_back(1);
    ndims_y = 2;
    y_broadcasted = true;
  }

  size_t M = 0, N = 0;
  if (trans_x) {
    M = dims_x[ndims_x - 1];
  } else {
    M = dims_x[ndims_x - 2];
  }
  if (trans_y) {
    N = dims_y[ndims_y - 2];
  } else {
    N = dims_y[ndims_y - 1];
  }

  std::vector<int64_t> new_dims;
  if (ndims_x > ndims_y) {
    new_dims.assign(dims_x.begin(), dims_x.end() - 2);
  } else if (ndims_x < ndims_y) {
    new_dims.assign(dims_y.begin(), dims_y.end() - 2);
  } else {
    new_dims.reserve(ndims_x);
    for (size_t i = 0; i < ndims_x - 2; ++i) {
      new_dims.push_back(std::max(dims_x[i], dims_y[i]));
    }
  }
  if (!x_broadcasted) {
    new_dims.push_back(M);  // NOLINT
  }
  if (!y_broadcasted) {
    new_dims.push_back(N);  // NOLINT
  }

  return {new_dims};
}

std::vector<paddle::Tensor> DotBiasOp(const paddle::Tensor& x,
                                      const paddle::Tensor& y,
                                      const paddle::Tensor& y2,
                                      bool trans_x,
                                      bool trans_y,
                                      int axis) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(x.place()));

  auto x_tensor = static_cast<const phi::DenseTensor*>(x.impl().get());
  auto y_tensor = static_cast<const phi::DenseTensor*>(y.impl().get());
  auto y2_tensor = static_cast<const phi::DenseTensor*>(y2.impl().get());
  // Infershape
  auto out_shapes = DotBiasOpInferShape(x.shape(), y.shape(), trans_x, trans_y);
  paddle::Tensor out = paddle::empty(out_shapes[0], x.dtype(), x.place());

  custom_kernel::TensorNameMap input_names;
  custom_kernel::TensorValueMap inputs;
  input_names["X"] = {"x"};
  input_names["Y"] = {"y"};
  input_names["Y2"] = {"y2"};
  inputs["X"] = {const_cast<phi::DenseTensor*>(x_tensor)};
  inputs["Y"] = {const_cast<phi::DenseTensor*>(y_tensor)};
  inputs["Y2"] = {const_cast<phi::DenseTensor*>(y2_tensor)};

  custom_kernel::GcuAttributeMap attrs;
  attrs["trans_x"] = trans_x;
  attrs["trans_y"] = trans_y;
  attrs["axis"] = axis;

  custom_kernel::TensorNameMap output_names;
  custom_kernel::TensorValueMap outputs;
  output_names["Out"] = {"Out"};
  outputs["Out"] = {static_cast<phi::DenseTensor*>(out.impl().get())};

  custom_kernel::GcuRunner(
      input_names, inputs, output_names, outputs, attrs, "dot_bias", *dev_ctx);

  return {out};
}

PD_BUILD_OP(GCUDotBias)
    .Inputs({"X", "Y", "Y2"})
    .Outputs({"Out"})
    .Attrs({"trans_x: bool", "trans_y: bool", "axis: int"})
    .SetKernelFn(PD_KERNEL(DotBiasOp))
    .SetInferShapeFn(PD_INFER_SHAPE(DotBiasOpInferShape));
