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

#include "paddle/extension.h"

std::vector<paddle::Tensor> QuantInt8(
    const paddle::Tensor& input,
    const paddle::optional<paddle::Tensor>& shift,
    const paddle::optional<paddle::Tensor>& smooth,
    float scale,
    int32_t round_type,
    float max_bound,
    float min_bound) {
  std::vector<int64_t> input_shape = input.shape();
  auto output =
      paddle::full(input_shape, -1, paddle::DataType::INT8, input.place());
  return {output};
}

std::vector<std::vector<int64_t>> QuantInt8Shape(
    const std::vector<int64_t>& input_shape,
    const paddle::optional<std::vector<int64_t>>& shift_shape,
    const paddle::optional<std::vector<int64_t>>& smooth_shape) {
  return {input_shape};
}

std::vector<paddle::DataType> QuantInt8Dtype(
    const paddle::DataType& input_dtype,
    const paddle::optional<paddle::DataType>& shift_dtype,
    const paddle::optional<paddle::DataType>& smooth_dtype) {
  return {paddle::DataType::INT8};
}

PD_BUILD_OP(quant_int8)
    .Inputs({"intput", paddle::Optional("shift"), paddle::Optional("smooth")})
    .Outputs({"output"})
    .Attrs({"scale: float",
            "round_type: int",
            "max_bound: float",
            "min_bound: float"})
    .SetKernelFn(PD_KERNEL(QuantInt8))
    .SetInferShapeFn(PD_INFER_SHAPE(QuantInt8Shape))
    .SetInferDtypeFn(PD_INFER_DTYPE(QuantInt8Dtype));
