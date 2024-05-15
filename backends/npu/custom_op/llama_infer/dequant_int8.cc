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

std::vector<std::vector<int64_t>> DequantInt8Shape(
    const std::vector<int64_t>& input_shape) {
  return {input_shape};
}

std::vector<paddle::DataType> DequantInt8Dtype(
    const paddle::DataType& input_dtype,
    const paddle::DataType& out_scale_dtype,
    std::string dtype) {
  paddle::DataType data_type;
  if (dtype == "float32")
    data_type = paddle::DataType::FLOAT32;
  else if (dtype == "bfloat16")
    data_type = paddle::DataType::BFLOAT16;
  else if (dtype == "float16")
    data_type = paddle::DataType::FLOAT16;
  else
    PD_THROW(
        "NOT supported data type. "
        "Only bfloat16, float16 and float32 are supported. ");
  return {data_type};
}

std::vector<paddle::Tensor> DequantInt8(const paddle::Tensor& input,
                                        const paddle::Tensor& out_scale,
                                        std::string dtype) {
  auto output_shape = DequantInt8Shape(input.shape());
  auto output_dtype = DequantInt8Dtype(input.dtype(), out_scale.dtype(), dtype);
  auto output =
      paddle::full(output_shape[0], 0, output_dtype[0], input.place());
  return {output};
}

PD_BUILD_OP(dequant_int8)
    .Inputs({"intput", "out_scale"})
    .Outputs({"output"})
    .Attrs({"dtype: std::string"})
    .SetKernelFn(PD_KERNEL(DequantInt8))
    .SetInferShapeFn(PD_INFER_SHAPE(DequantInt8Shape))
    .SetInferDtypeFn(PD_INFER_DTYPE(DequantInt8Dtype));
