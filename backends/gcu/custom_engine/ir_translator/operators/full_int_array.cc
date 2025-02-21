// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include <vector>

#include "custom_engine/ir_translator/translator_registry.h"

namespace custom_engine {

static GcuOpPtr TranslateFullIntArray(
    GcuBuilderPtr gcu_builder,
    const pir::Operation *op,
    const std::vector<std::vector<GcuOpPtr>> &gcu_op_inputs) {
  const auto &attrs = op->attributes();

  auto array_list =
      attrs.at("value").dyn_cast<pir::ArrayAttribute>().AsVector();
  std::vector<int64_t> vec_res;
  if (array_list.size() > 0) {
    PADDLE_ENFORCE_EQ(array_list[0].isa<pir::Int64Attribute>(),
                      true,
                      common::errors::Unimplemented(
                          "the 0th elementwise MUST be pir::Int64Attribute"));
    for (size_t i = 0; i < array_list.size(); ++i) {
      vec_res.push_back(array_list[i].dyn_cast<pir::Int64Attribute>().data());
    }
  }

  phi::DataType dtype =
      attrs.at("dtype").dyn_cast<paddle::dialect::DataTypeAttribute>().data();

  //   phi::Place place =
  //       attrs.at("place").dyn_cast<paddle::dialect::PlaceAttribute>().data();

  auto tmp_value_ptype =
      custom_engine::ConvertFromPhiDataType(phi::DataType::FLOAT32);
  auto tmp_value = builder::Const(
      gcu_builder, 0.0f, builder::Type(vec_res, tmp_value_ptype));

  auto shape_tensor = builder::Shape(tmp_value);
  auto ptype = custom_engine::ConvertFromPhiDataType(dtype);
  if (ptype == shape_tensor.GetType().GetPrimitiveType()) {
    return std::make_shared<GcuOp>(shape_tensor);
  }
  builder::Type output_type(shape_tensor.GetType().GetShape(), ptype);
  return std::make_shared<GcuOp>(builder::Convert(shape_tensor, output_type));
}

}  // namespace custom_engine

REGISTER_OP_TRANSLATOR(pd_op_full_int_array,
                       custom_engine::TranslateFullIntArray)
