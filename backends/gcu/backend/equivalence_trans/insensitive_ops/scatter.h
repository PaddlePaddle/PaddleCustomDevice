/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "backend/equivalence_trans/utils.h"
#include "backend/register/register.h"

namespace backend {

const char* const kScatter = "scatter";

// This implementation followed the xla scatter_op documentation:
// https://www.tensorflow.org/xla/operation_semantics?hl=en#scatter
IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, op, map_inputs, running_mode, ScatterEquivalenceTrans) {
  auto CreateBindingFunc =
      [](std::shared_ptr<builder::Builder> builder,
         const std::vector<BindingFuncType>& func_types,
         const std::vector<builder::PrimitiveType>& ptypes,
         const std::string& base_name) -> std::vector<std::string> {
    PADDLE_ENFORCE_EQ(ptypes.size(),
                      func_types.size(),
                      phi::errors::NotFound(
                          "func_types:%zu must have same size with ptypes:%zu",
                          func_types.size(),
                          ptypes.size()));
    std::vector<std::string> func_names;
    func_names.reserve(func_types.size());
    builder::Op arg0, arg1, out;
    for (size_t i = 0; i < func_types.size(); ++i) {
      std::string func_name = base_name + std::to_string(i);
      builder->AddFunc(func_name.c_str());
      builder::Type scalar_type({1}, ptypes[i]);
      arg0 = builder->CreateInput(scalar_type, func_name.c_str());
      arg1 = builder->CreateInput(scalar_type, func_name.c_str());
      switch (func_types[i]) {
        case BindingFuncType::IDENTITY:
          out = arg0;
          break;
        case BindingFuncType::ADD:
          out = arg0 + arg1;
          break;
        case BindingFuncType::MAX:
          out = builder::Max(arg0, arg1);
          break;
        case BindingFuncType::MIN:
          out = builder::Min(arg0, arg1);
          break;
        case BindingFuncType::GE:
          out = builder::GreaterEqual(arg0, arg1);
          break;
        case BindingFuncType::OR:
          out = builder::Or(arg0, arg1);
          break;
        default:
          PADDLE_THROW(phi::errors::NotFound("Unsupport BindingFuncType."));
          break;
      }
      builder->SetOutput({out}, func_name.c_str());
      func_names.emplace_back(func_name);
    }
    return std::move(func_names);
  };

  auto input = *(map_inputs["X"].at(0));
  auto ids = *(map_inputs["Ids"].at(0));
  auto source = *(map_inputs["Updates"].at(0));
  auto overwrite = PADDLE_GET_CONST(bool, op->GetAttr("overwrite"));

  PADDLE_ENFORCE_EQ(
      overwrite,
      true,
      phi::errors::InvalidArgument("overwrite == False not support yet."));

  auto input_shape = input.GetType().GetShape();
  auto index_shape = ids.GetType().GetShape();
  auto source_shape = source.GetType().GetShape();

  if (index_shape.size() == 2) {
    PADDLE_ENFORCE_EQ(
        index_shape[1],
        1,
        phi::errors::InvalidArgument("index.dims()[1] should be 1 when "
                                     "index.dims().size() =2 in scatter_op."
                                     "But received value is [%d]",
                                     index_shape[1]));
  } else {
    PADDLE_ENFORCE_EQ(index_shape.size(),
                      1,
                      phi::errors::InvalidArgument(
                          "index.dims().size() should be 1 or 2 in scatter_op."
                          "But received value is [%d]",
                          index_shape.size()));
  }

  PADDLE_ENFORCE_GE(
      source_shape.size(),
      index_shape.size(),
      phi::errors::InvalidArgument("The rank (%d) of the updates "
                                   "for scatter op must larger than the "
                                   "rank (%d) of index.",
                                   index_shape.size(),
                                   source_shape.size()));

  int64_t input_rank = input_shape.size();
  PADDLE_ENFORCE_GE(
      input_rank,
      1,
      phi::errors::InvalidArgument("The rank (%d) of the input "
                                   "for scatter op must larger-equal than the "
                                   "1.",
                                   input_rank));

  size_t index_vector_dim = 1;

  // "If index_vector_dim is equal to scatter_indices.rank we implicitly
  // consider scatter_indices to have a trailing 1 dimension".
  if (index_shape.size() == index_vector_dim) {
    index_shape.push_back(1);
  }

  std::vector<int64_t> update_window_dims;
  std::vector<int64_t> inserted_window_dims;
  std::vector<int64_t> scatter_dims_to_operand_dims;

  for (int64_t i = 1; i < input_rank; i++) {
    update_window_dims.emplace_back(i);
  }
  // "operand.rank must equal the sum of update_window_dims.size and
  // inserted_window_dims.size".
  inserted_window_dims.emplace_back(0);

  // "scatter_dims_to_operand_dims.size must be equal to
  // scatter_indices.shape.dims[index_vector_dim], and its values must be in the
  // range [0, operand.rank)." for each dimension in index, gives the
  // corresponding dimension in input. Must be a sequence of integers with size
  // equal to index_shape[-1].
  for (int64_t i = 0; i < index_shape[index_vector_dim]; ++i) {
    scatter_dims_to_operand_dims.emplace_back(i);
  }

  auto input_ptype = input.GetType().GetPrimitiveType();
  builder::ScatterDimensionNumbers dim_numbers(update_window_dims,
                                               inserted_window_dims,
                                               scatter_dims_to_operand_dims,
                                               index_vector_dim);

  BindingFuncType build_func_type =
      overwrite ? BindingFuncType::IDENTITY : BindingFuncType::ADD;
  if (!overwrite) {
    // In Paddle, the non-overwrite mode will accumulate the results for each
    // value. Paddle by default uses the input value when the index not appears
    // in Ids. In constract, hlir_builder will set it to zero, since there is no
    // value to accumulate. To align the results with Paddle, we must **select**
    // the indexes which not in Ids, and mask them out.
    // In addition, the accumulation process of Paddle ignores the values in
    // input, but hlir_builder uses the value in input as the base of
    // accumulation.
    //
    // For an instance:
    // in Paddle:
    // scatter([1,2],[0,0],[2,2],overrite=False) => [4,2]
    // in hlir_builder:
    // builder::Scatter([1,2],[0,0],[2,2],dim_numbers, { BindingFuncType::ADD})
    // => [5,0] for overwrite=True, which means:
    // builder::Scatter([1,2],[0,0],[2,2],dim_numbers, {
    // BindingFuncType::IDENTITY}) => [2,2] We can use a **magic tensor** as a
    // probe, to generate a mask that can identify which index not apears in
    // Ids.

    auto nele = input_shape[0];
    PADDLE_ENFORCE_GE(nele,
                      1,
                      phi::errors::InvalidArgument(
                          "The dim 0 (%d) of the input "
                          "for scatter op must larger-equal than the "
                          "1.",
                          nele));

    if (input_ptype != builder::PrimitiveType::F32()) {
      PADDLE_THROW(phi::errors::InvalidArgument(
          "The input oprand of scatter op must be float32"));
    }

    GcuOp zero_operand = builder::ZerosLike(input);
    GcuOp magic_operand = builder::FullLike(input, 1.14512 * 1e-7);

    auto tmp_region_list = CreateBindingFunc(input.GetBuilder(),
                                             {BindingFuncType::IDENTITY},
                                             {input_ptype},
                                             "body1_");
    std::vector<const char*> region_list;
    for (auto& region : tmp_region_list) region_list.push_back(region.c_str());

    auto mask =
        builder::Scatter(magic_operand, ids, source, dim_numbers, region_list);
    mask = builder::Equal(magic_operand, mask);

    input = builder::Select(mask, input, zero_operand);
  }

  auto tmp_region_list = CreateBindingFunc(
      input.GetBuilder(), {build_func_type}, {input_ptype}, "body0_");
  std::vector<const char*> region_list;
  for (auto& region : tmp_region_list) region_list.push_back(region.c_str());
  auto out = builder::Scatter(input, ids, source, dim_numbers, region_list);

  return std::make_shared<GcuOp>(out);
}

EQUIVALENCE_TRANS_FUNC_REG(kScatter, INSENSITIVE, ScatterEquivalenceTrans);

}  // namespace backend
