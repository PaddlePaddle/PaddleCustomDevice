// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "graph/graph_utils.h"

// NOLINT
#include "all_ops.h"
#include "ge/ge_api.h"
#include "ge/ge_api_types.h"
#include "ge/ge_error_codes.h"
#include "ge/ge_ir_build.h"
#include "graph/graph.h"
#include "graph/tensor.h"
#include "graph/types.h"

namespace graph {
namespace funcs {

template <typename T>
inline ge::Operator constant(const std::vector<int>& dim,
                             const std::vector<T>& value,
                             ge::Format format = ge::Format::FORMAT_NCHW,
                             const std::string& name = "") {
  ge::TensorDesc tensor_desc(
      ge::Shape(std::vector<int64_t>(dim.begin(), dim.end())),
      format,
      graph::utils::cpp_type_to_ge_dtype<T>::value);
  tensor_desc.SetRealDimCnt(tensor_desc.GetShape().GetDimNum());
  ge::Tensor tensor(tensor_desc,
                    reinterpret_cast<uint8_t*>(const_cast<T*>(value.data())),
                    value.size() * sizeof(T));
  if (name.size() == 0) {
    auto constant_op = ge::op::Constant().set_attr_value(tensor);
    constant_op.update_output_desc_y(tensor_desc);
    return constant_op;
  } else {
    auto constant_op =
        ge::op::Constant(ge::AscendString(name.c_str())).set_attr_value(tensor);
    constant_op.update_output_desc_y(tensor_desc);
    return constant_op;
  }
}

template <typename T>
inline ge::Operator variable(const std::vector<int>& dim,
                             ge::Format format = ge::Format::FORMAT_NCHW,
                             const std::string& name = "") {
  ge::TensorDesc tensor_desc(
      ge::Shape(std::vector<int64_t>(dim.begin(), dim.end())),
      format,
      graph::utils::cpp_type_to_ge_dtype<T>::value);
  tensor_desc.SetRealDimCnt(tensor_desc.GetShape().GetDimNum());

  if (name.size() == 0) {
    auto variable = ge::op::Variable();
    variable.update_output_desc_y(tensor_desc);
    return variable;
  } else {
    auto variable = ge::op::Variable(ge::AscendString(name.c_str()));
    variable.update_output_desc_y(tensor_desc);
    return variable;
  }
}

template <typename T>
inline ge::Operator variable(const std::vector<int>& dim,
                             const std::vector<T>& value,
                             ge::Format format = ge::Format::FORMAT_NCHW,
                             const std::string& name = "") {
  ge::TensorDesc tensor_desc(
      ge::Shape(std::vector<int64_t>(dim.begin(), dim.end())),
      format,
      graph::utils::cpp_type_to_ge_dtype<T>::value);
  tensor_desc.SetRealDimCnt(tensor_desc.GetShape().GetDimNum());
  ge::Tensor tensor(tensor_desc,
                    reinterpret_cast<uint8_t*>(const_cast<T*>(value.data())),
                    value.size() * sizeof(T));

  if (name.size() == 0) {
    auto variable = ge::op::Variable();
    variable.update_output_desc_y(tensor_desc);
    auto constant_op = ge::op::Constant().set_attr_value(tensor);
    constant_op.update_output_desc_y(tensor_desc);
    auto assign_op =
        ge::op::Assign().set_input_ref(variable).set_input_value(constant_op);
    return variable;
  } else {
    auto variable = ge::op::Variable(ge::AscendString(name.c_str()));
    variable.update_output_desc_y(tensor_desc);
    auto constant_op = ge::op::Constant().set_attr_value(tensor);
    constant_op.update_output_desc_y(tensor_desc);
    auto assign_op =
        ge::op::Assign().set_input_ref(variable).set_input_value(constant_op);
    return variable;
  }
}

template <typename T>
inline ge::Operator get_output_by_name(
    ge::Operator& node,
    const std::vector<int>& dim,
    const std::string& output_name,
    ge::Format format = ge::Format::FORMAT_NCHW,
    const std::string& name = "") {
  ge::TensorDesc tensor_desc(
      ge::Shape(std::vector<int64_t>(dim.begin(), dim.end())),
      format,
      graph::utils::cpp_type_to_ge_dtype<T>::value);
  tensor_desc.SetRealDimCnt(tensor_desc.GetShape().GetDimNum());

  if (name.size() == 0) {
    auto variable = ge::op::Variable();
    variable.update_output_desc_y(tensor_desc);
    auto assign_op = ge::op::Assign().set_input_ref(variable).set_input_value(
        node, output_name);
    return variable;
  } else {
    auto variable = ge::op::Variable(ge::AscendString(name.c_str()));
    variable.update_output_desc_y(tensor_desc);
    auto assign_op = ge::op::Assign().set_input_ref(variable).set_input_value(
        node, output_name);
    return variable;
  }
}

inline ge::Operator get_output_by_name(
    ge::Operator& node,
    const std::vector<int>& dim,
    paddle::framework::proto::VarType::Type dtype,
    const std::string& output_name,
    ge::Format format = ge::Format::FORMAT_NCHW,
    const std::string& name = "") {
  ge::TensorDesc tensor_desc(
      ge::Shape(std::vector<int64_t>(dim.begin(), dim.end())),
      format,
      graph::utils::pd_dtype_to_ge_dtype(dtype));
  tensor_desc.SetRealDimCnt(tensor_desc.GetShape().GetDimNum());

  if (name.size() == 0) {
    auto variable = ge::op::Variable();
    variable.update_output_desc_y(tensor_desc);
    auto assign_op = ge::op::Assign().set_input_ref(variable).set_input_value(
        node, output_name);
    return variable;
  } else {
    auto variable = ge::op::Variable(ge::AscendString(name.c_str()));
    variable.update_output_desc_y(tensor_desc);
    auto assign_op = ge::op::Assign().set_input_ref(variable).set_input_value(
        node, output_name);
    return variable;
  }
}

inline ge::Operator reshape(ge::Operator& node,
                            const std::vector<int>& dim,
                            const std::string& name = "") {
  auto shape = graph::funcs::constant({dim.size()}, dim);

  if (name.size() == 0) {
    auto reshape_op =
        ge::op::Reshape().set_input_x(node).set_input_shape(shape);
    return reshape_op;
  } else {
    auto reshape_op = ge::op::Reshape(ge::AscendString(name.c_str()))
                          .set_input_x(node)
                          .set_input_shape(shape);
    return reshape_op;
  }
}

template <typename T>
inline ge::Operator cast(ge::Operator& node, const std::string& name = "") {
  if (name.size() == 0) {
    auto cast_op = ge::op::Cast().set_input_x(node).set_attr_dst_type(
        graph::utils::cpp_type_to_ge_dtype<T>::value);
    return cast_op;
  } else {
    auto cast_op =
        ge::op::Cast(ge::AscendString(name.c_str()))
            .set_input_x(node)
            .set_attr_dst_type(graph::utils::cpp_type_to_ge_dtype<T>::value);
    return cast_op;
  }
}

inline ge::Operator cast(ge::Operator& node,
                         paddle::framework::proto::VarType::Type dtype,
                         const std::string& name = "") {
  if (name.size() == 0) {
    auto cast_op = ge::op::Cast().set_input_x(node).set_attr_dst_type(
        graph::utils::pd_dtype_to_ge_dtype(dtype));
    return cast_op;
  } else {
    auto cast_op =
        ge::op::Cast(ge::AscendString(name.c_str()))
            .set_input_x(node)
            .set_attr_dst_type(graph::utils::pd_dtype_to_ge_dtype(dtype));
    return cast_op;
  }
}

inline void update_input_format(ge::Operator& node,
                                const std::string& input_name,
                                const std::string& format) {
  ge::TensorDesc desc = node.GetInputDescByName(input_name.c_str());
  if (format == "NCHW") {
    desc.SetFormat(ge::Format::FORMAT_NCHW);
  } else {
    desc.SetFormat(ge::Format::FORMAT_NHWC);
  }
  node.UpdateInputDesc(input_name.c_str(), desc);
}

inline void update_output_format(ge::Operator& node,
                                 const std::string& output_name,
                                 const std::string& format) {
  ge::TensorDesc desc = node.GetOutputDescByName(output_name.c_str());
  if (format == "NCHW") {
    desc.SetFormat(ge::Format::FORMAT_NCHW);
  } else {
    desc.SetFormat(ge::Format::FORMAT_NHWC);
  }
  node.UpdateOutputDesc(output_name.c_str(), desc);
}

}  // namespace funcs
}  // namespace graph
