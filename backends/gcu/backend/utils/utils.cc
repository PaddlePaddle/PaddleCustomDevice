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

#include "backend/utils/utils.h"

#include <limits>
#include <map>
#include <sstream>
#include <type_traits>

#include "backend/register/register.h"

namespace backend {
static std::map<std::string, std::vector<std::string>>
    map_optimizer_linkage_param = {
        {"adam",
         {"Grad",
          "Moment1",
          "Moment2",
          "Param",
          "Moment1Out",
          "Moment2Out",
          "ParamOut"}},
        {"adamw",
         {"Grad",
          "Moment1",
          "Moment2",
          "Param",
          "Moment1Out",
          "Moment2Out",
          "ParamOut"}},
        {"merged_adam",
         {"Grad",
          "Moment1",
          "Moment2",
          "Param",
          "Moment1Out",
          "Moment2Out",
          "ParamOut"}},
        {"momentum", {"Param", "Velocity", "ParamOut", "VelocityOut"}},
        {"merged_momentum", {"Param", "Velocity", "ParamOut", "VelocityOut"}},
        {"rmsprop",
         {"Param",
          "MeanSquare",
          "MeanGrad",
          "Grad",
          "Moment",
          "ParamOut",
          "MomentOut",
          "MeanSquareOut",
          "MeanGradOut"}},
        {"merged_adamw",
         {"Grad",
          "Moment1",
          "Moment2",
          "Param",
          "Moment1Out",
          "Moment2Out",
          "ParamOut"}}};

GcuPrimitiveType TransformUtil::ConvertDataType(const phi::DataType& type) {
  switch (type) {
    case phi::DataType::BOOL:
      return builder::PrimitiveType::PRED();
    case phi::DataType::INT8:
      return builder::PrimitiveType::S8();
    case phi::DataType::INT16:
      return builder::PrimitiveType::S16();
    case phi::DataType::INT32:
      return builder::PrimitiveType::S32();
    case phi::DataType::INT64:
      return builder::PrimitiveType::S64();
    case phi::DataType::FLOAT16:
      return builder::PrimitiveType::F16();
    case phi::DataType::FLOAT32:
      return builder::PrimitiveType::F32();
    case phi::DataType::FLOAT64:
      return builder::PrimitiveType::F64();
    case phi::DataType::UINT8:
      return builder::PrimitiveType::U8();
    case phi::DataType::UINT16:
      return builder::PrimitiveType::U16();
    case phi::DataType::UINT32:
      return builder::PrimitiveType::U32();
    case phi::DataType::UINT64:
      return builder::PrimitiveType::U64();

    default:
      return builder::PrimitiveType::NONE();
  }
}

std::string TransformUtil::GetShapeStr(const std::vector<int64_t>& shape) {
  std::stringstream ss;
  ss << "[";
  for (const auto& dim : shape) {
    ss << dim << ",";
  }
  ss << "]";
  return ss.str();
}

template <class T>
std::pair<builder::Op, builder::Op> _GenerateLimitOp(GcuBuilderPtr gcu_builder,
                                                     GcuPrimitiveType pt) {
  auto scalar_type = builder::Type(pt);
  T max_value = std::numeric_limits<T>::max();
  T min_value = std::numeric_limits<T>::min();
  if (std::is_same<typename std::decay<T>::type, float>::value) {
    max_value = static_cast<T>(1.0e20);   // NOLINT
    min_value = static_cast<T>(-1.0e20);  // NOLINT
  }
  void* max_ptr = static_cast<void*>(&max_value);
  void* min_ptr = static_cast<void*>(&min_value);
  auto max_op = builder::Const(gcu_builder, max_ptr, scalar_type);
  auto min_op = builder::Const(gcu_builder, min_ptr, scalar_type);
  return std::make_pair(max_op, min_op);
}

template <class T>
builder::Op _GenerateConstOp(GcuBuilderPtr gcu_builder,
                             GcuPrimitiveType pt,
                             double target) {
  auto scalar_type = builder::Type(pt);
  T value = static_cast<T>(target);
  return builder::Const(gcu_builder, static_cast<void*>(&value), scalar_type);
}

std::pair<builder::Op, builder::Op> TransformUtil::GenerateNumericLimit(
    GcuBuilderPtr gcu_builder, GcuPrimitiveType pt) {
  if (pt == builder::PrimitiveType::F32()) {
    return _GenerateLimitOp<float>(gcu_builder, pt);
  } else if (pt == builder::PrimitiveType::F64()) {
    return _GenerateLimitOp<double>(gcu_builder, pt);
  } else if (pt == builder::PrimitiveType::S8()) {
    return _GenerateLimitOp<int8_t>(gcu_builder, pt);
  } else if (pt == builder::PrimitiveType::S16()) {
    return _GenerateLimitOp<int16_t>(gcu_builder, pt);
  } else if (pt == builder::PrimitiveType::S32()) {
    return _GenerateLimitOp<int32_t>(gcu_builder, pt);
  } else if (pt == builder::PrimitiveType::S64()) {
    return _GenerateLimitOp<int64_t>(gcu_builder, pt);
  } else if (pt == builder::PrimitiveType::PRED()) {
    return _GenerateLimitOp<bool>(gcu_builder, pt);
  }
  PADDLE_THROW(phi::errors::Unimplemented("GCU unsupport data type!"));
}

bool TransformUtil::IsDyn(const std::vector<int64_t>& shape) {
  return std::any_of(
      shape.begin(), shape.end(), [](const int64_t& dim) { return dim < 0; });
}

builder::Op TransformUtil::GetConst(const GcuBuilderPtr& gcu_builder,
                                    const GcuPrimitiveType& type,
                                    const double& target) {
  if (type == builder::PrimitiveType::F32()) {
    return _GenerateConstOp<float>(gcu_builder, type, target);
  } else if (type == builder::PrimitiveType::F64()) {
    return _GenerateConstOp<double>(gcu_builder, type, target);
  } else if (type == builder::PrimitiveType::S8()) {
    return _GenerateConstOp<int8_t>(gcu_builder, type, target);
  } else if (type == builder::PrimitiveType::S16()) {
    return _GenerateConstOp<int16_t>(gcu_builder, type, target);
  } else if (type == builder::PrimitiveType::S32()) {
    return _GenerateConstOp<int32_t>(gcu_builder, type, target);
  } else if (type == builder::PrimitiveType::S64()) {
    return _GenerateConstOp<int64_t>(gcu_builder, type, target);
  } else if (type == builder::PrimitiveType::PRED()) {
    return _GenerateConstOp<bool>(gcu_builder, type, target);
  }
  PADDLE_THROW(phi::errors::Unimplemented("GCU unsupport data type!"));
}

int64_t TransformUtil::StringToNumber(const std::string& str) {
  try {
    return static_cast<int64_t>(std::stol(str));
  } catch (const std::invalid_argument&) {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "StringToNumber get invalid arg:%s", str.c_str()));
  } catch (const std::out_of_range&) {
    PADDLE_THROW(phi::errors::OutOfRange(
        "StringToNumber get OutOfRange error, input str:%s", str.c_str()));
  }
  return 0;
}
}  // namespace backend
