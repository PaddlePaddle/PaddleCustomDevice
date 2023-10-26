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
#include <mutex>  // NOLINT [build/c++11]
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "backend/utils/types.h"
#include "dtu/hlir_builder/hlir_builder.h"
#include "gcu/umd/dtu_assembler_def.h"
#include "paddle/phi/common/data_type.h"

namespace backend {

class TransformUtil {
 public:
  static std::string GetShapeStr(const std::vector<int64_t>& shape);
  /*
   * Parameters:
   *     type: [DataType] the data type
   * Return：
   *     [GcuPrimitiveType] the data type for hlir tensor
   * */
  static GcuPrimitiveType ConvertDataType(const phi::DataType& type);

  /*
   * Parameters:
   *     gcu_builder: gcu_builder
   *     pt： gcu primitive type
   * Return：
   *     <max_op, min_op>
   * */
  static std::pair<builder::Op, builder::Op> GenerateNumericLimit(
      GcuBuilderPtr gcu_builder, GcuPrimitiveType pt);

  /*
   * Parameters:
   *     gcu_builder: gcu_builder
   *     type : gcu type
   * Return：
   *     const op
   * */
  static bool IsDyn(const std::vector<int64_t>& shape);

  static builder::Op GetConst(const GcuBuilderPtr& gcu_builder,
                              const GcuPrimitiveType& type,
                              const double& target);

  static int64_t StringToNumber(const std::string& str);
};

template <typename T>
static std::string VectorToString(std::vector<T> vec) {
  std::ostringstream os;
  os << "[";
  for (auto tmp : vec) {
    os << std::fixed << tmp << "; ";
  }
  os << "]";
  return os.str();
}

}  // namespace backend
