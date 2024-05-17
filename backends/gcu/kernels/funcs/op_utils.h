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

#pragma once
#include <string>
#include <vector>

#include "common/gcu_profiler.h"
#include "common/utils.h"
#include "paddle/phi/common/data_type.h"
#include "runtime/runtime.h"

namespace custom_kernel {
void *GcuDataPtr(const phi::DenseTensor &tensor);

std::string TensorToString(const phi::DenseTensor &tensor);

std::string TensorVectorToString(const std::vector<phi::DenseTensor> &tensors);

std::string ScalarToString(const phi::Scalar &scalar_value);

std::string GetOpNameFromCallStatement(const std::string &call_state);

std::vector<int64_t> InferSize(const std::vector<int64_t> &a,
                               const std::vector<int64_t> &b);

std::vector<int64_t> ComputeBroadcastShape(
    const std::vector<phi::DenseTensor> operands);

void GcuOpMaybeStreamSync(const phi::DeviceContext &dev_ctx);

template <typename T>
struct aot_op_variable_info {
  aot_op_variable_info(const T &var, const std::string &name) {
    std::stringstream ss;
    ss << "[" << name << ":" << var << "]; ";
    info = ss.str();
  }

  std::string info;
};

template <>
struct aot_op_variable_info<phi::DenseTensor> {
  aot_op_variable_info(const phi::DenseTensor &tensor,
                       const std::string &name) {
    std::stringstream ss;
    ss << "[" << name << ":" << TensorToString(tensor) << "]; ";
    info = ss.str();
  }

  std::string info;
};

template <>
struct aot_op_variable_info<paddle::optional<phi::DenseTensor>> {
  aot_op_variable_info(const paddle::optional<phi::DenseTensor> &opt_tensor,
                       const std::string &name) {
    std::stringstream ss;
    if (opt_tensor) {
      ss << aot_op_variable_info<phi::DenseTensor>(opt_tensor.get(), name).info;
    } else {
      ss << "OPTIONAL_NULLPTR_TENSOR";
    }
    info = ss.str();
  }

  std::string info;
};

template <>
struct aot_op_variable_info<std::vector<phi::DenseTensor>> {
  aot_op_variable_info(const std::vector<phi::DenseTensor> &tensor_list,
                       const std::string &name) {
    std::stringstream ss;
    ss << "[" << name << ":{";
    for (int64_t i = 0; i < tensor_list.size(); ++i) {
      std::string tensor_name = "list_tensor_" + std::to_string(i);
      ss << aot_op_variable_info<phi::DenseTensor>(tensor_list[i], tensor_name)
                .info;
    }
    ss << "}]; ";
    info = ss.str();
  }

  std::string info;
};

template <>
struct aot_op_variable_info<phi::Scalar> {
  aot_op_variable_info(const phi::Scalar &scalar, const std::string &name) {
    std::stringstream ss;
    ss << "[" << name << ":" << ScalarToString(scalar) << "]; ";
    info = ss.str();
  }

  std::string info;
};

template <>
struct aot_op_variable_info<paddle::optional<phi::Scalar>> {
  aot_op_variable_info(const paddle::optional<phi::Scalar> &opt_scalar,
                       const std::string &name) {
    std::stringstream ss;
    if (opt_scalar) {
      ss << aot_op_variable_info<phi::Scalar>(opt_scalar.get(), name).info;
    } else {
      ss << "OPTIONAL_NULLPTR_SCALAR";
    }
    info = ss.str();
  }

  std::string info;
};

template <>
struct aot_op_variable_info<phi::IntArray> {
  aot_op_variable_info(const phi::IntArray &int_array,
                       const std::string &name) {
    std::stringstream ss;
    ss << "[" << name << ":IntArray<"
       << VectorToStr<int64_t>(int_array.GetData()) << ">]; ";
    info = ss.str();
  }

  std::string info;
};

template <>
struct aot_op_variable_info<paddle::optional<phi::IntArray>> {
  aot_op_variable_info(const paddle::optional<phi::IntArray> &opt_array,
                       const std::string &name) {
    std::stringstream ss;
    if (opt_array) {
      ss << aot_op_variable_info<phi::IntArray>(opt_array.get(), name).info;
    } else {
      ss << "OPTIONAL_NULLPTR_INTARRAY";
    }
    info = ss.str();
  }

  std::string info;
};

template <>
struct aot_op_variable_info<std::vector<int64_t>> {
  aot_op_variable_info(const std::vector<int64_t> &int_array,
                       const std::string &name) {
    std::stringstream ss;
    ss << "[" << name << ":VectorInt64<" << VectorToStr<int64_t>(int_array)
       << ">]; ";
    info = ss.str();
  }

  std::string info;
};

template <>
struct aot_op_variable_info<phi::DataType> {
  aot_op_variable_info(const phi::DataType &data_type,
                       const std::string &name) {
    std::stringstream ss;
    ss << "[" << name << ":" << phi::DataTypeToString(data_type) << "]; ";
    info = ss.str();
  }

  std::string info;
};

template <>
struct aot_op_variable_info<std::pair<uint64_t, uint64_t>> {
  aot_op_variable_info(const std::pair<uint64_t, uint64_t> &gen_data,
                       const std::string &name) {
    std::stringstream ss;
    ss << "[" << name << ":"
       << "topsaotGenerator(seed:" << gen_data.first
       << ", offset:" << gen_data.second << ")"
       << "]; ";
    info = ss.str();
  }

  std::string info;
};

#define varNAME(var) #var

template <typename... Args>
inline std::string GetOpInfo(const std::string &op_name, const Args &...args) {
  auto variables_info = std::initializer_list<std::string>{
      aot_op_variable_info(args, varNAME(args)).info...};
  std::string op_info = op_name + ": {";
  for (std::string var_info : variables_info) {
    op_info += var_info;
  }
  op_info += "}\n";
  return op_info;
}

inline bool IsNarrowType(const phi::DataType &dtype) {
  return dtype == phi::DataType::FLOAT64 || dtype == phi::DataType::INT64;
}

inline void WarnTypeNarrow(const phi::DataType &dtype) {
  if (dtype == phi::DataType::FLOAT64) {
    LOG_FIRST_N(WARNING, 1)
        << "GCU not support " << phi::DataTypeToString(dtype)
        << ", use float32 replace, maybe lead to unexpected overflow issues.";
  } else if (dtype == phi::DataType::INT64) {
    LOG_FIRST_N(WARNING, 1)
        << "GCU not support " << phi::DataTypeToString(dtype)
        << ", use int32 replace, maybe lead to unexpected overflow issues.";
  }
}
}  // namespace custom_kernel
