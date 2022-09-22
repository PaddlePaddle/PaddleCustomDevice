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

#include <unistd.h>

#include "graph/paddle_graph.h"
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
namespace utils {

inline std::ostream& log() {
  static std::fstream ofs(
      "pd_ge.pid_" + std::to_string(static_cast<uint64_t>(getpid())) + ".txt",
      std::ios::out);
  return ofs;
}

template <typename T>
struct cpp_type_to_ge_dtype;

#define REG_CPP_TYPE_TO_GE_DTYPE(cpp_type, ge_dtype)          \
  template <>                                                 \
  struct cpp_type_to_ge_dtype<cpp_type> {                     \
    static const ge::DataType value = ge::DataType::ge_dtype; \
  };

REG_CPP_TYPE_TO_GE_DTYPE(int16_t, DT_INT16);
REG_CPP_TYPE_TO_GE_DTYPE(int32_t, DT_INT32);
REG_CPP_TYPE_TO_GE_DTYPE(int64_t, DT_INT64);
REG_CPP_TYPE_TO_GE_DTYPE(float, DT_FLOAT);
REG_CPP_TYPE_TO_GE_DTYPE(phi::dtype::float16, DT_FLOAT16);
REG_CPP_TYPE_TO_GE_DTYPE(phi::dtype::bfloat16, DT_BF16);
REG_CPP_TYPE_TO_GE_DTYPE(double, DT_DOUBLE);
REG_CPP_TYPE_TO_GE_DTYPE(phi::dtype::complex<float>, DT_COMPLEX64);
REG_CPP_TYPE_TO_GE_DTYPE(phi::dtype::complex<double>, DT_COMPLEX128);

inline ge::DataType string_to_ge_dtype(const std::string& data_type_str) {
  if (data_type_str == "float32") {
    return ge::DataType::DT_FLOAT;
  } else if (data_type_str == "float16") {
    return ge::DataType::DT_FLOAT16;
  } else if (data_type_str == "int8") {
    return ge::DataType::DT_INT8;
  } else if (data_type_str == "int16") {
    return ge::DataType::DT_INT16;
  } else if (data_type_str == "uint16") {
    return ge::DataType::DT_UINT16;
  } else if (data_type_str == "uint8") {
    return ge::DataType::DT_UINT8;
  } else if (data_type_str == "int32") {
    return ge::DataType::DT_INT32;
  } else if (data_type_str == "int64") {
    return ge::DataType::DT_INT64;
  } else if (data_type_str == "uint32") {
    return ge::DataType::DT_UINT32;
  } else if (data_type_str == "uint64") {
    return ge::DataType::DT_UINT64;
  } else if (data_type_str == "bool") {
    return ge::DataType::DT_BOOL;
  } else if (data_type_str == "float32") {
    return ge::DataType::DT_DOUBLE;
  } else if (data_type_str == "complex64") {
    return ge::DataType::DT_COMPLEX64;
  } else if (data_type_str == "complex128") {
    return ge::DataType::DT_COMPLEX128;
  } else {
    graph::utils::log() << "[ERROR] string_to_ge_dtype unknown data type: "
                        << data_type_str << std::endl;
    exit(-1);
  }
}

template <typename T>
struct cpp_type_to_pd_dtype;

#define REG_CPP_TYPE_TO_PD_DTYPE(cpp_type, pd_dtype)             \
  template <>                                                    \
  struct cpp_type_to_pd_dtype<cpp_type> {                        \
    static const paddle::framework::proto::VarType::Type value = \
        paddle::framework::proto::VarType::pd_dtype;             \
  };

REG_CPP_TYPE_TO_PD_DTYPE(int16_t, INT16);
REG_CPP_TYPE_TO_PD_DTYPE(int32_t, INT32);
REG_CPP_TYPE_TO_PD_DTYPE(int64_t, INT64);
REG_CPP_TYPE_TO_PD_DTYPE(float, FP32);
REG_CPP_TYPE_TO_PD_DTYPE(double, FP64);
REG_CPP_TYPE_TO_PD_DTYPE(phi::dtype::float16, FP16);
REG_CPP_TYPE_TO_PD_DTYPE(phi::dtype::bfloat16, BF16);
REG_CPP_TYPE_TO_PD_DTYPE(phi::dtype::complex<float>, COMPLEX64);
REG_CPP_TYPE_TO_PD_DTYPE(phi::dtype::complex<double>, COMPLEX128);

inline ge::DataType pd_dtype_to_ge_dtype(
    paddle::framework::proto::VarType::Type var_type) {
  if (var_type == paddle::framework::proto::VarType::BOOL) {
    return ge::DataType::DT_BOOL;
  } else if (var_type == paddle::framework::proto::VarType::INT16) {
    return ge::DataType::DT_INT16;
  } else if (var_type == paddle::framework::proto::VarType::INT32) {
    return ge::DataType::DT_INT32;
  } else if (var_type == paddle::framework::proto::VarType::INT64) {
    return ge::DataType::DT_INT64;
  } else if (var_type == paddle::framework::proto::VarType::FP16) {
    return ge::DataType::DT_FLOAT16;
  } else if (var_type == paddle::framework::proto::VarType::FP32) {
    return ge::DataType::DT_FLOAT;
  } else if (var_type == paddle::framework::proto::VarType::FP64) {
    return ge::DataType::DT_DOUBLE;
  } else if (var_type == paddle::framework::proto::VarType::SIZE_T) {
    return ge::DataType::DT_UINT64;
  } else if (var_type == paddle::framework::proto::VarType::UINT8) {
    return ge::DataType::DT_UINT8;
  } else if (var_type == paddle::framework::proto::VarType::INT8) {
    return ge::DataType::DT_INT8;
  } else if (var_type == paddle::framework::proto::VarType::BF16) {
    return ge::DataType::DT_BF16;
  } else if (var_type == paddle::framework::proto::VarType::COMPLEX64) {
    return ge::DataType::DT_COMPLEX64;
  } else if (var_type == paddle::framework::proto::VarType::COMPLEX128) {
    return ge::DataType::DT_COMPLEX128;
  } else {
    graph::utils::log()
        << "[ERROR] proto_var_type_to_ge_dtype unknown var_type: " << var_type
        << std::endl;
    exit(-1);
  }
}

inline int get_pd_dtype_size(paddle::framework::proto::VarType::Type var_type) {
  if (var_type == paddle::framework::proto::VarType::BOOL) {
    return sizeof(bool);
  } else if (var_type == paddle::framework::proto::VarType::INT16) {
    return sizeof(int16_t);
  } else if (var_type == paddle::framework::proto::VarType::INT32) {
    return sizeof(int32_t);
  } else if (var_type == paddle::framework::proto::VarType::INT64) {
    return sizeof(int64_t);
  } else if (var_type == paddle::framework::proto::VarType::FP16) {
    return sizeof(uint16_t);
  } else if (var_type == paddle::framework::proto::VarType::FP32) {
    return sizeof(float);
  } else if (var_type == paddle::framework::proto::VarType::FP64) {
    return sizeof(double);
  } else if (var_type == paddle::framework::proto::VarType::SIZE_T) {
    return sizeof(size_t);
  } else if (var_type == paddle::framework::proto::VarType::UINT8) {
    return sizeof(uint8_t);
  } else if (var_type == paddle::framework::proto::VarType::INT8) {
    return sizeof(int8_t);
  } else if (var_type == paddle::framework::proto::VarType::BF16) {
    return sizeof(uint16_t);
  } else if (var_type == paddle::framework::proto::VarType::COMPLEX64) {
    return sizeof(uint64_t);
  } else if (var_type == paddle::framework::proto::VarType::COMPLEX128) {
    return sizeof(uint64_t) * 2;
  } else {
    graph::utils::log()
        << "[ERROR] proto_var_type_to_ge_dtype unknown var_type: " << var_type
        << std::endl;
    exit(-1);
  }
}

}  // namespace utils
}  // namespace graph
