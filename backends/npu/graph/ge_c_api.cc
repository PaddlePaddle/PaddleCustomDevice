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

#include "ge_c_api.h"

// NOLINT
#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "all_ops.h"
#include "ge/ge_api.h"
#include "ge/ge_api_types.h"
#include "ge/ge_error_codes.h"
#include "ge/ge_ir_build.h"
#include "graph/graph.h"
#include "graph/tensor.h"
#include "graph/types.h"

#define GE_OP(x) reinterpret_cast<ge::Operator*>(x)
#define C_OP(x) reinterpret_cast<C_GE_Operator*>(x)
#define GE_TENSOR(x) reinterpret_cast<ge::Tensor*>(x)
#define C_TENSOR(x) reinterpret_cast<C_GE_Tensor*>(x)

C_GE_Operator* CreateOperator(const char* op_name, const char* op_type) {
  auto op = new ge::Operator;
  *op = ge::OperatorFactory::CreateOperator(op_name, op_type);
  return C_OP(op);
}

void DestroyOperator(C_GE_Operator* self) { delete GE_OP(self); }

void OperatorAddControlInput(C_GE_Operator* self, C_GE_Operator* op) {
  GE_OP(self)->AddControlInput(*GE_OP(op));
}

void OperatorSetInput(C_GE_Operator* self,
                      uint32_t input_index,
                      C_GE_Operator* op,
                      uint32_t output_index) {
  GE_OP(self)->SetInput(input_index, *GE_OP(op), output_index);
}

void OperatorSetAttrInt64(C_GE_Operator* self,
                          const char* attr_name,
                          int64_t attr_value) {
  GE_OP(self)->SetAttr(attr_name, attr_value);
}

void OperatorSetAttrInt32(C_GE_Operator* self,
                          const char* attr_name,
                          int32_t attr_value) {
  GE_OP(self)->SetAttr(attr_name, attr_value);
}

void OperatorSetAttrUint32(C_GE_Operator* self,
                           const char* attr_name,
                           uint32_t attr_value) {
  GE_OP(self)->SetAttr(attr_name, attr_value);
}

void OperatorSetAttrFloat(C_GE_Operator* self,
                          const char* attr_name,
                          float attr_value) {
  GE_OP(self)->SetAttr(attr_name, attr_value);
}

void OperatorSetAttrBool(C_GE_Operator* self,
                         const char* attr_name,
                         uint8_t attr_value) {
  GE_OP(self)->SetAttr(attr_name, static_cast<bool>(attr_value));
}

void OperatorSetAttrString(C_GE_Operator* self,
                           const char* attr_name,
                           const char* attr_value) {
  GE_OP(self)->SetAttr(attr_name, ge::AscendString(attr_value));
}

void OperatorSetAttrTensor(C_GE_Operator* self,
                           const char* attr_name,
                           C_GE_Tensor* self_value) {
  GE_OP(self)->SetAttr(attr_name, *GE_TENSOR(self_value));
}

void OperatorSetAttrInt64List(C_GE_Operator* self,
                              const char* attr_name,
                              const int64_t* list,
                              int count) {
  GE_OP(self)->SetAttr(attr_name, std::vector<int64_t>(list, list + count));
}

void OperatorSetAttrInt32List(C_GE_Operator* self,
                              const char* attr_name,
                              const int32_t* list,
                              int count) {
  GE_OP(self)->SetAttr(attr_name, std::vector<int32_t>(list, list + count));
}

void OperatorSetAttrUint32List(C_GE_Operator* self,
                               const char* attr_name,
                               const uint32_t* list,
                               int count) {
  GE_OP(self)->SetAttr(attr_name, std::vector<uint32_t>(list, list + count));
}

void OperatorSetAttrFloatList(C_GE_Operator* self,
                              const char* attr_name,
                              const float* list,
                              int count) {
  GE_OP(self)->SetAttr(attr_name, std::vector<float>(list, list + count));
}

void OperatorSetAttrBoolList(C_GE_Operator* self,
                             const char* attr_name,
                             const uint8_t* list,
                             int count) {
  std::vector<bool> bool_list;
  for (auto i = 0; i < count; ++i) {
    bool_list.push_back(list[i]);
  }
  GE_OP(self)->SetAttr(attr_name, bool_list);
}

void OperatorSetAttrStringList(C_GE_Operator* self,
                               const char* attr_name,
                               const char** list,
                               int count) {
  std::vector<ge::AscendString> string_list;
  for (auto i = 0; i < count; ++i) {
    string_list.push_back(ge::AscendString(list[i]));
  }
  GE_OP(self)->SetAttr(attr_name, string_list);
}
