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

#include <string>
#include <vector>

#include "graph/types.h"
#include "paddle/phi/backends/device_ext.h"

#ifdef __cplusplus
extern "C" {
#endif

struct C_GE_Operator;
struct C_GE_TensorDesc;
struct C_GE_Tensor;
struct C_GE_Graph;
struct C_GE_Session;

C_GE_TensorDesc* CreateTensorDesc();

void DestroyTensorDesc(C_GE_TensorDesc*);

void TensorDescSetShape(C_GE_TensorDesc* desc, int64_t* dims, int64_t rank);

void TensorDescSetDType(C_GE_TensorDesc* desc, int64_t dtype);

void TensorDescSetFormat(C_GE_TensorDesc* desc, int64_t format);

uint8_t* TensorGetData(C_GE_Tensor* tensor);

size_t TensorGetSize(C_GE_Tensor* tensor);

void* SetTensor(C_GE_Tensor* tensor,
                void* data,
                int64_t* dims,
                int64_t rank,
                int64_t dtype,
                int64_t format);

C_GE_Tensor* CreateTensor();

void DestroyTensor(C_GE_Tensor* tensor);

C_GE_Session* GetSession(const C_Scope scope);

void SessionAddGraph(C_GE_Session* session, size_t graph_id, C_GE_Graph* graph);

void SessionRunGraph(C_GE_Session* session,
                     size_t graph_id,
                     C_GE_Tensor** ins,
                     size_t ins_count,
                     C_GE_Tensor** outs,
                     size_t outs_count);

C_GE_Graph* CreateGraph(const C_Scope scope, const char* name);

void GraphSetInput(C_GE_Graph* graph, C_GE_Operator** ops, size_t count);

void GraphSetOutput(C_GE_Graph* graph, C_GE_Operator** ops, size_t count);

void GraphSetTarget(C_GE_Graph* graph, C_GE_Operator** ops, size_t count);

C_GE_Operator* CreateOperator(const char* op_name, const char* op_type);

void DestroyOperator(C_GE_Operator* self);

void OperatorAddControlInput(C_GE_Operator* self, C_GE_Operator* op);

void OperatorSetInput(C_GE_Operator* self,
                      uint32_t input_index,
                      C_GE_Operator* op,
                      uint32_t output_index);

void OperatorSetAttrInt64(C_GE_Operator* self,
                          const char* attr_name,
                          int64_t attr_value);

void OperatorSetAttrInt32(C_GE_Operator* self,
                          const char* attr_name,
                          int32_t attr_value);

void OperatorSetAttrUint32(C_GE_Operator* self,
                           const char* attr_name,
                           uint32_t attr_value);

void OperatorSetAttrFloat(C_GE_Operator* self,
                          const char* attr_name,
                          float attr_value);

void OperatorSetAttrBool(C_GE_Operator* self,
                         const char* attr_name,
                         uint8_t attr_value);

void OperatorSetAttrString(C_GE_Operator* self,
                           const char* attr_name,
                           const char* attr_value);

void OperatorSetAttrTensor(C_GE_Operator* self,
                           const char* attr_name,
                           C_GE_Tensor* self_value);

void OperatorSetAttrInt64List(C_GE_Operator* self,
                              const char* attr_name,
                              const int64_t* list,
                              int count);

void OperatorSetAttrInt32List(C_GE_Operator* self,
                              const char* attr_name,
                              const int32_t* list,
                              int count);

void OperatorSetAttrUint32List(C_GE_Operator* self,
                               const char* attr_name,
                               const uint32_t* list,
                               int count);

void OperatorSetAttrFloatList(C_GE_Operator* self,
                              const char* attr_name,
                              const float* list,
                              int count);

void OperatorSetAttrBoolList(C_GE_Operator* self,
                             const char* attr_name,
                             const uint8_t* list,
                             int count);

void OperatorSetAttrStringList(C_GE_Operator* self,
                               const char* attr_name,
                               const char** list,
                               int count);

void OperatorUpdateInputDesc(C_GE_Operator* self,
                             const char* desc_name,
                             int64_t* dims,
                             int64_t rank,
                             int64_t dtype,
                             int64_t format);

void OperatorUpdateOutputDesc(C_GE_Operator* self,
                              const char* desc_name,
                              int64_t* dims,
                              int64_t rank,
                              int64_t dtype,
                              int64_t format);

#ifdef __cplusplus
}  // extern "C"
#endif

template <typename T>
inline void OperatorSetAttr(C_GE_Operator* self,
                            const std::string& attr_name,
                            const T& attr_value);

template <>
inline void OperatorSetAttr<int64_t>(C_GE_Operator* self,
                                     const std::string& attr_name,
                                     const int64_t& attr_value) {
  OperatorSetAttrInt64(self, attr_name.c_str(), attr_value);
}

template <>
inline void OperatorSetAttr<int32_t>(C_GE_Operator* self,
                                     const std::string& attr_name,
                                     const int32_t& attr_value) {
  OperatorSetAttrInt32(self, attr_name.c_str(), attr_value);
}

template <>
inline void OperatorSetAttr<uint32_t>(C_GE_Operator* self,
                                      const std::string& attr_name,
                                      const uint32_t& attr_value) {
  OperatorSetAttrUint32(self, attr_name.c_str(), attr_value);
}

template <>
inline void OperatorSetAttr<float>(C_GE_Operator* self,
                                   const std::string& attr_name,
                                   const float& attr_value) {
  OperatorSetAttrFloat(self, attr_name.c_str(), attr_value);
}

template <>
inline void OperatorSetAttr<bool>(C_GE_Operator* self,
                                  const std::string& attr_name,
                                  const bool& attr_value) {
  OperatorSetAttrBool(self, attr_name.c_str(), attr_value);
}

template <>
inline void OperatorSetAttr<std::string>(C_GE_Operator* self,
                                         const std::string& attr_name,
                                         const std::string& attr_value) {
  OperatorSetAttrString(self, attr_name.c_str(), attr_value.c_str());
}

template <>
inline void OperatorSetAttr<std::vector<int64_t>>(
    C_GE_Operator* self,
    const std::string& attr_name,
    const std::vector<int64_t>& attr_value) {
  OperatorSetAttrInt64List(
      self, attr_name.c_str(), attr_value.data(), attr_value.size());
}
template <>
inline void OperatorSetAttr<std::vector<int32_t>>(
    C_GE_Operator* self,
    const std::string& attr_name,
    const std::vector<int32_t>& attr_value) {
  OperatorSetAttrInt32List(
      self, attr_name.c_str(), attr_value.data(), attr_value.size());
}

template <>
inline void OperatorSetAttr<std::vector<uint32_t>>(
    C_GE_Operator* self,
    const std::string& attr_name,
    const std::vector<uint32_t>& attr_value) {
  OperatorSetAttrUint32List(
      self, attr_name.c_str(), attr_value.data(), attr_value.size());
}

template <>
inline void OperatorSetAttr<std::vector<float>>(
    C_GE_Operator* self,
    const std::string& attr_name,
    const std::vector<float>& attr_value) {
  OperatorSetAttrFloatList(
      self, attr_name.c_str(), attr_value.data(), attr_value.size());
}

template <>
inline void OperatorSetAttr<std::vector<bool>>(
    C_GE_Operator* self,
    const std::string& attr_name,
    const std::vector<bool>& attr_value) {
  std::vector<uint8_t> uint8_list;
  for (auto i = 0; i < attr_value.size(); ++i) {
    uint8_list.push_back(attr_value[i]);
  }
  OperatorSetAttrBoolList(
      self, attr_name.c_str(), uint8_list.data(), uint8_list.size());
}

template <>
inline void OperatorSetAttr<std::vector<std::string>>(
    C_GE_Operator* self,
    const std::string& attr_name,
    const std::vector<std::string>& attr_value) {
  std::vector<const char*> string_list;
  for (auto i = 0; i < attr_value.size(); ++i) {
    string_list.push_back(attr_value[i].data());
  }
  OperatorSetAttrStringList(
      self, attr_name.c_str(), string_list.data(), string_list.size());
}

namespace ge {
namespace capi {

template <typename T, T* (*Allocator)(), void (*Deleter)(T*)>
class GEClassWrapper {};

// struct TensorDesc {
//   TensorDesc() { raw_data = CreateTensorDesc(); }

//   ~TensorDesc() {
//     DestroyTensorDesc(raw_data);
//     raw_data = nullptr;
//   }

//   TensorDesc(const TensorDesc& other) {

//   }

//   TensorDesc& operator=(TensorDesc& other) {

//     return *this;
//   }

//   void SetShape(const std::vector<int64_t>& shape) {
//     TensorDescSetShape(
//         raw_data, const_cast<int64_t*>(shape.data()), shape.size());
//   }

//   void SetDType(ge::DataType dtype) {
//     TensorDescSetDType(raw_data, static_cast<int64_t>(dtype));
//   }

//   void SetFormat(ge::Format format) {
//     TensorDescSetFormat(raw_data, static_cast<int64_t>(format));
//   }

//   C_GE_TensorDesc* raw_data{nullptr};
// };

}  // namespace capi
}  // namespace ge
