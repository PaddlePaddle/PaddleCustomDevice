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

#include "graph/ge_c_api.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <numeric>

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
#define GE_DESC(x) reinterpret_cast<ge::TensorDesc*>(x)
#define C_DESC(x) reinterpret_cast<C_GE_TensorDesc*>(x)

C_GE_TensorDesc* CreateTensorDesc() { return C_DESC(new ge::TensorDesc); }

void DestroyTensorDesc(C_GE_TensorDesc* desc) { return delete GE_DESC(desc); }

void TensorDescSetShape(C_GE_TensorDesc* desc, int64_t* dims, int64_t rank) {
  GE_DESC(desc)->SetShape(ge::Shape(std::vector<int64_t>(dims, dims + rank)));
}

void TensorDescSetDType(C_GE_TensorDesc* desc, int64_t dtype) {
  GE_DESC(desc)->SetDataType(static_cast<ge::DataType>(dtype));
}

void TensorDescSetFormat(C_GE_TensorDesc* desc, int64_t format) {
  GE_DESC(desc)->SetFormat(static_cast<ge::Format>(format));
}

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

void OperatorUpdateInputDesc(C_GE_Operator* self,
                             const char* desc_name,
                             int64_t* dims,
                             int64_t rank,
                             int64_t dtype,
                             int64_t format) {
  auto desc = GE_OP(self)->GetInputDescByName(desc_name);
  desc.SetShape(ge::Shape(std::vector<int64_t>(dims, dims + rank)));
  desc.SetDataType(static_cast<ge::DataType>(dtype));
  desc.SetFormat(static_cast<ge::Format>(format));
}

void OperatorUpdateOutputDesc(C_GE_Operator* self,
                              const char* desc_name,
                              int64_t* dims,
                              int64_t rank,
                              int64_t dtype,
                              int64_t format) {
  auto desc = GE_OP(self)->GetOutputDescByName(desc_name);
  desc.SetShape(ge::Shape(std::vector<int64_t>(dims, dims + rank)));
  desc.SetDataType(static_cast<ge::DataType>(dtype));
  desc.SetFormat(static_cast<ge::Format>(format));
}

namespace c_api {

bool ge_initialized = false;
std::map<ge::AscendString, ge::AscendString> config;
ge::Session* session = nullptr;
std::vector<std::shared_ptr<ge::Graph>> graph_cache;

}  // namespace c_api

C_GE_Session* GetSession(const C_Scope scope) {
  if (!c_api::session) {
    c_api::session = new ge::Session(c_api::config);
    return reinterpret_cast<C_GE_Session*>(c_api::session);
  }
}

void SessionAddGraph(C_GE_Session* session,
                     size_t graph_id,
                     C_GE_Graph* graph) {
  std::string graph_name = "graph_" + std::to_string(graph_id);

  if (ge::aclgrphDumpGraph(*reinterpret_cast<ge::Graph*>(graph),
                           graph_name.c_str(),
                           graph_name.size()) != ge::SUCCESS) {
  } else {
  }

  if (reinterpret_cast<ge::Session*>(session)->AddGraph(
          graph_id, *reinterpret_cast<ge::Graph*>(graph)) != ge::SUCCESS) {
    // std::cerr << "[ERROR] add graph  " << graph_id << ": " << graph
    //           << " failed." << std::endl;
  } else {
    // std::cerr << "[INFO] add graph " << graph_id << ": " << graph << "
    // success."
    //           << std::endl;
  }
}

void SessionRunGraph(C_GE_Session* session,
                     size_t graph_id,
                     C_GE_Tensor** ins,
                     size_t ins_count,
                     C_GE_Tensor** outs,
                     size_t outs_count) {
  if (ins_count == 0) {
    // SetInputs failed: input operator size can not be 0.
    return;
  }
  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  for (auto i = 0; i < ins_count; ++i) {
    inputs.push_back(*reinterpret_cast<ge::Tensor*>(ins[i]));
  }
  if (reinterpret_cast<ge::Session*>(session)->RunGraph(
          graph_id, inputs, outputs) != ge::SUCCESS) {
    std::cerr << "[ERROR] run graph " << graph_id << " failed\n";
  }
  for (auto i = 0; i < outputs.size(); ++i) {
    *reinterpret_cast<ge::Tensor*>(outs[i]) = outputs[i];
  }
}

C_GE_Graph* CreateGraph(const C_Scope scope, const char* name) {
  c_api::graph_cache.emplace_back(new ge::Graph(name));
  return reinterpret_cast<C_GE_Graph*>(c_api::graph_cache.back().get());
}

void GraphSetInput(C_GE_Graph* graph, C_GE_Operator** ops, size_t count) {
  std::vector<ge::Operator> ops_vec;
  for (auto i = 0; i < count; ++i) {
    ops_vec.push_back(*reinterpret_cast<ge::Operator*>(ops[i]));
  }
  reinterpret_cast<ge::Graph*>(graph)->SetInputs(ops_vec);
}

void GraphSetOutput(C_GE_Graph* graph, C_GE_Operator** ops, size_t count) {
  std::vector<ge::Operator> ops_vec;
  for (auto i = 0; i < count; ++i) {
    ops_vec.push_back(*reinterpret_cast<ge::Operator*>(ops[i]));
  }
  reinterpret_cast<ge::Graph*>(graph)->SetOutputs(ops_vec);
}

void GraphSetTarget(C_GE_Graph* graph, C_GE_Operator** ops, size_t count) {
  std::vector<ge::Operator> ops_vec;
  for (auto i = 0; i < count; ++i) {
    ops_vec.push_back(*reinterpret_cast<ge::Operator*>(ops[i]));
  }
  reinterpret_cast<ge::Graph*>(graph)->SetTargets(ops_vec);
}

C_GE_Tensor* CreateTensor() {
  return reinterpret_cast<C_GE_Tensor*>(new ge::Tensor);
}

uint8_t* TensorGetData(C_GE_Tensor* tensor) {
  return reinterpret_cast<ge::Tensor*>(tensor)->GetData();
}

size_t TensorGetSize(C_GE_Tensor* tensor) {
  return reinterpret_cast<ge::Tensor*>(tensor)->GetSize();
}

void* SetTensor(C_GE_Tensor* tensor,
                void* data,
                int64_t* dims,
                int64_t rank,
                int64_t dtype,
                int64_t format) {
  std::vector<int64_t> dims_vec(dims, dims + rank);
  auto count = std::accumulate(
      dims_vec.begin(), dims_vec.end(), 1, std::multiplies<int64_t>());
  auto desc = ge::TensorDesc(ge::Shape(dims_vec),
                             static_cast<ge::Format>(format),
                             static_cast<ge::DataType>(dtype));
  reinterpret_cast<ge::Tensor*>(tensor)->SetTensorDesc(desc);
  reinterpret_cast<ge::Tensor*>(tensor)->SetData(
      reinterpret_cast<uint8_t*>(data),
      count * ge::GetSizeByDataType(static_cast<ge::DataType>(dtype)));
}

void DestroyTensor(C_GE_Tensor* tensor) {
  delete reinterpret_cast<ge::Tensor*>(tensor);
}

C_Status graph_engine_initialize(const C_Device device, const C_Stream stream) {
  if (!c_api::ge_initialized) {
    c_api::ge_initialized = true;
    auto soc_name = aclrtGetSocName();
    c_api::config = {{"ge.exec.deviceId",
                      ge::AscendString(std::to_string(device->id).c_str())},
                     {"ge.graphRunMode", "0"},
                     {"ge.exec.precision_mode", "allow_fp32_to_fp16"},
                     {"ge.graphMemoryMaxSize", "22548578304"},
                     {"ge.variableMemoryMaxSize",
                      "10737418240"}, /* graphMemoryMaxSize +
                                         variableMemoryMaxSize <= 31 GB */
                     {"ge.socVersion", ge::AscendString(soc_name)},
                     {"ge.opSelectImplmode", "high_performance"}};

    ge::Status ret = ge::GEInitialize(c_api::config);
    if (ret != ge::SUCCESS) {
      // std::cerr << "[ERROR] graph_engine_initialize failed." << std::endl;
      return C_FAILED;
    }
    // std::cerr << "[INFO] graph_engine_initialize success." << std::endl;
  }
  return C_SUCCESS;
}

C_Status graph_engine_finalize(const C_Device device, const C_Stream stream) {
  if (c_api::ge_initialized) {
    c_api::ge_initialized = false;
    if (c_api::session) {
      //   graph_cache.clear();
      //   custom_graph::Tensor::TensorStorage().clear();
      delete c_api::session;
      c_api::session = nullptr;
    }

    ge::Status ret = ge::GEFinalize();
    if (ret != ge::SUCCESS) {
      // std::cerr << "[ERROR] graph_engine_finalize failed." << std::endl;
      return C_FAILED;
    }
    // std::cerr << "[INFO] graph_engine_finalize success." << std::endl;
  }
  return C_SUCCESS;
}
