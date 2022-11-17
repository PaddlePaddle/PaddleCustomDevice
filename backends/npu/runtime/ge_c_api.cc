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

void TensorDescSetDType(C_GE_TensorDesc* desc, ge::DataType dtype) {
  GE_DESC(desc)->SetDataType(dtype);
}

void TensorDescSetFormat(C_GE_TensorDesc* desc, ge::Format format) {
  GE_DESC(desc)->SetFormat(format);
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

C_GE_Tensor* OperatorGetAttrTensor(C_GE_Operator* self, const char* attr_name) {
  auto tensor = new ge::Tensor;
  GE_OP(self)->GetAttr(attr_name, *tensor);
  return C_TENSOR(tensor);
}

void OperatorUpdateInputDesc(C_GE_Operator* self,
                             const char* desc_name,
                             int64_t* dims,
                             int64_t rank,
                             ge::DataType dtype,
                             ge::Format format) {
  auto desc = GE_OP(self)->GetInputDescByName(desc_name);
  if (dims == nullptr) {
    desc.SetShape(ge::Shape(std::vector<int64_t>()));
    desc.SetOriginShape(ge::Shape({std::vector<int64_t>()}));
  } else {
    desc.SetShape(ge::Shape(std::vector<int64_t>(dims, dims + rank)));
    desc.SetOriginShape(ge::Shape(std::vector<int64_t>(dims, dims + rank)));
  }
  desc.SetDataType(dtype);
  desc.SetFormat(format);
  desc.SetOriginFormat(format);
  desc.SetRealDimCnt(rank);
  GE_OP(self)->UpdateInputDesc(desc_name, desc);
}

void OperatorUpdateOutputDesc(C_GE_Operator* self,
                              const char* desc_name,
                              int64_t* dims,
                              int64_t rank,
                              ge::DataType dtype,
                              ge::Format format) {
  auto desc = GE_OP(self)->GetOutputDescByName(desc_name);
  if (dims == nullptr) {
    desc.SetShape(ge::Shape(std::vector<int64_t>()));
    desc.SetOriginShape(ge::Shape({std::vector<int64_t>()}));
  } else {
    desc.SetShape(ge::Shape(std::vector<int64_t>(dims, dims + rank)));
    desc.SetOriginShape(ge::Shape(std::vector<int64_t>(dims, dims + rank)));
  }
  desc.SetDataType(dtype);
  desc.SetFormat(format);
  desc.SetOriginFormat(format);
  desc.SetRealDimCnt(rank);
  GE_OP(self)->UpdateOutputDesc(desc_name, desc);
}

char* OperatorGetOpType(C_GE_Operator* self) {
  auto type = GE_OP(self)->GetOpType();
  auto op_type = new char[type.size() + 1];
  memcpy(op_type, type.data(), type.size());
  op_type[type.size()] = '\0';
  return op_type;
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
  std::vector<ge::Tensor> inputs;
  std::vector<ge::Tensor> outputs;
  for (auto i = 0; i < ins_count; ++i) {
    inputs.push_back(*reinterpret_cast<ge::Tensor*>(ins[i]));
  }
  if (reinterpret_cast<ge::Session*>(session)->RunGraph(
          graph_id, inputs, outputs) != ge::SUCCESS) {
    std::cerr << "[ERROR] run graph " << graph_id << " failed\n";
  }
  if (outputs.size() != outs_count) {
    std::cerr << "[ERROR] SessionRunGraph outputs size != outs_count\n";
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
    ops_vec.push_back(*GE_OP(ops[i]));
  }
  reinterpret_cast<ge::Graph*>(graph)->SetInputs(ops_vec);
}

void GraphSetOutput(C_GE_Graph* graph,
                    C_GE_Operator** ops,
                    size_t* index,
                    size_t count) {
  std::vector<std::pair<ge::Operator, std::vector<size_t>>> ops_vec;
  for (auto i = 0; i < count; ++i) {
    ops_vec.push_back({*GE_OP(ops[i]), std::vector<size_t>({index[i]})});
  }
  reinterpret_cast<ge::Graph*>(graph)->SetOutputs(ops_vec);
}

void GraphSetTarget(C_GE_Graph* graph, C_GE_Operator** ops, size_t count) {
  std::vector<ge::Operator> ops_vec;
  for (auto i = 0; i < count; ++i) {
    ops_vec.push_back(*GE_OP(ops[i]));
  }
  reinterpret_cast<ge::Graph*>(graph)->SetTargets(ops_vec);
}

C_GE_Tensor* CreateTensor() { return C_TENSOR(new ge::Tensor); }

// C_GE_Tensor* CreateTensor(C_GE_TensorDesc* desc,
//                           ge::DataType dtype,
//                           ge::Format format) {
//   return C_TENSOR(new ge::Tensor(*GE_DESC(desc), dtype, format));
// }

uint8_t* TensorGetData(C_GE_Tensor* tensor) {
  return GE_TENSOR(tensor)->GetData();
}

size_t TensorGetSize(C_GE_Tensor* tensor) {
  return GE_TENSOR(tensor)->GetSize();
}

void* SetTensor(C_GE_Tensor* tensor,
                void* data,
                int64_t* dims,
                int64_t rank,
                ge::DataType dtype,
                ge::Format format) {
  std::vector<int64_t> dims_vec(dims, dims + rank);

  auto desc = ge::TensorDesc(ge::Shape(dims_vec), format, dtype);
  desc.SetRealDimCnt(desc.GetShape().GetDimNum());
  GE_TENSOR(tensor)->SetTensorDesc(desc);
  GE_TENSOR(tensor)->SetData(
      reinterpret_cast<uint8_t*>(data),
      std::accumulate(
          dims_vec.begin(), dims_vec.end(), 1, std::multiplies<int64_t>()) *
          ge::GetSizeByDataType(dtype));
}

void DestroyTensor(C_GE_Tensor* tensor) {
  if (tensor) {
    delete GE_TENSOR(tensor);
  }
}

C_Status graph_initialize(const C_Device device, const C_Stream stream) {
  if (!c_api::ge_initialized) {
    c_api::ge_initialized = true;
    auto soc_name = aclrtGetSocName();
    c_api::config = {
        {"ge.exec.deviceId",
         ge::AscendString(std::to_string(device->id).c_str())},
        {"ge.graphRunMode", "0"},
        {"ge.exec.precision_mode", "allow_fp32_to_fp16"},
        {"ge.graphMemoryMaxSize", "22548578304"},
        {"ge.variableMemoryMaxSize",
         "10737418240"},  // graphMemoryMaxSize + variableMemoryMaxSize <= 31 GB
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

C_Status graph_finalize(const C_Device device, const C_Stream stream) {
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
