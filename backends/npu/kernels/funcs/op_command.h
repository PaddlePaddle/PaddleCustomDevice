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

#include "glog/logging.h"
#include "graph/ge_c_api.h"
#include "graph/types.h"
#include "kernels/funcs/npu_enforce.h"
#include "paddle/phi/extension.h"
#include "paddle/utils/blank.h"
#include "paddle/utils/variant.h"
#include "runtime/runtime.h"

USE_ENV_bool(use_graph_engine);

#define GRAPH_RUN(expr)         \
  if (FLAGS_use_graph_engine) { \
    expr;                       \
  }
#define ACL_RUN(expr)            \
  if (!FLAGS_use_graph_engine) { \
    expr;                        \
  }

namespace custom_kernel {
namespace experimental {

aclDataType ConvertToNpuDtype(paddle::experimental::DataType dtype);
aclFormat ConvertToNpuFormat(phi::DataLayout layout);
ge::DataType ConvertToGEDtype(paddle::experimental::DataType dtype);
ge::Format ConvertToGEFormat(phi::DataLayout layout);

using NpuAttribute = paddle::variant<paddle::blank,
                                     int,
                                     float,
                                     std::string,
                                     std::vector<int>,
                                     std::vector<float>,
                                     std::vector<std::string>,
                                     bool,
                                     std::vector<bool>,
                                     int64_t,
                                     std::vector<int64_t>,
                                     std::vector<double>,
                                     std::vector<std::vector<int64_t>>,
                                     C_GE_Tensor*>;

struct Tensor;

struct IrNode {
  IrNode(const std::string& op_type) {
    static std::unordered_map<std::string, size_t> op_count;
    auto op_name_ = op_type + std::to_string(op_count[op_type]++);
    ge_op_ = CreateOperator(op_name_.c_str(), op_type.c_str());
  }

  ~IrNode() {
    if (ge_op_) {
      DestroyOperator(ge_op_);
      ge_op_ = nullptr;
    }
  }

  std::vector<Tensor*> ins_;
  C_GE_Operator* ge_op_{nullptr};
};

struct Tensor {
  Tensor() {}

  Tensor(const phi::DenseTensor& tensor) {
    ACL_RUN({
      // create aclTensorDesc
      auto dtype = ConvertToNpuDtype(tensor.dtype());
      auto format = ConvertToNpuFormat(tensor.layout());
      auto dims = phi::vectorize(tensor.dims());
      int size = dims.size();

      VLOG(4) << "NPU dtype:" << dtype << " "
              << "rank:" << dims.size() << " dims: " << tensor.dims()
              << " format:" << format;

      desc_ = aclCreateTensorDesc(dtype, size, dims.data(), format);
      PADDLE_ENFORCE_NOT_NULL(
          desc_, phi::errors::External("Call aclCreateTensorDesc failed."));
      PADDLE_ENFORCE_NPU_SUCCESS(aclSetTensorStorageFormat(desc_, format));
      PADDLE_ENFORCE_NPU_SUCCESS(
          aclSetTensorStorageShape(desc_, size, dims.data()));

      // create aclDataBuffer
      auto* ptr = tensor.data();
      VLOG(4) << "NPU ptr: " << ptr << ", size: " << tensor.capacity();
      buffer_ = aclCreateDataBuffer(const_cast<void*>(ptr), tensor.capacity());
    });
  }

  Tensor(const std::vector<int64_t>& tensor) {
    ACL_RUN({
      vec_ = tensor;
      // create aclTensorDesc
      auto dtype = ConvertToNpuDtype(phi::DataType::INT64);
      auto format = ConvertToNpuFormat(phi::DataLayout::ANY);
      std::vector<int64_t> dims({vec_.size()});
      int size = dims.size();

      VLOG(4) << "NPU dtype:" << dtype << " "
              << "rank:" << dims.size() << " dims: " << vec_.size()
              << " format:" << format;

      desc_ = aclCreateTensorDesc(dtype, size, dims.data(), format);
      PADDLE_ENFORCE_NOT_NULL(
          desc_, phi::errors::External("Call aclCreateTensorDesc failed."));
      PADDLE_ENFORCE_NPU_SUCCESS(aclSetTensorStorageFormat(desc_, format));
      PADDLE_ENFORCE_NPU_SUCCESS(
          aclSetTensorStorageShape(desc_, size, dims.data()));
      PADDLE_ENFORCE_NPU_SUCCESS(
          aclSetTensorPlaceMent(desc_, ACL_MEMTYPE_HOST));

      // create aclDataBuffer
      auto* ptr = vec_.data();
      VLOG(4) << "NPU ptr: " << ptr
              << ", size: " << vec_.size() * sizeof(int64_t);
      buffer_ = aclCreateDataBuffer(reinterpret_cast<void*>(ptr),
                                    vec_.size() * sizeof(int64_t));
    });
  }

  // graph tensor
  std::shared_ptr<IrNode> in_{nullptr};
  size_t in_index_{0};
  std::vector<IrNode*> outs_;
  C_GE_Tensor* ge_tensor_{nullptr};  // graph host tensor
  bool is_input_{false};             // mark tensor without node as input

  // acl tensor
  aclDataBuffer* buffer_{nullptr};
  aclTensorDesc* desc_{nullptr};
  std::vector<int64_t> vec_;  // acl host tensor
};

class NpuCommand {
 public:
  NpuCommand(const std::string& op_type) : op_type_(op_type) {}

  virtual ~NpuCommand() {}

  virtual void AddInput(Tensor&) = 0;

  virtual void AddOutput(Tensor&) = 0;

  virtual void AddAttribute(const std::string&, const NpuAttribute&) = 0;

  virtual void Run(const phi::CustomContext&) = 0;

 protected:
  const std::string op_type_;
};

class OpCommand {
 public:
  OpCommand(const std::string& op_type);

  OpCommand& Input(const std::vector<int64_t>& tensor) {
    Tensor t(tensor);
    cmd_->AddInput(t);
    return *this;
  }

  OpCommand& Input(Tensor& tensor) {
    cmd_->AddInput(tensor);
    return *this;
  }

  OpCommand& Output(Tensor& tensor) {
    cmd_->AddInput(tensor);
    return *this;
  }

  OpCommand& Input(const phi::DenseTensor& tensor);

  OpCommand& Output(const phi::DenseTensor& tensor);

  OpCommand& Attr(const std::string& key, const NpuAttribute& value) {
    cmd_->AddAttribute(key, value);
    return *this;
  }

  void Run(const phi::CustomContext& ctx) { cmd_->Run(ctx); }

 private:
  std::shared_ptr<NpuCommand> cmd_;
  // std::vector<phi::DenseTensor> storage_;
};

class TensorHelper {
 public:
  static void Assign(const phi::DenseTensor& dst, const phi::DenseTensor& src);

  static void Cache(OpCommand& cmd_, const phi::DenseTensor& src);
};

}  // namespace experimental
}  // namespace custom_kernel
