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
#include "kernels/funcs/npu_funcs.h"
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

struct TensorDescMaker {
  explicit TensorDescMaker(const std::string& desc_name)
      : desc_name_(desc_name) {}

  TensorDescMaker(const std::string& desc_name, const phi::DenseTensor& tensor)
      : desc_name_(desc_name),
        dims_(tensor.dims()),
        layout_(tensor.layout()),
        dtype_(tensor.dtype()) {
    if (tensor.place() == phi::CPUPlace()) {
      MarkAsHost();
    }
  }

  TensorDescMaker& SetDims(const phi::DDim& dims) {
    dims_ = dims;
    return *this;
  }

  TensorDescMaker& SetDataType(phi::DataType dtype) {
    dtype_ = dtype;
    return *this;
  }

  TensorDescMaker& SetDataLayout(phi::DataLayout layout) {
    layout_ = layout;
    return *this;
  }

  TensorDescMaker& ChangeStorage() {
    change_storage_ = true;
    return *this;
  }

  TensorDescMaker& MarkAsHost() {
    is_host_ = true;
    return *this;
  }

  TensorDescMaker& MarkAsConst() {
    is_const_ = true;
    return *this;
  }

  TensorDescMaker& MarkAsScalar() {
    is_scalar_ = true;
    return *this;
  }

  std::string desc_name_;
  phi::DDim dims_;
  phi::DataLayout layout_{phi::DataLayout::ANY};
  phi::DataType dtype_{phi::DataType::UNDEFINED};
  bool change_storage_{false};
  bool is_host_{false};
  bool is_const_{false};
  bool is_scalar_{false};
};

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
                                     phi::DenseTensor>;

struct Tensor;

struct IrNode {
  explicit IrNode(const std::string& op_type) {
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

  // graph tensor
  std::shared_ptr<IrNode> in_{nullptr};
  size_t in_index_{0};
  std::vector<std::pair<int, IrNode*>> outs_;
  C_GE_Tensor* ge_tensor_{nullptr};  // graph host tensor
  bool is_input_{false};             // mark tensor without node as input

  // acl tensor
  aclDataBuffer* buffer_{nullptr};
  aclTensorDesc* desc_{nullptr};
  std::vector<int64_t> vec_;  // acl host tensor
};

class NpuCommand {
 public:
  explicit NpuCommand(const std::string& op_type) : op_type_(op_type) {}

  virtual ~NpuCommand() {}

  virtual void AddInput(const phi::DenseTensor&, TensorDescMaker) = 0;
  virtual void AddHostInput(const phi::DenseTensor&, TensorDescMaker) = 0;
  virtual void AddScalarInput(const phi::DenseTensor&, TensorDescMaker) = 0;
  virtual void AddHostScalarInput(const phi::DenseTensor&, TensorDescMaker) = 0;
  virtual void AddOutput(phi::DenseTensor&, TensorDescMaker) = 0;

  virtual void AddInput(const phi::DenseTensor&) = 0;
  virtual void AddHostInput(const phi::DenseTensor&) = 0;
  virtual void AddScalarInput(const phi::DenseTensor&) = 0;
  virtual void AddHostScalarInput(const phi::DenseTensor&) = 0;
  virtual void AddOutput(phi::DenseTensor&) = 0;

  virtual void AddInput() = 0;
  virtual void AddOutput() = 0;
  virtual void AddInput(const phi::Scalar&) = 0;

  virtual void AddAttribute(const std::string&, const NpuAttribute&) = 0;

  virtual void Run(const phi::CustomContext&) = 0;

 protected:
  const std::string op_type_;
};

class OpCommand {
 public:
  explicit OpCommand(const std::string& op_type);

  OpCommand& Input(const phi::DenseTensor& tensor, TensorDescMaker maker);
  OpCommand& ScalarInput(const phi::DenseTensor& tensor, TensorDescMaker maker);
  OpCommand& Output(phi::DenseTensor& tensor, TensorDescMaker maker);

  OpCommand& Input(const phi::DenseTensor& tensor);
  OpCommand& ScalarInput(const phi::DenseTensor& tensor);
  OpCommand& Output(phi::DenseTensor& tensor);

  OpCommand& Input();
  OpCommand& Output();
  OpCommand& Input(const phi::Scalar& Scalar);

  OpCommand& Attr(const std::string& key, const NpuAttribute& value) {
    cmd_->AddAttribute(key, value);
    return *this;
  }

  void Run(const phi::CustomContext& ctx) { cmd_->Run(ctx); }

 private:
  std::shared_ptr<NpuCommand> cmd_;
  // std::vector<phi::DenseTensor> storage_;
};

struct OpCommandHelper {
  template <typename T>
  static void ScalarToHostTensor(const phi::CustomContext& ctx,
                                 const phi::Scalar& scalar,
                                 phi::DenseTensor* tensor) {
    tensor->Resize({1});
    ctx.template HostAlloc<T>(tensor);
    *(tensor->data<T>()) = scalar.to<T>();
  }

  template <typename T>
  static void ScalarToHostTensor(const phi::CustomContext& ctx,
                                 T scalar,
                                 phi::DenseTensor* tensor) {
    tensor->Resize({1});
    ctx.template HostAlloc<T>(tensor);
    *(tensor->data<T>()) = scalar;
  }

  template <typename T>
  static void VectorToHostTensor(const phi::CustomContext& ctx,
                                 const std::vector<T>& vector,
                                 phi::DenseTensor* tensor) {
    custom_kernel::TensorFromVector(ctx, vector, phi::CPUContext(), tensor);
  }

  static void Assign(const phi::CustomContext& ctx,
                     const phi::DenseTensor& src,
                     phi::DenseTensor* dst);

  template <typename Context>
  static void Reshape(const Context& dev_ctx,
                      const phi::DenseTensor& src,
                      const std::vector<int>& shape,
                      phi::DenseTensor* dst) {
    ACL_RUN({
      if (dst != &src) {
        *dst = src;
      }
      dst->Resize(phi::make_ddim(shape));
    });

    GRAPH_RUN({
      dst->Resize(phi::make_ddim(shape));
      dev_ctx.Alloc(dst, src.dtype());
      phi::DenseTensor shape_tensor;
      custom_kernel::TensorFromVector(
          dev_ctx, shape, phi::CPUContext(), &shape_tensor);
      experimental::OpCommand("Reshape")
          .Input(src)
          .Input(shape_tensor)
          .Output(*dst)
          .Run(dev_ctx);
    });
  }

  template <typename Context>
  static void BroadcastTo(const Context& dev_ctx,
                          const phi::DenseTensor& src,
                          int axis,
                          phi::DenseTensor* dst) {
    auto src_dims = phi::vectorize<int>(src.dims());
    if (src.dims().size() < dst->dims().size()) {
      for (auto i = 0; i < axis; ++i) {
        src_dims.insert(src_dims.begin(), 1);
      }
      for (auto i = src_dims.size(); i < dst->dims().size(); ++i) {
        src_dims.push_back(1);
      }
    }
    phi::DenseTensor dst_dims;
    custom_kernel::TensorFromVector(dev_ctx,
                                    phi::vectorize<int>(dst->dims()),
                                    phi::CPUContext(),
                                    &dst_dims);

    experimental::OpCommand("BroadcastTo")
        .Input(src,
               experimental::TensorDescMaker("x", src)
                   .SetDataLayout(phi::DataLayout::ANY)
                   .SetDims(phi::make_ddim(src_dims)))
        .Input(dst_dims,
               experimental::TensorDescMaker("shape", dst_dims)
                   .SetDataLayout(phi::DataLayout::ANY))
        .Output(*dst,
                experimental::TensorDescMaker("y", *dst).SetDataLayout(
                    phi::DataLayout::ANY))
        .Run(dev_ctx);
  }

  static void ElementwiseGradReduce(const phi::CustomContext& dev_ctx,
                                    const phi::DenseTensor& dout,
                                    int axis,
                                    phi::DenseTensor* dx) {
    std::vector<int> dst_dims_vec;
    std::vector<int> reduce_axes;
    auto src_dims = dx->dims();
    auto dout_dims = dout.dims();

    int src_axis = (src_dims.size() < dout_dims.size() ? axis : 0);
    for (int ax = 0; ax < dout_dims.size(); ++ax) {
      if ((ax < src_axis || ax >= src_axis + src_dims.size()) ||
          (dout_dims[ax] > 1 && src_dims[ax - src_axis] == 1)) {
        reduce_axes.push_back(ax);
      } else {
        dst_dims_vec.push_back(dout_dims[ax]);
      }
    }
    if (!reduce_axes.empty()) {
      experimental::OpCommand("ReduceSumD")
          .Input(dout)
          .Output(*dx,
                  experimental::TensorDescMaker("y", *dx)
                      .SetDataLayout(phi::DataLayout::ANY)
                      .SetDims(phi::make_ddim(dst_dims_vec)))
          .Attr("axes", reduce_axes)
          .Attr("keep_dims", false)
          .Run(dev_ctx);

      GRAPH_RUN({
        phi::DenseTensor out_dims_tensor;
        experimental::OpCommandHelper::VectorToHostTensor(
            dev_ctx, phi::vectorize<int>(dx->dims()), &out_dims_tensor);
        experimental::OpCommand("Reshape")
            .Input(*dx,
                   experimental::TensorDescMaker("x", *dx)
                       .SetDataLayout(phi::DataLayout::ANY)
                       .SetDims(phi::make_ddim(dst_dims_vec)))
            .Input(out_dims_tensor)
            .Output(*dx,
                    experimental::TensorDescMaker("y", *dx).SetDataLayout(
                        phi::DataLayout::ANY))
            .Run(dev_ctx);
      });
    }
  }
};
}  // namespace experimental
}  // namespace custom_kernel
