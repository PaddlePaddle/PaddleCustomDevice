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

#include "kernels/funcs/op_command.h"

namespace custom_kernel {
namespace experimental {

struct GETensorHelper {
  static C_GE_Tensor *ConvertDenseTensorToGETensor(const phi::DenseTensor &t) {
    auto r = CreateTensor();
    auto dims = phi::vectorize(t.dims());
    SetTensor(r,
              const_cast<void *>(t.data()),
              dims.data(),
              dims.size(),
              experimental::ConvertToGEDtype(t.dtype()),
              ge::Format::FORMAT_ND);
    return r;
  }

  static C_GE_Tensor *ConvertDenseTensorToGETensor(const phi::DenseTensor &t,
                                                   phi::DataLayout layout) {
    auto r = CreateTensor();
    auto dims = phi::vectorize(t.dims());
    SetTensor(r,
              const_cast<void *>(t.data()),
              dims.data(),
              dims.size(),
              experimental::ConvertToGEDtype(t.dtype()),
              experimental::ConvertToGEFormat(layout));
    return r;
  }

  static C_GE_Tensor *ConvertScalarToGETensor(const phi::Scalar &scalar) {
    auto r = CreateTensor();
    std::vector<int64_t> dims({1});
    if (scalar.dtype() == phi::DataType::FLOAT16) {
      auto data = scalar.to<phi::dtype::float16>();
      SetTensor(r,
                reinterpret_cast<void *>(&data),
                dims.data(),
                dims.size(),
                experimental::ConvertToGEDtype(scalar.dtype()),
                ge::Format::FORMAT_ND);
    } else if (scalar.dtype() == phi::DataType::FLOAT32) {
      auto data = scalar.to<float>();
      SetTensor(r,
                reinterpret_cast<void *>(&data),
                dims.data(),
                dims.size(),
                experimental::ConvertToGEDtype(scalar.dtype()),
                ge::Format::FORMAT_ND);
    } else if (scalar.dtype() == phi::DataType::FLOAT64) {
      auto data = scalar.to<double>();
      SetTensor(r,
                reinterpret_cast<void *>(&data),
                dims.data(),
                dims.size(),
                experimental::ConvertToGEDtype(scalar.dtype()),
                ge::Format::FORMAT_ND);
    } else if (scalar.dtype() == phi::DataType::INT8) {
      auto data = scalar.to<int8_t>();
      SetTensor(r,
                reinterpret_cast<void *>(&data),
                dims.data(),
                dims.size(),
                experimental::ConvertToGEDtype(scalar.dtype()),
                ge::Format::FORMAT_ND);
    } else if (scalar.dtype() == phi::DataType::INT16) {
      auto data = scalar.to<int16_t>();
      SetTensor(r,
                reinterpret_cast<void *>(&data),
                dims.data(),
                dims.size(),
                experimental::ConvertToGEDtype(scalar.dtype()),
                ge::Format::FORMAT_ND);
    } else if (scalar.dtype() == phi::DataType::INT32) {
      auto data = scalar.to<int32_t>();
      SetTensor(r,
                reinterpret_cast<void *>(&data),
                dims.data(),
                dims.size(),
                experimental::ConvertToGEDtype(scalar.dtype()),
                ge::Format::FORMAT_ND);
    } else if (scalar.dtype() == phi::DataType::INT64) {
      auto data = scalar.to<int64_t>();
      SetTensor(r,
                reinterpret_cast<void *>(&data),
                dims.data(),
                dims.size(),
                experimental::ConvertToGEDtype(scalar.dtype()),
                ge::Format::FORMAT_ND);
    } else if (scalar.dtype() == phi::DataType::BOOL) {
      auto data = scalar.to<bool>();
      SetTensor(r,
                reinterpret_cast<void *>(&data),
                dims.data(),
                dims.size(),
                experimental::ConvertToGEDtype(scalar.dtype()),
                ge::Format::FORMAT_ND);
    } else {
      PADDLE_THROW(phi::errors::Unimplemented(
          "Can not convert data type %d scalar to tensor", scalar.dtype()));
    }
    return r;
  }

  template <typename T>
  static C_GE_Tensor *ConvertVectorToGETensor(const std::vector<T> &t) {
    auto r = CreateTensor();
    std::vector<int64_t> dims({t.size()});
    SetTensor(r,
              reinterpret_cast<void *>(const_cast<T *>(t.data())),
              dims.data(),
              dims.size(),
              custom_kernel::experimental::ConvertToGEDtype(
                  paddle::experimental::CppTypeToDataType<T>::Type()),
              ge::Format::FORMAT_ND);
    return r;
  }
};

/**
 *
 *  TensorNode must be the output of OpNode.
 *
 *  Op --> Tensor
 *
 */

struct TensorNode;

struct OpNode {
  explicit OpNode(const std::string &op_type) {
    static std::unordered_map<std::string, size_t> op_count;
    op_type_ = op_type;
    op_name_ = op_type + "_" + std::to_string(op_count[op_type]);
    op_count[op_type] = op_count[op_type] + 1;
    ge_op_ = CreateOperator(op_name_.c_str(), op_type.c_str());
  }

  ~OpNode() {
    if (ge_op_) {
      DestroyOperator(ge_op_);
      ge_op_ = nullptr;
    }
  }

  const std::string &Type() const { return op_type_; }

  const std::string &Name() const { return op_name_; }

  C_GE_Operator *GeOp() const { return ge_op_; }

  const std::vector<TensorNode *> &Inputs() const { return ins_; }
  const std::vector<TensorNode *> &Outputs() const { return outs_; }
  const std::vector<TensorDescMaker> &InputDescs() const { return in_descs_; }
  const std::vector<TensorDescMaker> &OutputDescs() const { return out_descs_; }

  void AddInput(TensorNode *in, const TensorDescMaker &desc) {
    ins_.push_back(in);
    in_descs_.push_back(desc);
  }

  void AddOutput(TensorNode *out, const TensorDescMaker &desc) {
    outs_.push_back(out);
    out_descs_.push_back(desc);
  }

  void AddInput(const std::shared_ptr<TensorNode> &in,
                const TensorDescMaker &desc) {
    AddInput(in.get(), desc);
  }

  void AddOutput(const std::shared_ptr<TensorNode> &out,
                 const TensorDescMaker &desc) {
    AddOutput(out.get(), desc);
  }

  void AddInput(TensorNode *in) { AddInput(in, TensorDescMaker()); }

  void AddOutput(TensorNode *out) { AddOutput(out, TensorDescMaker()); }

  void AddInput(const std::shared_ptr<TensorNode> &in) { AddInput(in.get()); }

  void AddOutput(const std::shared_ptr<TensorNode> &out) {
    AddOutput(out.get());
  }

  void UpdateInputDesc(size_t index) {
    PADDLE_ENFORCE_LT(
        index,
        in_descs_.size(),
        phi::errors::InvalidArgument(
            "The index out of range of in_descs_ in op %s.", Name()));

    auto &desc = in_descs_[index];
    if (desc.Valid()) {
      auto dims = phi::vectorize(desc.dims_);
      OperatorUpdateInputDesc(ge_op_,
                              desc.desc_name_.c_str(),
                              dims.data(),
                              dims.size(),
                              ConvertToGEDtype(desc.dtype_),
                              ConvertToGEFormat(desc.layout_));
    }
  }

  void UpdateOutputDesc(size_t index) {
    PADDLE_ENFORCE_LT(
        index,
        out_descs_.size(),
        phi::errors::InvalidArgument(
            "The index out of range of out_descs_ in op %s.", Name()));
    auto &desc = out_descs_[index];
    if (desc.Valid()) {
      auto dims = phi::vectorize(desc.dims_);
      OperatorUpdateOutputDesc(ge_op_,
                               desc.desc_name_.c_str(),
                               dims.data(),
                               dims.size(),
                               ConvertToGEDtype(desc.dtype_),
                               ConvertToGEFormat(desc.layout_));
    }
  }

 private:
  std::vector<TensorNode *> ins_;
  std::vector<TensorNode *> outs_;
  std::vector<TensorDescMaker> in_descs_;
  std::vector<TensorDescMaker> out_descs_;

  C_GE_Operator *ge_op_{nullptr};
  std::string op_type_;
  std::string op_name_;
};

enum TensorNodeTag {
  UNDEFINED = 0,
  IN,
  OUT,
  EDGE,
  PARAMETER,
};

struct TensorNode {
  static std::vector<std::shared_ptr<TensorNode>> &storage() {
    static std::vector<std::shared_ptr<TensorNode>> ins;
    return ins;
  }

  static TensorNode *malloc() {
    TensorNode::storage().push_back(std::make_shared<TensorNode>());
    VLOG(10) << "TensorNode::malloc " << TensorNode::storage().back().get();
    return TensorNode::storage().back().get();
  }

  static void free(TensorNode *p) { VLOG(10) << "TensorNode::free " << p; }

  TensorNode() {}

  bool WithoutNode() const { return !node_; }

  C_GE_Operator *NodeGeOp() const {
    return WithoutNode() ? nullptr : node_->GeOp();
  }

  std::shared_ptr<OpNode> Node() const { return node_; }

  size_t NodeIndex() const { return index_; }

  template <typename T>
  std::vector<T> NodeShape() const {
    return OperatorGetOutputShapeByIndex<T>(node_->GeOp(), index_);
  }

  void FromOther(std::shared_ptr<OpNode> other, size_t index) {
    PADDLE_ENFORCE(!node_,
                   phi::errors::InvalidArgument(
                       "The TensorNode already has an OpNode %s: %d",
                       node_->Name(),
                       index_));
    ResetFromOther(other, index);
  }

  void ResetFromOther(std::shared_ptr<OpNode> other, size_t index) {
    LOG(INFO) << "Reset tensor " << this
              << " FromOther: " << (node_.get() ? node_->Name() : "None") << ":"
              << index_ << " -> " << other->Name() << ":" << index;
    node_ = other;
    index_ = index;
  }

  const std::vector<std::pair<std::shared_ptr<OpNode>, size_t>> &Links() {
    return link_to_;
  }

  void LinkTo(std::shared_ptr<OpNode> node, size_t index) {
    link_to_.push_back({node, index});
  }

  void SetTag(TensorNodeTag tag) { tag_ = tag; }

  TensorNodeTag Tag() const { return tag_; }

  bool IsInput() { return tag_ == TensorNodeTag::IN; }

  bool IsParameter() { return tag_ == TensorNodeTag::PARAMETER; }

  std::shared_ptr<OpNode> node_{nullptr};
  size_t index_{0};
  std::vector<std::pair<std::shared_ptr<OpNode>, size_t>> link_to_;
  TensorNodeTag tag_{TensorNodeTag::UNDEFINED};
};

class GraphCommandHelper {
 public:
  static void ConvertScalarToHostTensor(const phi::CustomContext &ctx,
                                        phi::Scalar &scalar,
                                        phi::DenseTensor *tensor) {
    tensor->Resize({1});
    if (scalar.dtype() == phi::DataType::FLOAT16) {
      auto data = ctx.template HostAlloc<phi::dtype::float16>(tensor);
      *data = scalar.to<phi::dtype::float16>();
    } else if (scalar.dtype() == phi::DataType::FLOAT32) {
      auto data = ctx.template HostAlloc<float>(tensor);
      *data = scalar.to<float>();
    } else if (scalar.dtype() == phi::DataType::FLOAT64) {
      auto data = ctx.template HostAlloc<double>(tensor);
      *data = scalar.to<double>();
    } else if (scalar.dtype() == phi::DataType::INT8) {
      auto data = ctx.template HostAlloc<int8_t>(tensor);
      *data = scalar.to<int8_t>();
    } else if (scalar.dtype() == phi::DataType::INT16) {
      auto data = ctx.template HostAlloc<int16_t>(tensor);
      *data = scalar.to<int16_t>();
    } else if (scalar.dtype() == phi::DataType::INT32) {
      auto data = ctx.template HostAlloc<int32_t>(tensor);
      *data = scalar.to<int32_t>();
    } else if (scalar.dtype() == phi::DataType::INT64) {
      auto data = ctx.template HostAlloc<int64_t>(tensor);
      *data = scalar.to<int64_t>();
    } else if (scalar.dtype() == phi::DataType::BOOL) {
      auto data = ctx.template HostAlloc<uint8_t>(tensor);
      *data = static_cast<uint8_t>(scalar.to<bool>());
    } else {
      PADDLE_THROW(phi::errors::Unimplemented(
          "Can not convert data type %d scalar to tensor", scalar.dtype()));
    }
  }

  static std::shared_ptr<OpNode> ConvertHostTensorToConstOp(
      const phi::DenseTensor &t) {
    auto ge_tensor = GETensorHelper::ConvertDenseTensorToGETensor(t);
    auto const_op = std::make_shared<OpNode>("Const");
    OperatorSetAttrTensor(const_op->GeOp(), "value", ge_tensor);
    DestroyTensor(ge_tensor);

    auto dims = phi::vectorize(t.dims());
    OperatorUpdateOutputDesc(
        const_op->GeOp(),
        "y",
        dims.data(),
        dims.size(),
        custom_kernel::experimental::ConvertToGEDtype(t.dtype()),
        ge::Format::FORMAT_ND);
    return std::move(const_op);
  }

  static std::shared_ptr<OpNode> ConvertHostTensorToConstOp(
      const phi::DenseTensor &t,
      const std::vector<int64_t> &dims,
      phi::DataType dtype,
      phi::DataLayout format) {
    auto ge_tensor = GETensorHelper::ConvertDenseTensorToGETensor(t);
    auto const_op = std::make_shared<OpNode>("Const");
    OperatorSetAttrTensor(const_op->GeOp(), "value", ge_tensor);
    DestroyTensor(ge_tensor);

    OperatorUpdateOutputDesc(
        const_op->GeOp(),
        "y",
        const_cast<int64_t *>(dims.data()),
        dims.size(),
        custom_kernel::experimental::ConvertToGEDtype(dtype),
        custom_kernel::experimental::ConvertToGEFormat(format));
    return std::move(const_op);
  }

  static std::shared_ptr<OpNode> ConvertScalarToConstOp(const phi::Scalar &t) {
    auto ge_tensor = GETensorHelper::ConvertScalarToGETensor(t);
    auto const_op = std::make_shared<OpNode>("Const");
    OperatorSetAttrTensor(const_op->GeOp(), "value", ge_tensor);
    DestroyTensor(ge_tensor);

    std::vector<int64_t> dims({1});
    OperatorUpdateOutputDesc(
        const_op->GeOp(),
        "y",
        dims.data(),
        dims.size(),
        custom_kernel::experimental::ConvertToGEDtype(t.dtype()),
        ge::Format::FORMAT_ND);
    return std::move(const_op);
  }

  template <typename T>
  static std::shared_ptr<OpNode> ConvertVectorToConstOp(
      const std::vector<T> &t) {
    auto ge_tensor = GETensorHelper::ConvertVectorToGETensor<T>(t);
    auto const_op = std::make_shared<OpNode>("Const");
    OperatorSetAttrTensor(const_op->GeOp(), "value", ge_tensor);
    DestroyTensor(ge_tensor);

    std::vector<int64_t> dims({1});
    OperatorUpdateOutputDesc(
        const_op->GeOp(),
        "y",
        dims.data(),
        dims.size(),
        custom_kernel::experimental::ConvertToGEDtype(
            paddle::experimental::CppTypeToDataType<T>::Type()),
        ge::Format::FORMAT_ND);
    return std::move(const_op);
  }
};

class GraphCommand : public NpuCommand {
 public:
  explicit GraphCommand(const std::string &op_type)
      : NpuCommand(op_type), op_type_(op_type) {
    node_ = std::make_shared<OpNode>(op_type);
  }

  ~GraphCommand() override {}

  void AddInput(const phi::DenseTensor &tensor, TensorDescMaker maker) {
    auto ge_tensor = const_cast<TensorNode *>(
        reinterpret_cast<const TensorNode *>(tensor.data()));

    VLOG(4) << "Input dtype:" << maker.dtype_
            << ", rank: " << maker.dims_.size() << ", dims: " << maker.dims_
            << ", format: " << maker.layout_;
    VLOG(4) << "ptr=" << ge_tensor;

    node_->AddInput(ge_tensor, maker);
    ge_tensor->LinkTo(node_, in_index_);
    in_index_++;
  }

  void AddHostInput(const phi::DenseTensor &tensor, TensorDescMaker maker) {
    auto host_tensor = TensorNode::malloc();
    host_tensor->SetTag(custom_kernel::experimental::TensorNodeTag::IN);
    auto const_node = GraphCommandHelper::ConvertHostTensorToConstOp(tensor);

    const_node->AddOutput(host_tensor);
    host_tensor->FromOther(const_node, 0);

    VLOG(4) << "HostInput dtype:" << maker.dtype_
            << ", rank: " << maker.dims_.size() << ", dims: " << maker.dims_
            << ", format: " << maker.layout_;
    VLOG(4) << "ptr=" << host_tensor;

    host_tensor->LinkTo(node_, in_index_);
    node_->AddInput(host_tensor, maker);
    in_index_++;
  }

  void AddScalarInput(const phi::DenseTensor &tensor, TensorDescMaker maker) {
    auto scalar = TensorNode::malloc();
    scalar->SetTag(custom_kernel::experimental::TensorNodeTag::IN);
    auto const_node = GraphCommandHelper::ConvertHostTensorToConstOp(
        tensor, {}, tensor.dtype(), phi::DataLayout::ANY);

    const_node->AddInput(scalar);
    scalar->FromOther(const_node, 0);

    VLOG(4) << "ScalarInput dtype:" << maker.dtype_
            << ", rank: " << maker.dims_.size() << ", dims: " << maker.dims_
            << ", format: " << maker.layout_;
    VLOG(4) << "ptr=" << scalar;

    scalar->LinkTo(node_, in_index_);
    node_->AddInput(scalar, maker);
    in_index_++;
  }

  void AddHostScalarInput(const phi::DenseTensor &tensor,
                          TensorDescMaker maker) {
    auto host_scalar = TensorNode::malloc();
    host_scalar->SetTag(custom_kernel::experimental::TensorNodeTag::IN);
    auto const_node = GraphCommandHelper::ConvertHostTensorToConstOp(
        tensor, {}, tensor.dtype(), phi::DataLayout::ANY);

    const_node->AddOutput(host_scalar);
    host_scalar->FromOther(const_node, 0);

    VLOG(4) << "HostScalarInput dtype:" << maker.dtype_
            << ", rank: " << maker.dims_.size() << ", dims: " << maker.dims_
            << ", format: " << maker.layout_;
    VLOG(4) << "ptr=" << host_scalar;

    host_scalar->LinkTo(node_, in_index_);
    node_->AddInput(host_scalar, maker);
    in_index_++;
  }

  void AddOutput(phi::DenseTensor &tensor, TensorDescMaker maker) {
    auto ge_tensor = reinterpret_cast<TensorNode *>(tensor.data());
    if (WithoutInput()) {
      ge_tensor->SetTag(TensorNodeTag::IN);
    }

    VLOG(4) << "Output dtype:" << maker.dtype_
            << ", rank: " << maker.dims_.size() << ", dims: " << maker.dims_
            << ", format: " << maker.layout_;
    VLOG(4) << "ptr=" << ge_tensor;

    node_->AddOutput(ge_tensor, maker);
    ge_tensor->FromOther(node_, out_index_++);
  }

  void AddInput() override {
    VLOG(4) << "Input dtype: None";
    in_index_++;
  }

  void AddInput(const phi::DenseTensor &tensor) override {
    AddInput(tensor, TensorDescMaker(""));
  }

  void AddInput(const phi::Scalar &scalar) override {
    auto scalar_node = TensorNode::malloc();
    scalar_node->SetTag(custom_kernel::experimental::TensorNodeTag::IN);
    auto const_node = GraphCommandHelper::ConvertScalarToConstOp(scalar);

    const_node->AddOutput(scalar_node);
    scalar_node->FromOther(const_node, 0);

    VLOG(4) << "ScalarInput dtype:" << scalar.dtype() << ", rank: " << 1
            << ", dims: " << 1 << ", format: " << ge::Format::FORMAT_ND;
    VLOG(4) << "ptr=" << scalar_node;

    scalar_node->LinkTo(node_, in_index_);
    node_->AddInput(scalar_node);
    in_index_++;
  }

  void AddHostInput(const phi::DenseTensor &tensor) override {
    AddHostInput(tensor, TensorDescMaker(""));
  }

  void AddScalarInput(const phi::DenseTensor &tensor) override {
    AddScalarInput(tensor, TensorDescMaker(""));
  }

  void AddHostScalarInput(const phi::DenseTensor &tensor) override {
    AddHostScalarInput(tensor, TensorDescMaker(""));
  }

  void AddOutput() override {
    VLOG(4) << "Output dtype: None";
    out_index_++;
  }

  void AddOutput(phi::DenseTensor &tensor) override {
    AddOutput(tensor, TensorDescMaker(""));
  }

  void AddAttribute(const std::string &name,
                    const NpuAttribute &attr) override {
    if (attr.type() == typeid(bool)) {
      OperatorSetAttrBool(node_->GeOp(), name.c_str(), paddle::get<bool>(attr));
    } else if (attr.type() == typeid(int)) {
      OperatorSetAttrInt32(
          node_->GeOp(), name.c_str(), paddle::get<int32_t>(attr));
    } else if (attr.type() == typeid(int64_t)) {
      OperatorSetAttrInt64(
          node_->GeOp(), name.c_str(), paddle::get<int64_t>(attr));
    } else if (attr.type() == typeid(float)) {
      OperatorSetAttrFloat(
          node_->GeOp(), name.c_str(), paddle::get<float>(attr));
    } else if (attr.type() == typeid(std::vector<bool>)) {
      auto a = paddle::get<std::vector<bool>>(attr);
      std::vector<uint8_t> cast_a;
      for (auto it : a) {
        cast_a.push_back(static_cast<uint8_t>(it));
      }
      OperatorSetAttrBoolList(
          node_->GeOp(), name.c_str(), cast_a.data(), cast_a.size());
    } else if (attr.type() == typeid(std::vector<int32_t>)) {
      auto a = paddle::get<std::vector<int32_t>>(attr);
      OperatorSetAttrInt32List(node_->GeOp(), name.c_str(), a.data(), a.size());
    } else if (attr.type() == typeid(std::vector<int64_t>)) {
      auto a = paddle::get<std::vector<int64_t>>(attr);
      OperatorSetAttrInt64List(node_->GeOp(), name.c_str(), a.data(), a.size());
    } else if (attr.type() == typeid(std::vector<float>)) {
      auto a = paddle::get<std::vector<float>>(attr);
      OperatorSetAttrFloatList(node_->GeOp(), name.c_str(), a.data(), a.size());
    } else if (attr.type() == typeid(std::string)) {
      auto a = paddle::get<std::string>(attr);
      OperatorSetAttrString(node_->GeOp(), name.c_str(), a.data());
    } else if (attr.type() == typeid(std::vector<std::string>)) {
      auto a = paddle::get<std::vector<std::string>>(attr);
      std::vector<const char *> s;
      for (auto &it : a) {
        s.push_back(it.data());
      }
      OperatorSetAttrStringList(
          node_->GeOp(), name.c_str(), s.data(), s.size());
    } else if (attr.type() == typeid(phi::DenseTensor)) {
      auto &a = paddle::get<phi::DenseTensor>(attr);
      auto cast_a = GETensorHelper::ConvertDenseTensorToGETensor(a);
      OperatorSetAttrTensor(node_->GeOp(), name.c_str(), cast_a);
      DestroyTensor(cast_a);
    } else {
      PADDLE_THROW(phi::errors::Unimplemented(
          "Can not convert attribubte '%s' to convert to OperatorAttr", name));
    }
  }

  void Run(const phi::CustomContext &dev_ctx) override {
    VLOG(10) << "op_type: " << op_type_;
  }

 private:
  bool WithoutInput() const { return in_index_ == 0; }

 private:
  std::shared_ptr<OpNode> node_{nullptr};
  size_t in_index_{0};
  size_t out_index_{0};
  std::unordered_map<std::string, NpuAttribute> attrs_;

  std::vector<std::tuple<size_t, phi::Scalar *>> scalar_inputs_;
  std::string op_type_;
};

}  // namespace experimental
}  // namespace custom_kernel
