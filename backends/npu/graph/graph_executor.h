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

#include "graph/graph_funcs.h"
#include "graph/graph_utils.h"
#include "graph/paddle_graph.h"

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

namespace custom_graph {

template <typename T = int>
inline void UpdatePaddingAndDilation(std::vector<T>* paddings,
                                     std::vector<T>* dilation,
                                     const std::string padding_algorithm,
                                     const std::vector<T> data_dims,
                                     const std::vector<T>& strides,
                                     const std::vector<T>& ksize) {
  // set padding size == data_dims.size() * 2
  auto data_shape = data_dims;
  if (static_cast<int>(paddings->size()) == data_dims.size()) {
    for (int i = 0; i < data_dims.size(); ++i) {
      T copy_pad = *(paddings->begin() + 2 * i);
      paddings->insert(paddings->begin() + 2 * i + 1, copy_pad);
    }
  } else {
    if (data_dims.size() * 2 != paddings->size()) {
      // error
    }
  }

  // when padding_algorithm is "VALID" or "SAME"
  if (padding_algorithm == "SAME") {
    for (int i = 0; i < data_dims.size(); ++i) {
      T out_size = (data_dims[i] + strides[i] - 1) / strides[i];
      T pad_sum =
          std::max((out_size - 1) * strides[i] + ksize[i] - data_shape[i],
                   static_cast<T>(0));
      T pad_0 = pad_sum / 2;
      T pad_1 = pad_sum - pad_0;
      *(paddings->begin() + i * 2) = pad_0;
      *(paddings->begin() + i * 2 + 1) = pad_1;

      // dilation
      *(dilation->begin() + i) = 1;
    }

  } else if (padding_algorithm == "VALID") {
    for (auto it = paddings->begin(); it != paddings->end(); it++) {
      *it = 0;
    }
  }
}

template <typename T = int>
inline void UpdatePadding(std::vector<T>* paddings,
                          const bool global_pooling,
                          const bool adaptive,
                          const std::string padding_algorithm,
                          const std::vector<T> data_dims,
                          const std::vector<T>& strides,
                          const std::vector<T>& kernel_size) {
  // set padding size == data_dims.size() * 2
  auto data_shape = data_dims;
  if (static_cast<int>(paddings->size()) == data_dims.size()) {
    for (int i = 0; i < data_dims.size(); ++i) {
      T copy_pad = *(paddings->begin() + 2 * i);
      paddings->insert(paddings->begin() + 2 * i + 1, copy_pad);
    }
  } else {
    // PADDLE_ENFORCE_EQ(data_dims.size() * 2,
    //                   paddings->size(),
    //                   errors::InvalidArgument(
    //                       "Paddings size %d should be the same or twice as
    //                       the " "pooling size %d.", paddings->size(),
    //                       data_dims.size() * 2));
  }

  // when padding_algorithm is "VALID" or "SAME"
  if (padding_algorithm == "SAME") {
    for (int i = 0; i < data_dims.size(); ++i) {
      T out_size = (data_dims[i] + strides[i] - 1) / strides[i];
      T pad_sum =
          std::max((out_size - 1) * strides[i] + kernel_size[i] - data_shape[i],
                   static_cast<T>(0));
      T pad_0 = pad_sum / 2;
      T pad_1 = pad_sum - pad_0;
      *(paddings->begin() + i * 2) = pad_0;
      *(paddings->begin() + i * 2 + 1) = pad_1;
    }
  } else if (padding_algorithm == "VALID") {
    for (auto it = paddings->begin(); it != paddings->end(); it++) {
      *it = 0;
    }
  }

  // if global_pooling == true or adaptive == true, padding will be ignore
  if (global_pooling || adaptive) {
    for (auto it = paddings->begin(); it != paddings->end(); it++) {
      *it = 0;
    }
  }
}

template <typename T>
inline std::vector<T> slice_ddim(const std::vector<T>& dim,
                                 int begin,
                                 int end) {
  return std::vector<T>(dim.cbegin() + begin, dim.cbegin() + end);
}

inline std::string GEGradVarName(const std::string var_name) {
  if (var_name.find("@GRAD") != std::string::npos) {
    return var_name.substr(0, var_name.find("@GRAD")) + "_grad";
  } else {
    return var_name + "_grad";
  }
}

class Tensor;
class Operator;

class Graph {
 public:
  static Graph*& global_graph() {
    static Graph* graph;
    return graph;
  }

  static void set_global_graph(Graph* graph) { Graph::global_graph() = graph; }

  static Graph* get_global_graph() { return Graph::global_graph(); }

 public:
  Graph(const std::string& name, size_t id, C_Graph c_graph)
      : name_(name),
        id_(id),
        ge_graph_(std::make_shared<ge::Graph>(name_.c_str())),
        ir_graph_(std::make_shared<paddle::framework::ir::IRGraph>(c_graph)) {}

  ~Graph() {}

  const std::string& GetName() const { return name_; }

  size_t GetId() const { return id_; }

  std::shared_ptr<ge::Graph> ge_graph() { return ge_graph_; }

  std::shared_ptr<paddle::framework::ir::IRGraph> ir_graph() {
    return ir_graph_;
  }

  void AddFeedInput(int index, Tensor* tensor) {
    if (feed_inputs_.size() < index + 1) {
      feed_inputs_.resize(index + 1);
    }
    feed_inputs_[index] = tensor;
  }

  void AddFetchOutput(int index, Tensor* tensor) {
    if (fetch_outputs_.size() < index + 1) {
      fetch_outputs_.resize(index + 1);
    }
    fetch_outputs_[index] = tensor;
  }

  void AddInput(std::shared_ptr<Operator> input) { inputs_.push_back(input); }

  void AddOutput(std::shared_ptr<Operator> output) {
    outputs_.push_back(output);
  }

  void AddTarget(std::shared_ptr<Operator> target) {
    targets_.push_back(target);
  }

  const std::vector<std::shared_ptr<Operator>>& GetInputs() const {
    return inputs_;
  }

  const std::vector<std::shared_ptr<Operator>>& GetOutputs() const {
    return outputs_;
  }

  const std::vector<std::shared_ptr<Operator>>& GetTargets() const {
    return targets_;
  }

  const std::vector<Tensor*>& GetFeedInputs() const { return feed_inputs_; }

  const std::vector<Tensor*>& GetFetchOutputs() const { return fetch_outputs_; }

 private:
  std::string name_;
  size_t id_;

  std::shared_ptr<ge::Graph> ge_graph_;
  std::shared_ptr<paddle::framework::ir::IRGraph> ir_graph_;
  std::vector<std::shared_ptr<Operator>> inputs_;
  std::vector<std::shared_ptr<Operator>> outputs_;
  std::vector<std::shared_ptr<Operator>> targets_;
  std::vector<Tensor*> feed_inputs_;
  std::vector<Tensor*> fetch_outputs_;
};

class Tensor {
 public:
  enum class Flag : size_t {
    FEED = 0x1,
    FETCH = 0x2,
    PERSISTABLE = 0x4,
    GRAD = 0x8,
    CONST = 0x10,
    TARGET = 0x20,
  };

  static std::unordered_map<std::string, Tensor>& TensorStorage() {
    static std::unordered_map<std::string, Tensor> tensor_map;
    return tensor_map;
  }

  static Tensor& Get(const std::string& name) {
    if (TensorStorage().find(name) == TensorStorage().end()) {
      TensorStorage().insert({name, Tensor(name)});
    }
    return TensorStorage()[name];
  }

  Tensor() {
    static size_t tensor_index = 0;
    name_ = "tensor_" + std::to_string(tensor_index++);
  }

  Tensor(const std::string& name) : name_(name) {}

  ~Tensor() {}

  const std::string& Name() { return name_; }

  const std::pair<std::shared_ptr<Operator>, size_t>& From() const {
    return op_;
  }

  Tensor& MarkAsFeedInput(int col) {
    flags_ |= static_cast<size_t>(Flag::FEED);

    if (op_.first.get()) {
      Graph::get_global_graph()->AddFeedInput(col, this);
      Graph::get_global_graph()->AddInput(op());
    }
    return *this;
  }

  Tensor& MarkAsFetchOutput(int col) {
    flags_ |= static_cast<size_t>(Flag::FETCH);

    if (op_.first.get()) {
      Graph::get_global_graph()->AddFetchOutput(col, this);
      Graph::get_global_graph()->AddOutput(op());
    }
    return *this;
  }

  Tensor& MarkAsConst() {
    flags_ |= static_cast<size_t>(Flag::CONST);

    if (op_.first.get()) {
      Graph::get_global_graph()->AddInput(op());
    }
    return *this;
  }

  Tensor& MarkAsTarget() {
    flags_ |= static_cast<size_t>(Flag::TARGET);
    if (op_.first.get()) {
      Graph::get_global_graph()->AddTarget(op());
      return *this;
    }
  }

  Tensor& SetFrom(std::pair<std::shared_ptr<Operator>, size_t> op) {
    op_ = op;
    return *this;
  }

  Tensor& SetShape(const std::vector<int>& dims) {
    dims_ = dims;
    return *this;
  }

  Tensor& SetFormat(const std::string& format) {
    format_ = format;
    return *this;
  }

  template <typename T>
  Tensor& SetDType() {
    dtype_ = graph::utils::cpp_type_to_pd_dtype<T>::value;
  }

  Tensor& SetDType(paddle::framework::proto::VarType::Type dtype) {
    dtype_ = dtype;
  }

  const std::vector<int>& Shape() const { return dims_; }

  const std::string& Format() const { return format_; }

  paddle::framework::proto::VarType::Type DType() const { return dtype_; }

  int Numel() const {
    return std::accumulate(
        dims_.cbegin(), dims_.cend(), 1, std::multiplies<int>());
  }

  size_t flags() const { return flags_; }

  std::shared_ptr<Operator> op() const { return op_.first; }

 private:
  std::string name_;

  std::pair<std::shared_ptr<Operator>, size_t> op_{nullptr, 0};
  std::vector<int> dims_{};
  std::string format_{};
  paddle::framework::proto::VarType::Type dtype_ =
      paddle::framework::proto::VarType::FP32;
  size_t flags_ = 0;
};

class Operator {
 public:
  Operator() = default;

  Operator(const std::string& op_type, const std::string& name) {
    static size_t op_index = 0;
    std::string op_name = name;
    if (op_name.size() == 0) {
      op_name = op_type + std::to_string(op_index++);
    }
    op = std::make_shared<ge::Operator>(
        ge::OperatorFactory::CreateOperator(op_name.c_str(), op_type.c_str()));
  }

  ~Operator() {}

  std::shared_ptr<ge::Operator> GetGEOp() const { return op; }

  void SetGEOp(std::shared_ptr<ge::Operator> other) { op = other; }

  void AddInput(Tensor* input) { inputs_.push_back(input); }

  void AddOutput(Tensor* output) { outputs_.push_back(output); }

  void AddDepend(Operator* other) { op->AddControlInput(*other->GetGEOp()); }

  void AddDepend(std::shared_ptr<Operator> other) {
    op->AddControlInput(*other->GetGEOp());
  }

  const std::vector<Tensor*>& Inputs() const { return inputs_; }

  const std::vector<Tensor*>& Outputs() const { return outputs_; }

 private:
  std::vector<Tensor*> inputs_;
  std::vector<Tensor*> outputs_;
  std::shared_ptr<ge::Operator> op;
};

class OpCommand {
 public:
  template <typename T>
  static void FillConstant(Tensor& tensor,
                           const std::vector<int>& dim,
                           const std::vector<T>& value,
                           ge::Format format = ge::Format::FORMAT_NCHW) {
    ge::TensorDesc desc(ge::Shape(std::vector<int64_t>(dim.begin(), dim.end())),
                        format,
                        graph::utils::cpp_type_to_ge_dtype<T>::value());
    desc.SetRealDimCnt(desc.GetShape().GetDimNum());
    ge::Tensor value_tensor(
        desc,
        reinterpret_cast<uint8_t*>(const_cast<T*>(value.data())),
        value.size() * sizeof(T));
    OpCommand("Const").Output(tensor).Attr("value", value_tensor);
    tensor.MarkAsConst();
  }

  template <typename T>
  static void FillConstant(Tensor& tensor,
                           const std::vector<int>& dim,
                           T* value,
                           ge::Format format = ge::Format::FORMAT_NCHW) {
    ge::TensorDesc desc(ge::Shape(std::vector<int64_t>(dim.begin(), dim.end())),
                        format,
                        graph::utils::cpp_type_to_ge_dtype<T>::value());
    desc.SetRealDimCnt(desc.GetShape().GetDimNum());
    ge::Tensor value_tensor(
        desc,
        reinterpret_cast<uint8_t*>(value),
        std::accumulate(dim.begin(), dim.end(), 1, std::multiplies<int>()) *
            sizeof(T));
    OpCommand("Const").Output(tensor).Attr("value", value_tensor);
    tensor.MarkAsConst();
  }

  template <typename T>
  static void Cast(Tensor& in, Tensor& out) {
    OpCommand("Cast").Input(in).Output(out).Attr(
        "dst_type",
        static_cast<int>(graph::utils::cpp_type_to_ge_dtype<T>::value()));
  }

  static void Cast(Tensor& in,
                   Tensor& out,
                   paddle::framework::proto::VarType::Type dtype) {
    OpCommand("Cast").Input(in).Output(out).Attr(
        "dst_type",
        static_cast<int>(graph::utils::pd_dtype_to_ge_dtype(dtype)));
  }

  static void Reshape(Tensor& in,
                      Tensor& out,
                      const std::vector<int32_t>& dims) {
    Tensor shape;
    OpCommand::FillConstant(shape, {dims.size()}, dims);
    OpCommand("Reshape").Input(in).Input(shape).Output(out);
  }

  static void BroadcastTo(Tensor& in,
                          Tensor& out,
                          const std::vector<int32_t>& dims) {
    Tensor shape;
    OpCommand::FillConstant(shape, {dims.size()}, dims);
    OpCommand("BroadcastTo").Input(in).Input(shape).Output(out);
  }

 public:
  OpCommand(const std::string& op_type, const std::string& op_name = "")
      : op(std::make_shared<Operator>(op_type, op_name)) {}

  ~OpCommand() {}

  OpCommand& Input() {
    op->AddInput(nullptr);
    return *this;
  }

  OpCommand& Output() {
    op->AddOutput(nullptr);
    return *this;
  }

  OpCommand& Input(Tensor& input) {
    int32_t in_index = op->Inputs().size();
    op->AddInput(&input);
    op->GetGEOp()->SetInput(in_index,
                            *op->Inputs().back()->From().first->GetGEOp(),
                            op->Inputs().back()->From().second);
    return *this;
  }

  OpCommand& Input(Tensor& input, const std::string& desc_name) {
    Input(input);
    if (input.Shape().size() > 0) {
      graph::funcs::update_input_shape(
          *op->GetGEOp(), desc_name, input.Shape());
    }
    if (input.Format().size() > 0) {
      graph::funcs::update_input_format(
          *op->GetGEOp(), desc_name, input.Format());
    }
    graph::funcs::update_input_dtype(*op->GetGEOp(), desc_name, input.DType());
    return *this;
  }

  OpCommand& Output(Tensor& output) {
    int32_t out_index = op->Outputs().size();
    op->AddOutput(&output);
    output.SetFrom({op, out_index});
    return *this;
  }

  OpCommand& Output(Tensor& output, const std::string& desc_name) {
    Output(output);
    if (output.Shape().size() > 0) {
      graph::funcs::update_output_shape(
          *op->GetGEOp(), desc_name, output.Shape());
    }
    if (output.Format().size() > 0) {
      graph::funcs::update_output_format(
          *op->GetGEOp(), desc_name, output.Format());
    }
    graph::funcs::update_output_dtype(
        *op->GetGEOp(), desc_name, output.DType());
    return *this;
  }

  template <typename T>
  OpCommand& Attr(const std::string& attr_name, const T& attr_value) {
    op->GetGEOp()->SetAttr(attr_name.c_str(), attr_value);
    return *this;
  }

  OpCommand& Depend(Tensor& tensor) {
    op->AddDepend(tensor.op());
    return *this;
  }

 private:
  std::shared_ptr<Operator> op;
};

class OpCommandPipe {
 public:
  OpCommandPipe() = default;
  OpCommandPipe(const std::vector<std::string>& op_pipe) {
    for (auto& op_type : op_pipe) {
      Op(op_type);
    }
  }

  OpCommandPipe& Op(const std::string& op_type) {
    op_pipe_.emplace_back(new OpCommand(op_type));
    return *this;
  }

  OpCommandPipe& Cast() { return *this; }

  OpCommandPipe& Reshape(const std::vector<int>& dims) { return *this; }

  OpCommandPipe& BroadcastTo(const std::vector<int>& dims) { return *this; }

  template <typename T>
  OpCommandPipe& Attr(const std::string& attr_name, const T& attr_value) {
    op_pipe_.back()->Attr(attr_name, attr_value);
    return *this;
  }

  OpCommandPipe& Input(Tensor& input) {
    inputs_.push_back(&input);
    return *this;
  }

  OpCommandPipe& Output(Tensor& output) {
    outputs_.push_back(&output);
    return *this;
  }

  void End() {
    std::shared_ptr<OpCommand> prev_op = op_pipe_[0];
    std::shared_ptr<OpCommand> cur_op = prev_op;

    for (auto& in : inputs_) {
      prev_op->Input(*in);
    }

    for (auto i = 1; i < op_pipe_.size(); ++i) {
      Tensor tmp;
      cur_op = op_pipe_[i];
      prev_op->Output(tmp);
      cur_op->Input(tmp);
      prev_op = cur_op;
    }

    for (auto& out : outputs_) {
      cur_op->Output(*out);
    }
  }

 private:
  std::vector<Tensor*> inputs_;
  std::vector<Tensor*> outputs_;
  std::vector<std::shared_ptr<OpCommand>> op_pipe_;
};

class Context {
 public:
  Context(paddle::framework::ir::OpNode* op_node) : op_node_(op_node) {}

  bool HasInput(const std::string& name) const {
    return !!op_node_->Input(name);
  }

  bool HasOutput(const std::string& name) const {
    return !!op_node_->Output(name);
  }

  Tensor& Input(const std::string& name) const {
    auto* var_node = op_node_->Input(name);
    Tensor& tensor = Tensor::Get(var_node->Name());
    tensor.SetShape(var_node->dims());
    tensor.SetDType(var_node->dtype());
    return tensor;
  }

  Tensor& Output(const std::string& name) const {
    auto* var_node = op_node_->Output(name);
    Tensor& tensor = Tensor::Get(var_node->Name());
    tensor.SetShape(var_node->dims());
    tensor.SetDType(var_node->dtype());
    return tensor;
  }

  std::vector<Tensor*> MultiInput(const std::string& name) const {
    std::vector<Tensor*> ret;
    for (auto& var_node : op_node_->MultiInput(name)) {
      auto& tensor = Tensor::Get(var_node->Name());
      tensor.SetShape(var_node->dims());
      tensor.SetDType(var_node->dtype());
      ret.push_back(&tensor);
    }
    return ret;
  }

  std::vector<Tensor*> MultiOutput(const std::string& name) const {
    std::vector<Tensor*> ret;
    for (auto& var_node : op_node_->MultiOutput(name)) {
      auto& tensor = Tensor::Get(var_node->Name());
      tensor.SetShape(var_node->dims());
      tensor.SetDType(var_node->dtype());
      ret.push_back(&tensor);
    }
    return ret;
  }

  template <typename T>
  T Attr(const std::string& name) const {
    return op_node_->Attr<T>(name);
  }

 private:
  paddle::framework::ir::OpNode* op_node_;
};

class OpAdapter;

using adapter_creator_t = std::function<std::shared_ptr<OpAdapter>()>;

class OpAdapter {
 public:
  static std::unordered_map<std::string, adapter_creator_t>& Factory() {
    static std::unordered_map<std::string, adapter_creator_t> factory;
    return factory;
  }

  OpAdapter() = default;

  OpAdapter& self() { return *this; }

  virtual ~OpAdapter() {}

  virtual void run(const custom_graph::Context& ctx) = 0;
};

template <typename AdapterT>
class Registrar {
 public:
  explicit Registrar(const std::string& ir) {
    adapter_creator_t adapter_creator = []() -> std::shared_ptr<OpAdapter> {
      return std::make_shared<AdapterT>();
    };
    OpAdapter::Factory()[ir] = std::move(adapter_creator);
  }

  int Touch() { return 0; }
};

#define REG_OP_ADAPTER(ir, adapter)                                          \
  static ::custom_graph::Registrar<adapter> __op_adapter_registrar_##ir##__( \
      #ir);                                                                  \
  int __op_adapter_registrar_##ir##__touch__() {                             \
    return __op_adapter_registrar_##ir##__.Touch();                          \
  }

#define USE_OP_ADAPTER(ir)                                              \
  extern int __op_adapter_registrar_##ir##__touch__();                  \
  static __attribute__((unused)) int __use_op_adapter_##ir##__touch__ = \
      __op_adapter_registrar_##ir##__touch__()

}  // namespace custom_graph
