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

#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "framework.pb.h"
#include "paddle/phi/backends/device_ext.h"
#include "paddle/utils/any.h"

namespace paddle {
namespace framework {

inline std::string GradVarName(const std::string& var_name) {
  return var_name + "@GRAD";
}

namespace ir {

class VarNode;
class OpNode;
class IRGraph;

template <typename T>
inline std::string to_string(const std::vector<T>& vec) {
  std::stringstream ss;
  ss << "[";
  for (int i = 0; i < vec.size(); ++i) {
    ss << vec[i];
    if (i < vec.size() - 1) {
      ss << ", ";
    }
  }
  ss << ']';
  return ss.str();
}

std::string ProtoAttrTypeToString(paddle::framework::proto::AttrType attr_type);

std::string ProtoVarTypeToString(
    paddle::framework::proto::VarType_Type var_type);

int GetTypeSize(const std::string& type_string);

class Node {
 public:
  Node(const std::string& name, const std::string& type)
      : name_(name), type_(type) {}

  const std::string& Name() const { return name_; }

  const std::string& Type() const { return type_; }

  virtual bool IsInputNode() const = 0;

  virtual bool IsOutputNode() const = 0;

  bool IsLeafNode() const { return IsInputNode() || IsOutputNode(); }

  virtual std::string to_string() const = 0;

 private:
  Node();
  Node(const Node&);
  Node(Node&&);
  Node& operator=(const Node&);
  Node& operator=(Node&&);

  std::string name_;
  std::string type_;
};

class OpNode : public Node {
 public:
  std::string to_string() const override;

  explicit OpNode(paddle::framework::proto::OpDesc* op_desc);

  bool IsInputNode() const override;

  bool IsOutputNode() const override;

  bool IsTarget() const { return is_target_; }

  std::vector<std::string> InputsName() const {
    std::vector<std::string> ret;
    for (const auto& input : inputs_) {
      ret.push_back(input.first);
    }
    return ret;
  }

  std::vector<std::string> OutputsName() const {
    std::vector<std::string> ret;
    for (const auto& output : outputs_) {
      ret.push_back(output.first);
    }
    return ret;
  }

  const std::unordered_map<std::string, std::vector<VarNode*>>& Inputs() const {
    return inputs_;
  }

  const std::unordered_map<std::string, std::vector<VarNode*>>& Outputs()
      const {
    return outputs_;
  }

  bool HasInput(const std::string& input) const {
    return inputs_.find(input) != inputs_.cend() &&
           inputs_.at(input).size() > 0;
  }

  bool HasOutput(const std::string& output) const {
    return outputs_.find(output) != outputs_.cend() &&
           outputs_.at(output).size() > 0;
  }

  bool HasAttribute(const std::string& attr) const {
    return attributes_.find(attr) != attributes_.cend();
  }

  std::vector<VarNode*> MultiInput(const std::string& input) const {
    if (HasInput(input)) {
      return inputs_.at(input);
    }
    return std::vector<VarNode*>();
  }

  std::vector<VarNode*> MultiOutput(const std::string& output) const {
    if (HasOutput(output)) {
      return outputs_.at(output);
    }
    return std::vector<VarNode*>();
  }

  VarNode* Input(const std::string& input) const {
    if (HasInput(input)) {
      return inputs_.at(input).at(0);
    }
    return nullptr;
  }

  VarNode* Output(const std::string& output) const {
    if (HasOutput(output)) {
      return outputs_.at(output).at(0);
    }
    return nullptr;
  }

  template <typename T>
  T Attr(const std::string attr_name) const {
    return paddle::any_cast<T>(attributes_.at(attr_name));
  }

  std::vector<std::string> Attrs() const {
    std::vector<std::string> ret;
    for (auto& attr : attributes_) {
      ret.push_back(attr.first);
    }
    return ret;
  }

 private:
  friend IRGraph;

  OpNode();
  OpNode(const OpNode&);
  OpNode(OpNode&&);
  OpNode& operator=(const OpNode&);
  OpNode& operator=(OpNode&&);

  void build_from_op_desc(paddle::framework::proto::OpDesc* op_desc);
  paddle::any build_attribute_from_op_desc_attr(
      paddle::framework::proto::OpDesc_Attr* attr);
  void add_input_node(const std::string& parameter, VarNode* node) {
    inputs_[parameter].emplace_back(node);
  }
  void add_output_node(const std::string& parameter, VarNode* node) {
    outputs_[parameter].emplace_back(node);
  }
  void add_input_nodes(const std::string& parameter,
                       std::vector<VarNode*> nodes) {
    inputs_[parameter].insert(
        inputs_[parameter].end(), nodes.begin(), nodes.end());
  }
  void add_output_nodes(const std::string& parameter,
                        std::vector<VarNode*> nodes) {
    outputs_[parameter].insert(
        outputs_[parameter].end(), nodes.begin(), nodes.end());
  }

 private:
  std::unordered_map<std::string, std::vector<VarNode*>> inputs_;
  std::unordered_map<std::string, std::vector<VarNode*>> outputs_;
  std::unordered_map<std::string, paddle::any> attributes_;
  bool is_target_;
};

class VarNode : public Node {
 public:
  explicit VarNode(paddle::framework::proto::VarDesc* var_desc);

  bool IsInputNode() const override { return inputs_.size() == 0; }

  bool IsOutputNode() const override { return outputs_.size() == 0; }

  bool Persistable() const { return persistable_; }

  bool IsParameter() const { return is_parameter_; }

  bool StopGradient() const { return stop_gradient_; }

  void SetPersistable(bool value) { persistable_ = value; }

  void SetIsParameter(bool value) { is_parameter_ = value; }

  void SetStopGradient(bool value) { stop_gradient_ = value; }

  std::vector<OpNode*> Inputs() { return inputs_; }

  std::vector<OpNode*> Outputs() { return outputs_; }

  std::string dtype_str() const;

  paddle::framework::proto::VarType::Type dtype() const { return data_type_; }

  const std::vector<int>& dims() const;

  int numel() const;

  std::string to_string() const override;

  template <typename T>
  T Attr(const std::string attr_name) const {
    return paddle::any_cast<T>(attributes_.at(attr_name));
  }

  std::vector<std::string> Attrs() const {
    std::vector<std::string> ret;
    for (auto& attr : attributes_) {
      ret.push_back(attr.first);
    }
    return ret;
  }

 private:
  friend IRGraph;

  VarNode();
  VarNode(const VarNode&);
  VarNode(VarNode&&);
  VarNode& operator=(const VarNode&);
  VarNode& operator=(VarNode&&);

  void build_from_var_desc(paddle::framework::proto::VarDesc* var_desc);
  paddle::any build_attribute_from_var_desc_attr(
      paddle::framework::proto::VarDesc_Attr* attr);
  void add_input_node(OpNode* node) { inputs_.emplace_back(node); }
  void add_output_node(OpNode* node) { outputs_.emplace_back(node); }
  void add_input_nodes(std::vector<OpNode*> nodes) {
    inputs_.insert(inputs_.end(), nodes.begin(), nodes.end());
  }
  void add_output_nodes(std::vector<OpNode*> nodes) {
    outputs_.insert(outputs_.end(), nodes.begin(), nodes.end());
  }

 private:
  bool persistable_;
  bool is_parameter_;
  bool stop_gradient_;

  std::vector<OpNode*> inputs_;
  std::vector<OpNode*> outputs_;
  std::unordered_map<std::string, paddle::any> attributes_;

  paddle::framework::proto::VarType::Type var_type_;
  paddle::framework::proto::VarType::Type data_type_;
  std::vector<int> dims_;
};

class IRGraph {
 public:
  explicit IRGraph(C_Graph graph);

  const std::vector<std::unique_ptr<VarNode>>& Vars() const {
    return var_nodes_;
  }

  const std::vector<std::unique_ptr<OpNode>>& Ops() const { return op_nodes_; }

  VarNode* Var(const std::string& var) {
    for (auto& node : var_nodes_) {
      if (node->Name() == var) {
        return node.get();
      }
    }
    return nullptr;
  }

  OpNode* Op(const std::string& op) {
    for (auto& node : op_nodes_) {
      if (node->Name() == op) {
        return node.get();
      }
    }
    return nullptr;
  }

  VarNode* GetVar(const std::string& var) {
    auto node = Var(var);
    if (!node) {
      std::cerr << "[ERROR] not found VarNode " << var << std::endl;
      exit(-1);
    }
    return node;
  }

  OpNode* GetOp(const std::string& op) {
    auto node = Op(op);
    if (!node) {
      std::cerr << "[ERROR] not found OpNode " << op << std::endl;
      exit(-1);
    }
    return node;
  }

 private:
  void build_from_block_desc(paddle::framework::proto::BlockDesc* block);

  paddle::framework::proto::ProgramDesc* prog_;
  std::vector<std::unique_ptr<OpNode>> op_nodes_;
  std::vector<std::unique_ptr<VarNode>> var_nodes_;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
