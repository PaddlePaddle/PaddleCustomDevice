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

#include "graph/paddle_graph.h"

#include <sys/syscall.h>
#include <unistd.h>

#define gettid() syscall(SYS_gettid)

namespace paddle {
namespace framework {
namespace ir {

std::string GetUniqueOpName(const std::string& name) {
  static std::unordered_map<std::string, size_t> OpNameCount;
  std::stringstream ss;
  size_t id = OpNameCount[name]++;
  ss << name << "_" << std::to_string(id);
  return ss.str();
}

// std::string GetUniqueVarName(const std::string& name) {
//   static std::unordered_map<std::string, size_t> VarNameCount;
//   std::stringstream ss;
//   size_t id = VarNameCount[name]++;
//   ss << name << "_" << std::to_string(id);
//   return ss.str();
// }

std::string ProtoAttrTypeToString(
    paddle::framework::proto::AttrType attr_type) {
  if (attr_type == paddle::framework::proto::AttrType::INT) {
    return "int";
  } else if (attr_type == paddle::framework::proto::AttrType::FLOAT) {
    return "float";
  } else if (attr_type == paddle::framework::proto::AttrType::STRING) {
    return "string";
  } else if (attr_type == paddle::framework::proto::AttrType::INTS) {
    return "ints";
  } else if (attr_type == paddle::framework::proto::AttrType::FLOATS) {
    return "floats";
  } else if (attr_type == paddle::framework::proto::AttrType::STRINGS) {
    return "strings";
  } else if (attr_type == paddle::framework::proto::AttrType::BOOLEAN) {
    return "boolean";
  } else if (attr_type == paddle::framework::proto::AttrType::BOOLEANS) {
    return "booleas";
  } else if (attr_type == paddle::framework::proto::AttrType::BLOCK) {
    return "block";
  } else if (attr_type == paddle::framework::proto::AttrType::LONG) {
    return "int64_t";
  } else if (attr_type == paddle::framework::proto::AttrType::BLOCKS) {
    return "blocks";
  } else if (attr_type == paddle::framework::proto::AttrType::LONGS) {
    return "longs";
  } else if (attr_type == paddle::framework::proto::AttrType::FLOAT64S) {
    return "float64s";
  } else if (attr_type == paddle::framework::proto::AttrType::VAR) {
    return "var";
  } else if (attr_type == paddle::framework::proto::AttrType::VARS) {
    return "vars";
  } else if (attr_type == paddle::framework::proto::AttrType::FLOAT64) {
    return "float64";
  } else {
    std::cerr << "unsupported attr type " << attr_type << std::endl;
    exit(-1);
  }
}

std::string ProtoVarTypeToString(
    paddle::framework::proto::VarType::Type var_type) {
  if (var_type == paddle::framework::proto::VarType::BOOL) {
    return "bool";
  } else if (var_type == paddle::framework::proto::VarType::INT16) {
    return "int16";
  } else if (var_type == paddle::framework::proto::VarType::INT32) {
    return "int32";
  } else if (var_type == paddle::framework::proto::VarType::INT64) {
    return "int64";
  } else if (var_type == paddle::framework::proto::VarType::FP16) {
    return "float16";
  } else if (var_type == paddle::framework::proto::VarType::FP32) {
    return "float32";
  } else if (var_type == paddle::framework::proto::VarType::FP64) {
    return "float64";
  } else if (var_type == paddle::framework::proto::VarType::SIZE_T) {
    return "size_t";
  } else if (var_type == paddle::framework::proto::VarType::UINT8) {
    return "uint8";
  } else if (var_type == paddle::framework::proto::VarType::INT8) {
    return "int8";
  } else if (var_type == paddle::framework::proto::VarType::BF16) {
    return "bfloat16";
  } else if (var_type == paddle::framework::proto::VarType::COMPLEX64) {
    return "complex64";
  } else if (var_type == paddle::framework::proto::VarType::COMPLEX128) {
    return "complex128";
  } else if (var_type == paddle::framework::proto::VarType::LOD_TENSOR) {
    return "lod_tensor";
  } else if (var_type == paddle::framework::proto::VarType::SELECTED_ROWS) {
    return "selected_rows";
  } else if (var_type == paddle::framework::proto::VarType::FEED_MINIBATCH) {
    return "feed_minibatch";
  } else if (var_type == paddle::framework::proto::VarType::FETCH_LIST) {
    return "fetch_list";
  } else if (var_type == paddle::framework::proto::VarType::STEP_SCOPES) {
    return "step_scopes";
  } else if (var_type == paddle::framework::proto::VarType::LOD_RANK_TABLE) {
    return "lod_rank_table";
  } else if (var_type == paddle::framework::proto::VarType::LOD_TENSOR_ARRAY) {
    return "lod_tensor_array";
  } else if (var_type == paddle::framework::proto::VarType::PLACE_LIST) {
    return "place_list";
  } else if (var_type == paddle::framework::proto::VarType::READER) {
    return "reader";
  } else if (var_type == paddle::framework::proto::VarType::RAW) {
    return "raw";
  } else if (var_type == paddle::framework::proto::VarType::TUPLE) {
    return "tuple";
  } else if (var_type == paddle::framework::proto::VarType::STRING) {
    return "string";
  } else if (var_type == paddle::framework::proto::VarType::STRINGS) {
    return "strings";
  } else if (var_type == paddle::framework::proto::VarType::VOCAB) {
    return "vocab";
  } else if (var_type == paddle::framework::proto::VarType::FEED_LIST) {
    return "feed_list";
  } else if (var_type == paddle::framework::proto::VarType::FEED_MINIBATCH) {
    return "feed_minibatch";
  } else if (var_type == paddle::framework::proto::VarType::PSTRING) {
    return "pstring";
  } else {
    std::cerr << "ProtoVarTypeToString unknown var_type: " << var_type
              << std::endl;
    return "unknown";
  }
}

int GetTypeSize(const std::string& type_string) {
  if (type_string == "int8") {
    return sizeof(int8_t);
  } else if (type_string == "int16") {
    return sizeof(int16_t);
  } else if (type_string == "int32") {
    return sizeof(int32_t);
  } else if (type_string == "int64") {
    return sizeof(int64_t);
  } else if (type_string == "uint8") {
    return sizeof(uint8_t);
  } else if (type_string == "uint16") {
    return sizeof(uint16_t);
  } else if (type_string == "uint32") {
    return sizeof(uint32_t);
  } else if (type_string == "uint64") {
    return sizeof(uint64_t);
  } else if (type_string == "float32") {
    return sizeof(float);
  } else if (type_string == "float64") {
    return sizeof(double);
  } else if (type_string == "float16") {
    return sizeof(uint16_t);
  } else {
    std::cerr << "unsupported type string " << type_string << std::endl;
    exit(-1);
    return 0;
  }
}

OpNode::OpNode(paddle::framework::proto::OpDesc* op_desc, std::fstream& ofs)
    : Node(GetUniqueOpName(op_desc->type().c_str()), op_desc->type().c_str()) {
  ofs << "Build OpNode: " << this << ", type: " << Type()
      << ", name: " << Name() << " from op_desc " << op_desc << std::endl;
  build_from_op_desc(op_desc, ofs);
  ofs << "Build OpNode " << this << " finished\n";
}

std::string OpNode::to_string() const {
  std::stringstream ss;
  ss << "OpNode " << this << ", type: " << Type() << ", name: " << Name();
  ss << "\n\tinputs: " << inputs_.size();
  for (auto& input : inputs_) {
    ss << " " << input.first;
    ss << "[";
    for (auto i = 0; i < input.second.size(); ++i) {
      ss << input.second[i]->Name();
      if (i < input.second.size() - 1) {
        ss << ", ";
      }
    }
    ss << "]";
  }

  ss << "\n\toutputs: " << outputs_.size();
  for (auto& output : outputs_) {
    ss << " " << output.first;
    ss << "[";
    for (auto i = 0; i < output.second.size(); ++i) {
      ss << output.second[i]->Name();
      if (i < output.second.size() - 1) {
        ss << ", ";
      }
    }
    ss << "]";
  }

  ss << "\n\tattributes: " << attributes_.size();
  auto attrs = Attrs();
  for (auto i = 0; i < attrs.size(); ++i) {
    ss << " " << attrs[i];
    if (i != attrs.size() - 1) {
      ss << ",";
    }
  }
  return ss.str();
}

bool OpNode::IsInputNode() const {
  for (auto& input : inputs_) {
    for (auto& arg : input.second) {
      if (!arg->IsInputNode()) {
        return false;
      }
    }
  }
  return true;
}

bool OpNode::IsOutputNode() const {
  for (auto& output : outputs_) {
    for (auto& arg : output.second) {
      if (!arg->IsOutputNode()) {
        return false;
      }
    }
  }
  return true;
}

void OpNode::build_from_op_desc(paddle::framework::proto::OpDesc* op_desc,
                                std::fstream& ofs) {
  is_target_ = op_desc->has_is_target() && op_desc->is_target();

  auto& inputs = op_desc->inputs();
  for (auto j = 0; j < inputs.size(); ++j) {
    std::string parameter(inputs[j].parameter().c_str());
    inputs_[parameter].clear();
  }

  auto& outputs = op_desc->outputs();
  for (auto j = 0; j < outputs.size(); ++j) {
    std::string parameter(outputs[j].parameter().c_str());
    outputs_[parameter].clear();
  }

  auto& attrs = op_desc->attrs();
  for (auto& attr : attrs) {
    std::string attr_name(attr.name().c_str());
    attributes_[attr_name] = build_attribute_from_op_desc_attr(
        const_cast<paddle::framework::proto::OpDesc_Attr*>(&attr), ofs);
  }
}

paddle::any OpNode::build_attribute_from_op_desc_attr(
    paddle::framework::proto::OpDesc_Attr* attr, std::fstream& ofs) {
  if (attr->type() == paddle::framework::proto::AttrType::INT) {
    return paddle::any(static_cast<int>(attr->i()));
  } else if (attr->type() == paddle::framework::proto::AttrType::INTS) {
    std::vector<int> ints;
    for (auto& item : attr->ints()) {
      ints.emplace_back(item);
    }
    return paddle::any(ints);
  } else if (attr->type() == paddle::framework::proto::AttrType::LONG) {
    return paddle::any(static_cast<int64_t>(attr->l()));
  } else if (attr->type() == paddle::framework::proto::AttrType::LONGS) {
    std::vector<int64_t> longs;
    for (auto& item : attr->longs()) {
      longs.emplace_back(item);
    }
    return paddle::any(longs);
  } else if (attr->type() == paddle::framework::proto::AttrType::FLOAT) {
    return paddle::any(static_cast<float>(attr->f()));
  } else if (attr->type() == paddle::framework::proto::AttrType::FLOATS) {
    std::vector<float> floats;
    for (auto& item : attr->floats()) {
      floats.emplace_back(item);
    }
    return paddle::any(floats);
  } else if (attr->type() == paddle::framework::proto::AttrType::FLOAT64) {
    return paddle::any(static_cast<double>(attr->float64()));
  } else if (attr->type() == paddle::framework::proto::AttrType::FLOAT64S) {
    std::vector<double> float64s;
    for (auto& item : attr->float64s()) {
      float64s.emplace_back(item);
    }
    return paddle::any(float64s);
  } else if (attr->type() == paddle::framework::proto::AttrType::STRING) {
    return paddle::any(std::string(attr->s().c_str()));
  } else if (attr->type() == paddle::framework::proto::AttrType::STRINGS) {
    std::vector<std::string> strings;
    for (auto& item : attr->strings()) {
      strings.emplace_back(item.c_str());
    }
    return paddle::any(strings);
  } else if (attr->type() == paddle::framework::proto::AttrType::BOOLEAN) {
    return paddle::any(static_cast<bool>(attr->b()));
  } else if (attr->type() == paddle::framework::proto::AttrType::BOOLEANS) {
    std::vector<bool> bools;
    for (auto& item : attr->bools()) {
      bools.push_back(item);
    }
    return paddle::any(bools);
  } else {
    // error
  }
  return paddle::any();
}

/// VarNode

VarNode::VarNode(paddle::framework::proto::VarDesc* var_desc, std::fstream& ofs)
    : Node(var_desc->name().c_str(),
           ProtoVarTypeToString(var_desc->type().type())) {
  ofs << "Build VarNode: " << this << ", type: " << Type()
      << ", name: " << Name() << " from var_desc " << var_desc << std::endl;
  build_from_var_desc(var_desc, ofs);
  ofs << "Build VarNode " << this << " finished\n";
}

void VarNode::build_from_var_desc(paddle::framework::proto::VarDesc* var_desc,
                                  std::fstream& ofs) {
  persistable_ = (var_desc->has_persistable() && var_desc->persistable());
  is_parameter_ = (var_desc->has_is_parameter() && var_desc->is_parameter());
  stop_gradient_ = (var_desc->has_stop_gradient() && var_desc->stop_gradient());

  auto& var = var_desc->type();
  var_type_ = var.type();
  if (var_type_ == paddle::framework::proto::VarType::LOD_TENSOR) {
    data_type_ = var.lod_tensor().tensor().data_type();
    ofs << "datatype is " << data_type_ << std::endl;
    auto dims = var.lod_tensor().tensor().dims();
    for (auto& dim : dims) {
      dims_.emplace_back(dim);
    }
  } else if (var_type_ == paddle::framework::proto::VarType::FEED_MINIBATCH ||
             var_type_ == paddle::framework::proto::VarType::FETCH_LIST) {
  } else {
    ofs << "[ERROR] VarType " << Type() << " is not supported\n";
    exit(-1);
  }

  auto& attrs = var_desc->attrs();
  for (auto& attr : attrs) {
    std::string attr_name(attr.name().c_str());
    attributes_[attr_name] = build_attribute_from_var_desc_attr(
        const_cast<paddle::framework::proto::VarDesc_Attr*>(&attr), ofs);
  }
}

paddle::any VarNode::build_attribute_from_var_desc_attr(
    paddle::framework::proto::VarDesc_Attr* attr, std::fstream& ofs) {
  if (attr->type() == paddle::framework::proto::AttrType::INT) {
    return paddle::any(static_cast<int>(attr->i()));
  } else if (attr->type() == paddle::framework::proto::AttrType::INTS) {
    std::vector<int> ints;
    for (auto& item : attr->ints()) {
      ints.emplace_back(item);
    }
    return paddle::any(ints);
  } else if (attr->type() == paddle::framework::proto::AttrType::STRING) {
    return paddle::any(std::string(attr->s().c_str()));
  } else {
    // error
  }
  return paddle::any();
}

std::string VarNode::to_string() const {
  std::stringstream ss;
  ss << "VarNode " << this << " type: " << Type() << ", name: " << Name()
     << ", dims: " << paddle::framework::ir::to_string(dims_);
  ss << "\n\tIsInputNode " << IsInputNode() << "\n\tis_output_node "
     << IsOutputNode() << "\n\tis_leaf_node " << IsLeafNode()
     << "\n\tpersitable " << Persistable() << "\n\tis_parameter "
     << IsParameter() << "\n\tstop_gradient " << StopGradient();

  ss << "\n\tinputs: ";
  if (inputs_.size() > 0) {
    for (auto i = 0; i < inputs_.size(); ++i) {
      ss << inputs_[i]->Name();
      if (i < inputs_.size() - 1) {
        ss << ", ";
      }
    }
  } else {
    ss << "None";
  }

  ss << "\n\toutputs: ";
  if (outputs_.size()) {
    for (auto i = 0; i < outputs_.size(); ++i) {
      ss << outputs_[i]->Name();
      if (i < outputs_.size() - 1) {
        ss << ", ";
      }
    }
  } else {
    ss << "None";
  }

  ss << "\n\tattributes: " << attributes_.size();
  auto attrs = Attrs();
  for (auto i = 0; i < attrs.size(); ++i) {
    ss << " " << attrs[i];
    if (i != attrs.size() - 1) {
      ss << ",";
    }
  }
  return ss.str();
}

std::string VarNode::dtype_str() const {
  return ProtoVarTypeToString(data_type_);
}

const std::vector<int>& VarNode::dims() const {
  if (var_type_ != paddle::framework::proto::VarType::LOD_TENSOR) {
    // error
  }
  return dims_;
}

int VarNode::numel() const {
  return std::accumulate(
      dims_.cbegin(), dims_.cend(), 1, std::multiplies<int>());
}

/// Graph

IRGraph::IRGraph(C_Graph graph)
    : prog_(reinterpret_cast<paddle::framework::proto::ProgramDesc*>(graph)) {
  std::stringstream ss;
  ss << "pd_graph"
     << ".pid_" << static_cast<uint64_t>(getpid()) << ".tid_"
     << static_cast<uint64_t>(gettid()) << ".graph_" << std::hex
     << reinterpret_cast<uint64_t>(graph) << ".txt";

  std::fstream ofs(ss.str().c_str(), std::ios::out);

  ofs << "Build Graph: " << prog_ << " blocks_size is " << prog_->blocks_size()
      << " from prog_desc " << graph;
  auto block = prog_->blocks(0);
  build_from_block_desc(&block, ofs);
  ofs << "Graph " << this << " has Vars: " << var_nodes_.size()
      << ", Ops: " << op_nodes_.size() << std::endl;
  ofs << "Build Graph " << this << " finished\n";
}

void IRGraph::build_from_block_desc(paddle::framework::proto::BlockDesc* block,
                                    std::fstream& ofs) {
  auto ops = block->ops();
  auto vars = block->vars();

  for (auto i = 0; i < vars.size(); ++i) {
    std::string var_name(vars[i].name().c_str());
    var_nodes_.emplace_back(new VarNode(&vars[i], ofs));
  }

  for (auto i = 0; i < ops.size(); ++i) {
    std::string op_type(ops[i].type().c_str());
    if (op_type == "share_buffer") {
      // remove share_buffer op
      continue;
    }

    if (ops[i].inputs_size() == 0 && ops[i].outputs_size() == 0) {
      // remove unused op
      continue;
    }

    op_nodes_.emplace_back(new OpNode(&ops[i], ofs));

    // build link
    ofs << "Build the link of OpNode " << op_nodes_.back().get() << std::endl;
    auto& inputs = ops[i].inputs();
    for (auto j = 0; j < inputs.size(); ++j) {
      std::string parameter(inputs[j].parameter().c_str());
      for (auto k = 0; k < inputs[j].arguments_size(); ++k) {
        std::string argument(inputs[j].arguments(k).c_str());
        ofs << "\tlink VarNode " << argument << " -> "
            << "OpNode " << op_nodes_.back()->Name() << std::endl;
        op_nodes_.back()->add_input_node(parameter, GetVar(argument));
        GetVar(argument)->add_output_node(op_nodes_.back().get());
      }
    }

    auto& outputs = ops[i].outputs();
    for (auto j = 0; j < outputs.size(); ++j) {
      std::string parameter(outputs[j].parameter().c_str());
      for (auto k = 0; k < outputs[j].arguments_size(); ++k) {
        std::string argument(outputs[j].arguments(k).c_str());
        ofs << "\tlink OpNode " << op_nodes_.back()->Name() << " -> "
            << "VarNode " << argument << std::endl;
        op_nodes_.back()->add_output_node(parameter, GetVar(argument));
        GetVar(argument)->add_input_node(op_nodes_.back().get());
      }
    }

    ofs << "Build link for OpNode " << &op_nodes_.back() << " finished"
        << std::endl;
  }

  for (auto& node : var_nodes_) {
    ofs << node->to_string() << std::endl;
  }

  for (auto& node : op_nodes_) {
    ofs << node->to_string() << std::endl;
  }

  // remove unused var
  // do {
  //   bool has_unused_var = false;
  //   for (auto i = 0; i < var_nodes_.size(); ++i) {
  //     auto& var = var_nodes_[i];
  //     if (var->inputs().size() == 0 && var->outputs().size() == 0) {
  //       ofs << "VarNode " << var->name() << " is unused, remove
  //       it.\n"; std::remove(
  //           var_nodes_.begin(), var_nodes_.end(), var_nodes_.begin() + i);
  //       has_unused_var = true;
  //       break;
  //     }
  //   }
  //   if (has_unused_var == false) {
  //     break;
  //   }
  // } while (1);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
