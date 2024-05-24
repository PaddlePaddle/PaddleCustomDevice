// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "backend/utils/gcu_op_desc.h"

#include <string>

#include "backend/utils/types.h"
#include "glog/logging.h"
#include "paddle/common/macros.h"
#include "paddle/phi/common/complex.h"
#include "paddle/utils/blank.h"

namespace backend {

GcuOpDesc::GcuOpDesc(const std::string &type,
                     const TensorNameMap &inputs,
                     const TensorNameMap &outputs,
                     const GcuAttributeMap &attrs) {
  type_ = type;
  inputs_ = inputs;
  outputs_ = outputs;
  attrs_ = attrs;
}

GcuOpDesc::GcuOpDesc(const GcuOpDesc &other) {
  type_ = other.type_;
  inputs_ = other.inputs_;
  outputs_ = other.outputs_;
  attrs_ = other.attrs_;
}

// Explicitly implement the assign operator, Since the added
// unique_ptr data member does not have the implicit assign operator.
GcuOpDesc &GcuOpDesc::operator=(const GcuOpDesc &other) {
  type_ = other.type_;
  inputs_ = other.inputs_;
  outputs_ = other.outputs_;
  attrs_ = other.attrs_;
  return *this;
}

const std::vector<std::string> &GcuOpDesc::Input(
    const std::string &name) const {
  auto it = inputs_.find(name);
  PADDLE_ENFORCE_NE(
      it,
      inputs_.end(),
      phi::errors::NotFound(
          "Input %s cannot be found in operator %s.", name, Type()));
  return it->second;
}

void GcuOpDesc::SetInput(const std::string &param_name,
                         const std::vector<std::string> &args) {
  inputs_[param_name] = args;
}

const std::vector<std::string> &GcuOpDesc::Output(
    const std::string &name) const {
  auto it = outputs_.find(name);
  PADDLE_ENFORCE_NE(
      it,
      outputs_.end(),
      phi::errors::NotFound(
          "Output %s cannot be found in operator %s.", name, Type()));
  return it->second;
}

bool GcuOpDesc::HasOutput(const std::string &name) const {
  return outputs_.find(name) != outputs_.end();
}

bool GcuOpDesc::HasInput(const std::string &name) const {
  return inputs_.find(name) != inputs_.end();
}

void GcuOpDesc::SetOutput(const std::string &param_name,
                          const std::vector<std::string> &args) {
  this->outputs_[param_name] = args;
}

void GcuOpDesc::RemoveOutput(const std::string &name) { outputs_.erase(name); }

void GcuOpDesc::RemoveInput(const std::string &name) { inputs_.erase(name); }

std::vector<std::string> GcuOpDesc::AttrNames() const {
  std::vector<std::string> retv;
  retv.reserve(attrs_.size());
  for (auto &attr : attrs_) {
    retv.push_back(attr.first);
  }
  return retv;
}

bool GcuOpDesc::HasAttr(const std::string &name) const {
  auto iter = attrs_.find(name);
  bool is_found = true;
  if (iter == attrs_.end()) {
    is_found = false;
  }

  return is_found;
}

void GcuOpDesc::RemoveAttr(const std::string &name) { attrs_.erase(name); }

void GcuOpDesc::SetAttr(const std::string &name, const GcuAttribute &v) {
  attrs_[name] = v;
}

void GcuOpDesc::SetAttrMap(
    const std::unordered_map<std::string, GcuAttribute> &attr_map) {
  attrs_ = attr_map;
}

GcuAttribute GcuOpDesc::GetAttr(const std::string &name) const {
  auto it = attrs_.find(name);
  PADDLE_ENFORCE_NE(
      it,
      attrs_.end(),
      phi::errors::NotFound("GcuAttribute %s is not found.", name));

  return it->second;
}

}  // namespace backend
