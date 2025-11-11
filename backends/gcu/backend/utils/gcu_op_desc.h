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

#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "backend/utils/types.h"
#include "paddle/common/macros.h"
#include "paddle/phi/core/enforce.h"

namespace backend {

class GcuOpDesc {
 public:
  GcuOpDesc() {}

  GcuOpDesc(const std::string &type,
            const TensorNameMap &inputs,
            const TensorNameMap &outputs,
            const GcuAttributeMap &attrs);

  GcuOpDesc(const GcuOpDesc &desc);

  GcuOpDesc &operator=(const GcuOpDesc &other);

  std::string Type() const { return type_; }

  void SetType(const std::string &type) { type_ = type; }

  const std::vector<std::string> &Input(const std::string &name) const;

  const std::vector<std::string> &Output(const std::string &name) const;

  bool HasOutput(const std::string &name) const;

  bool HasInput(const std::string &name) const;

  void SetInput(const std::string &param_name,
                const std::vector<std::string> &args);

  void SetOutput(const std::string &param_name,
                 const std::vector<std::string> &args);
  void RemoveOutput(const std::string &name);

  void RemoveInput(const std::string &name);

  bool HasAttr(const std::string &name) const;

  std::vector<std::string> AttrNames() const;

  void SetAttr(const std::string &name, const GcuAttribute &v);
  void RemoveAttr(const std::string &name);

  // NOTE(chenfeiyu): this template is added to avoid using a
  // variant(GcuAttribute) as a parameter of a function which is bound to
  // python, which causes unexpected type conversion due to the overload
  // resolution mechanism
  // https://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html#c-17-library-containers
  template <typename T>
  void SetPlainAttr(const std::string &name, const T &value) {
    SetAttr(name, value);
  }

  GcuAttribute GetAttr(const std::string &name) const;

  template <typename T>
  T GetAttrIfExists(const std::string &name) const {
    T result{};
    if (HasAttr(name)) {
      result = PADDLE_GET_CONST(T, GetAttr(name));
    }
    return result;
  }

  // Only be used in C++
  const GcuAttributeMap &GetAttrMap() const;

  // Only be used in C++
  void SetAttrMap(const GcuAttributeMap &attr_map);

  std::vector<std::string> InputNames() const { return MapKeys(inputs_); }
  std::vector<std::string> OutputNames() const { return MapKeys(outputs_); }

  const TensorNameMap &Inputs() const { return inputs_; }

  const TensorNameMap &Outputs() const { return outputs_; }

  TensorNameMap *MutableInputs() { return &this->inputs_; }

  TensorNameMap *MutableOutputs() { return &this->outputs_; }

  GcuAttributeMap *MutableAttrMap() { return &this->attrs_; }

 private:
  template <typename MapType>
  static std::vector<typename MapType::key_type> MapKeys(const MapType &map) {
    std::vector<typename MapType::key_type> ret_val;
    ret_val.reserve(map.size());
    std::transform(
        map.begin(),
        map.end(),
        std::back_inserter(ret_val),
        [](const typename MapType::value_type &pair) { return pair.first; });
    return ret_val;
  }

  // input arg name => input variable names
  TensorNameMap inputs_;
  // output arg name => output variable names
  TensorNameMap outputs_;
  // attribute name => all original attrs
  GcuAttributeMap attrs_;
  std::string type_;
};

}  // namespace backend
