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

#include "runtime/ge_c_api.h"

namespace ge {
namespace capi {

class Operator {
 public:
  Operator(const std::string& op_type, const std::string& op_name = "") {
    static std::unordered_map<std::string, size_t> op_count;
    op_type_ = op_type;
    if (op_name.size()) {
      op_name_ = op_name;
    } else {
      op_name_ = op_type_ + "_" + std::to_string(op_count[op_type_]);
      op_count[op_type_] = op_count[op_type_] + 1;
    }
    raw_data_ = CreateOperator(op_name_.c_str(), op_type_.c_str());
    own_data_ = true;
  }

  Operator(const Operator& other) {
    raw_data_ = other.raw_data_;
    own_data_ = false;
  }

  Operator(Operator&& other) {
    raw_data_ = other.raw_data_;
    own_data_ = other.own_data_;
    other.own_data_ = false;
    other.raw_data_ = nullptr;
  }

  Operator& operator=(const Operator& other) {
    raw_data_ = other.raw_data_;
    own_data_ = false;
    return *this;
  }

  Operator& operator=(Operator&& other) {
    raw_data_ = other.raw_data_;
    own_data_ = other.own_data_;
    other.own_data_ = false;
    other.raw_data_ = nullptr;
    return *this;
  }

  ~Operator() {
    if (own_data_ && raw_data_) {
      DestroyOperator(raw_data_);
      raw_data_ = nullptr;
      own_data_ = false;
    }
  }

  const std::string& Name() const { return op_name_; }

  const std::string& Type() const { return op_type_; }

  std::vector<int> GetOutputShape(int index) {
    return OperatorGetOutputShapeByIndex<int>(raw_data_, index);
  }

  template <typename T>
  void SetAttr(const std::string& key, const T& value) {
    OperatorSetAttr(raw_data, key.c_str(), value);
  }

  void SetInput(int input_index, Operator* other, int other_output_index) {
    OperatorSetInput(
        raw_data_, input_index, other->raw_data_, other_output_index);
  }

  void AddControlInput(Operator* other) {
    OperatorAddControlInput(raw_data_, other->raw_data_);
  }

 private:
  std::string op_type_;
  std::string op_name_;
  C_GE_Operator* raw_data_{nullptr};
  bool own_data_{false};
};

}  // namespace capi
}  // namespace ge
