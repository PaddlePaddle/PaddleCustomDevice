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
#include "acl/acl.h"
#include "glog/logging.h"
#include "kernels/funcs/npu_funcs.h"
#include "paddle/extension.h"
#include "paddle/phi/extension.h"

template <typename T>
std::string GetVectorString(const std::vector<T>& shape);

std::string GetDataBufferString(const aclDataBuffer* buf);

std::string GetTensorDescString(const aclTensorDesc* desc);

std::string GetOpDescString(std::vector<aclTensorDesc*> descs,
                            const std::string msg);

std::string GetOpInfoString(std::vector<aclTensorDesc*> descs,
                            std::vector<aclDataBuffer*> buffs,
                            const std::string msg);

template <typename Context, typename T>
void FormatData(const Context& dev_ctx,
                const phi::DenseTensor& print_tensor,
                std::stringstream& log_stream);

template <typename Context>
std::string GetPDTensorString(const Context& dev_ctx,
                              const phi::DenseTensor& print_tensor,
                              const std::string& tensor_name = "",
                              const std::string& message = "");

template <typename T = std::string>
std::vector<T> split_string(const std::string& str, const std::string& delim);

template <typename T = std::string>
std::vector<T> split_string(const std::string& str);

template <class Container>
std::string join_strings(const Container& strs, char delim) {
  std::string str;

  size_t i = 0;
  for (auto& elem : strs) {
    if (i > 0) {
      str += delim;
    }

    std::stringstream ss;
    ss << elem;
    str += ss.str();
    ++i;
  }

  return str;
}

template <class Container>
std::string join_strings(const Container& strs, const std::string& delim) {
  std::string str;

  size_t i = 0;
  for (auto& elem : strs) {
    if (i > 0) {
      str += delim;
    }

    std::stringstream ss;
    ss << elem;
    str += ss.str();
    ++i;
  }

  return str;
}

template <class Container, class DelimT, class ConvertFunc>
std::string join_strings(const Container& strs,
                         DelimT&& delim,
                         ConvertFunc&& func) {
  std::stringstream ss;
  size_t i = 0;
  for (const auto& elem : strs) {
    if (i > 0) {
      ss << delim;
    }
    ss << func(elem);
    ++i;
  }

  return ss.str();
}
