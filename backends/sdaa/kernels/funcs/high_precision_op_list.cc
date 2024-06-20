// BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
// reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
// OF SUCH DAMAGE.

#include "kernels/funcs/high_precision_op_list.h"

#include <mutex>
#include <string>
#include <unordered_set>

#include "kernels/funcs/sdaa_funcs.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

static void tokenize(const std::string& ops,
                     char delim,
                     std::unordered_set<std::string>* op_set) {
  std::string::size_type beg = 0;
  for (uint64_t end = 0; (end = ops.find(delim, end)) != std::string::npos;
       ++end) {
    op_set->insert(ops.substr(beg, end - beg));
    beg = end + 1;
  }
  op_set->insert(ops.substr(beg));
}

static void check_support(const std::unordered_set<std::string>& ops) {
  std::unordered_set<std::string> support_high_precision_ops{
      "softmax",
      "softmax_grad",
      "softmax_with_cross_entropy",
      "softmax_with_cross_entropy_grad"};
  for (const auto& item : ops) {
    if (support_high_precision_ops.find(item) ==
        support_high_precision_ops.end()) {
      PADDLE_THROW(phi::errors::InvalidArgument(
          "Teco-Paddle does not support the high precision mode for %s, please "
          "reset it.",
          item));
    }
  }
}

std::unordered_set<std::string> get_op_list() {
  std::unordered_set<std::string> op_list;

  if (std::getenv("HIGH_PRECISION_OP_LIST") != nullptr) {
    std::string ops(std::getenv("HIGH_PRECISION_OP_LIST"));

    tokenize(ops, ',', &op_list);

    check_support(op_list);
  }

  return op_list;
}

bool is_in_high_precision_op_list(const std::string& op_name) {
  static std::unordered_set<std::string> high_precision_op_list = get_op_list();

  if (high_precision_op_list.find(op_name) != high_precision_op_list.end()) {
    return true;
  }
  return false;
}

}  // namespace custom_kernel
