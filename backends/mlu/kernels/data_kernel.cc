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

#include "kernels/funcs/mlu_baseop.h"
#include "kernels/funcs/mlu_funcs.h"
#include "paddle/phi/kernels/funcs/tensor_formatter.h"

namespace custom_kernel {

const char kForward[] = "FORWARD";
const char kBackward[] = "BACKWARD";

template <typename T, typename Context>
void PrintKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 int first_n,
                 const std::string& message,
                 int summarize,
                 bool print_tensor_name,
                 bool print_tensor_type,
                 bool print_tensor_shape,
                 bool print_tensor_layout,
                 bool print_tensor_lod,
                 const std::string& print_phase,
                 bool is_forward,
                 phi::DenseTensor* out) {
  TensorCopy(dev_ctx, x, false, out);
  phi::DenseTensorMeta meta = {x.dtype(), x.dims()};
  out->set_meta(meta);

  if ((is_forward && print_phase == kBackward) ||
      (!is_forward && print_phase == kForward)) {
    return;
  }

  paddle::funcs::TensorFormatter formatter;
  const std::string& name = print_tensor_name ? "var" : "";
  formatter.SetPrintTensorType(print_tensor_type);
  formatter.SetPrintTensorShape(print_tensor_shape);
  formatter.SetPrintTensorLod(print_tensor_lod);
  formatter.SetPrintTensorLayout(print_tensor_layout);
  formatter.SetSummarize(summarize);
  formatter.Print(x, name, message);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(print_kernel,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::PrintKernel,
                          bool,
                          float,
                          int32_t,
                          int64_t,
                          double,
                          phi::float16) {}
