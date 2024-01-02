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

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void EyeKernel(const Context& dev_ctx,
               const phi::Scalar& rows,
               const phi::Scalar& columns,
               phi::DataType dtype,
               phi::DenseTensor* out) {
  auto npu_dtype = ConvertToNpuDtype(dtype);
  auto num_columns = columns.to<int64_t>();
  auto num_rows = rows.to<int64_t>();
  if (num_columns == -1) num_columns = num_rows;

  dev_ctx.template Alloc<T>(out);
  if (out->dtype() == phi::DataType::INT64 ||
      out->dtype() == phi::DataType::FLOAT64) {
    auto op_func = [](const std::vector<phi::DenseTensor>& inputs,
                      const std::vector<phi::DenseTensor>& outputs,
                      const NPUAttributeMap& attrs,
                      const Context& dev_ctx) {
      const auto& runner = NpuOpRunner("Eye", {}, outputs, attrs);
      runner.Run(dev_ctx.stream());
    };
    NPUAttributeMap attr_input = {
        {"num_rows", num_rows},
        {"num_columns", num_columns},
        {"dtype", ConvertToNpuDtype(phi::DataType::FLOAT32)}};
    NpuOpRunner::TypeAdapter(
        {}, {*out}, attr_input, dev_ctx, op_func, {}, {phi::DataType::FLOAT32});
  } else {
    NPUAttributeMap attr_input = {{"num_rows", num_rows},
                                  {"num_columns", num_columns},
                                  {"dtype", npu_dtype}};
    const auto& runner = NpuOpRunner("Eye", {}, {*out}, attr_input);
    auto stream = dev_ctx.stream();
    runner.Run(stream);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(eye,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::EyeKernel,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16) {}
