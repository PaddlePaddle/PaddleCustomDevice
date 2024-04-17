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
void AclopMinRawKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::IntArray& axes,
                       bool keep_dim,
                       bool reduce_all,
                       phi::DenseTensor* out) {
  auto dims = axes.GetData();
  dev_ctx.template Alloc<T>(out);

  NPUAttributeMap attr_input = {{"axes", dims}, {"keep_dims", keep_dim}};

  if (reduce_all) {
    std::vector<int> dim_vec;
    for (int i = 0; i < x.dims().size(); i++) {
      dim_vec.push_back(i);
    }

    attr_input = {{"axes", dim_vec}, {"keep_dims", keep_dim}};
  }

  if (x.dtype() == phi::DataType::INT64) {
    auto op_func = [](const std::vector<phi::DenseTensor>& inputs,
                      const std::vector<phi::DenseTensor>& outputs,
                      const NPUAttributeMap& attrs,
                      const phi::CustomContext& dev_ctx) {
      const auto& runner =
          NpuOpRunner("ReduceMinD", {inputs[0]}, {outputs[0]}, attrs);
      runner.Run(dev_ctx.stream());
    };

    NpuOpRunner::TypeAdapter({x},
                             {*out},
                             attr_input,
                             dev_ctx,
                             op_func,
                             {phi::DataType::INT32},
                             {phi::DataType::INT32});
  } else {
    const auto& runner = NpuOpRunner("ReduceMinD", {x}, {*out}, attr_input);
    runner.Run(dev_ctx.stream());
  }
}

template <typename T, typename Context>
void MinRawKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::IntArray& axes,
                  bool keep_dim,
                  bool reduce_all,
                  phi::DenseTensor* out) {
  DO_COMPATIBILITY(aclnnAmin,
                   (custom_kernel::AclopMinRawKernel<T, Context>(
                       dev_ctx, x, axes, keep_dim, reduce_all, out)));
  dev_ctx.template Alloc<T>(out);

  if (reduce_all) {
    std::vector<int64_t> dim_vec;
    for (int64_t i = 0; i < x.dims().size(); i++) {
      dim_vec.push_back(i);
    }
    EXEC_NPU_CMD(aclnnAmin, dev_ctx, x, dim_vec, keep_dim, *out);
  } else {
    EXEC_NPU_CMD(aclnnAmin, dev_ctx, x, axes, keep_dim, *out);
  }
}

template <typename T, typename Context>
void MinKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::IntArray& dims,
               bool keep_dim,
               phi::DenseTensor* out) {
  bool reduce_all = false;
  if (dims.size() == 0) {
    reduce_all = true;
  }
  custom_kernel::MinRawKernel<T>(dev_ctx, x, dims, keep_dim, reduce_all, out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(min_raw,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::MinRawKernel,
                          int32_t,
                          int64_t,
                          phi::dtype::float16,
                          float) {}

PD_REGISTER_PLUGIN_KERNEL(min,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::MinKernel,
                          int32_t,
                          int64_t,
                          phi::dtype::float16,
                          float) {}
