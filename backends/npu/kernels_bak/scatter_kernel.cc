
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
void ScatterKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& index,
                   const phi::DenseTensor& updates,
                   bool overwrite,
                   phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  phi::DenseTensor tmp_tensor(index);
  const auto index_dims = index.dims();
  if (index_dims.size() == 1) {
    std::vector<int64_t> new_dim = {index_dims[0], 1};
    tmp_tensor.Resize(phi::make_ddim(new_dim));
  }

  auto op_func_update = [](const std::vector<phi::DenseTensor>& inputs,
                           const std::vector<phi::DenseTensor>& outputs,
                           const NPUAttributeMap& attrs,
                           const Context& dev_ctx) {
    const auto& runner =
        NpuOpRunner("TensorScatterUpdate", inputs, outputs, attrs);
    runner.Run(dev_ctx.stream());
  };
  auto op_func_add = [](const std::vector<phi::DenseTensor>& inputs,
                        const std::vector<phi::DenseTensor>& outputs,
                        const NPUAttributeMap& attrs,
                        const Context& dev_ctx) {
    const auto& runner =
        NpuOpRunner("TensorScatterAdd", inputs, outputs, attrs);
    runner.Run(dev_ctx.stream());
  };

  if (overwrite) {
    if (x.dtype() == phi::DenseTensorMeta::DataType::INT64) {
      NpuOpRunner::TypeAdapter({x, tmp_tensor, updates},
                               {*out},
                               {},
                               dev_ctx,
                               op_func_update,
                               {phi::DenseTensorMeta::DataType::INT32,
                                phi::DenseTensorMeta::DataType::INT32,
                                phi::DenseTensorMeta::DataType::INT32},
                               {phi::DenseTensorMeta::DataType::INT32});
    } else {
      const auto& runner_update = NpuOpRunner(
          "TensorScatterUpdate", {x, tmp_tensor, updates}, {*out}, {});
      runner_update.Run(dev_ctx.stream());
    }
  } else {
    if (x.dtype() == phi::DenseTensorMeta::DataType::INT64) {
      NpuOpRunner::TypeAdapter({x, tmp_tensor, updates},
                               {*out},
                               {},
                               dev_ctx,
                               op_func_add,
                               {phi::DenseTensorMeta::DataType::INT32,
                                phi::DenseTensorMeta::DataType::INT32,
                                phi::DenseTensorMeta::DataType::INT32},
                               {phi::DenseTensorMeta::DataType::INT32});
    } else {
      const auto& runner_add =
          NpuOpRunner("TensorScatterAdd", {x, tmp_tensor, updates}, {*out}, {});
      runner_add.Run(dev_ctx.stream());
    }
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(scatter,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::ScatterKernel,
                          float,
                          int64_t,
                          int,
                          phi::dtype::float16) {}
