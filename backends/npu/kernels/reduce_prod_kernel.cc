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
void ProdKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::IntArray& axes,
                   bool keep_dim,
                   bool reduce_all,
                   phi::DenseTensor* out) {
  auto dims = axes.GetData();
  auto x_dims = x.dims();
  auto x_dims_size = x_dims.size();
  dev_ctx.template Alloc<T>(out);

  NPUAttributeMap attr_input = {{"axes", dims}, {"keep_dims", keep_dim}};

  if (reduce_all) {
    std::vector<int> dim_vec;
    for (int i = 0; i < x_dims_size; i++) {
      dim_vec.push_back(i);
    }

    attr_input = {{"axes", dim_vec}, {"keep_dims", keep_dim}};
  }

  if (x.dtype() == phi::DenseTensorMeta::DataType::INT64) {
    auto op_func = [](const std::vector<phi::DenseTensor>& inputs,
                      const std::vector<phi::DenseTensor>& outputs,
                      const NPUAttributeMap& attrs,
                      const phi::CustomContext& dev_ctx) {
      const auto& runner =
          NpuOpRunner("ReduceProdD", {inputs[0]}, {outputs[0]}, attrs);
      runner.Run(dev_ctx.stream());
    };

    NpuOpRunner::TypeAdapter({x},
                             {*out},
                             attr_input,
                             dev_ctx,
                             op_func,
                             {phi::DenseTensorMeta::DataType::INT32},
                             {phi::DenseTensorMeta::DataType::INT32});
  } else {
    // TODO(Aganlengzi): remove this branch when performance of ReduceProdD
    // is good enough for big shapes.
    // Here, we use SplitV and Mul to deal with special cases.
    if (x_dims[x_dims_size - 1] == 2 && dims.size() == 1 &&
        (dims[0] == -1 || dims[0] == x_dims_size - 1)) {
      auto stream = dev_ctx.stream();
      phi::DenseTensor x1, x2;
      x1.set_meta(out->meta());
      x2.set_meta(out->meta());
      dev_ctx.template Alloc<T>(&x1);
      dev_ctx.template Alloc<T>(&x2);
      // split
      std::vector<phi::DenseTensor> outputs;
      outputs.push_back(x1);
      outputs.push_back(x2);
      std::vector<int> sections = {1, 1};
      NpuOpRunner runner_split;
      runner_split.SetType("SplitV")
          .AddInput(x)
          .AddInput(dev_ctx, std::move(sections))
          .AddInput(dev_ctx, std::vector<int32_t>({-1}))
          .AddOutputs(outputs)
          .AddAttrs({{"num_split", static_cast<int32_t>(sections.size())}})
          .Run(stream);
      // elementwise mul
      const auto& runner = NpuOpRunner("Mul", {x1, x2}, {*out}, {});
      runner.Run(stream);
    } else {
      const auto& runner = NpuOpRunner("ReduceProdD", {x}, {*out}, attr_input);
      runner.Run(dev_ctx.stream());
    }
  }
}

template <typename T, typename Context>
void ProdInferKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::IntArray& dims,
                bool keep_dim,
                phi::DenseTensor* out) {
  bool reduce_all = false;
  custom_kernel::ProdKernel<T>(dev_ctx, x, dims, keep_dim, reduce_all, out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(prod,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::ProdKernel,
                          phi::dtype::float16,
                          float) {}

PD_REGISTER_PLUGIN_KERNEL(prod_infer,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::ProdInferKernel,
                          phi::dtype::float16,
                          float) {}
