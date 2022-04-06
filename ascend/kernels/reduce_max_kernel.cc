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

#include "npu_funcs.h"
#include "npu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void MaxRawKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const std::vector<int64_t>& axes,
                  bool keep_dim,
                  bool reduce_all,
                  phi::DenseTensor* out) {
  phi::DenseTensor cast_out;
  auto dims = axes;
  cast_out.Resize(out->dims());
  dev_ctx.template Alloc<T>(&cast_out);

  auto cast_out_dtype = x.dtype();

  if (x.dtype() != cast_out_dtype) {
    if (cast_out_dtype == phi::DenseTensorMeta::DataType::FLOAT32) {
      dev_ctx.template Alloc<float>(out);
    } else if (cast_out_dtype == phi::DenseTensorMeta::DataType::FLOAT16) {
      dev_ctx.template Alloc<float>(out);
    } else if (cast_out_dtype == phi::DenseTensorMeta::DataType::INT16) {
      dev_ctx.template Alloc<int16_t>(out);
    } else if (cast_out_dtype == phi::DenseTensorMeta::DataType::INT32) {
      dev_ctx.template Alloc<int32_t>(out);
    } else if (cast_out_dtype == phi::DenseTensorMeta::DataType::INT64) {
      dev_ctx.template Alloc<int64_t>(out);
    } else if (cast_out_dtype == phi::DenseTensorMeta::DataType::FLOAT64) {
      dev_ctx.template Alloc<double>(out);
    } else if (cast_out_dtype == phi::DenseTensorMeta::DataType::BOOL) {
      dev_ctx.template Alloc<bool>(out);
    }
  } else {
    *out = cast_out;
  }

  NPUAttributeMap attr_input = {{"axes", dims}, {"keep_dims", keep_dim}};

  if (reduce_all) {
    std::vector<int> dim_vec;
    for (int i = 0; i < x.dims().size(); i++) {
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
          NpuOpRunner("ReduceMaxD", {inputs[0]}, {outputs[0]}, attrs);
      runner.Run(dev_ctx.stream());
    };

    NpuOpRunner::TypeAdapter({x},
                             {cast_out},
                             attr_input,
                             dev_ctx,
                             op_func,
                             {phi::DenseTensorMeta::DataType::INT32},
                             {phi::DenseTensorMeta::DataType::INT32});
  } else {
    const auto& runner = NpuOpRunner("ReduceMaxD", {x}, {cast_out}, attr_input);
    runner.Run(dev_ctx.stream());
  }

  if (x.dtype() != cast_out_dtype) {
    auto dst_dtype = ConvertToNpuDtype(cast_out_dtype);
    const auto& runner_cast =
        NpuOpRunner("Cast",
                    {cast_out},
                    {*out},
                    {{"dst_type", static_cast<int>(dst_dtype)}});
    runner_cast.Run(dev_ctx.stream());
  }
}

template <typename T, typename Context>
void MaxKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const std::vector<int64_t>& dims,
               bool keep_dim,
               phi::DenseTensor* out) {
  bool reduce_all = false;
  custom_kernel::MaxRawKernel<T>(dev_ctx, x, dims, keep_dim, reduce_all, out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(max_raw,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::MaxRawKernel,
                          int32_t,
                          int64_t,
                          phi::dtype::float16,
                          float) {}

PD_REGISTER_PLUGIN_KERNEL(max,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::MaxKernel,
                          int32_t,
                          int64_t,
                          phi::dtype::float16,
                          float) {}
