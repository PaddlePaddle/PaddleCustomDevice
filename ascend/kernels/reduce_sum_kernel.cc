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
void SumRawKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const std::vector<int64_t>& axes,
                  bool keep_dim,
                  bool reduce_all,
                  phi::DenseTensorMeta::DataType out_dtype,
                  phi::DenseTensor* out) {
  auto dims = axes;
  dev_ctx.template Alloc<T>(out);

  // special case
  if (x.dims().size() == 1 && keep_dim == false) {
    keep_dim = true;
  }

  aclrtStream stream = static_cast<aclrtStream>(dev_ctx.stream());

  phi::DenseTensor cast_x;
  phi::DenseTensor cast_out;
  // NOTE: ReduceSumD only supports fp32 and fp16
  if (x.dtype() != phi::DenseTensorMeta::DataType::FLOAT32 &&
      x.dtype() != phi::DenseTensorMeta::DataType::FLOAT16) {
    cast_x.Resize(x.dims());
    dev_ctx.template Alloc<T>(&cast_x);
    cast_out.Resize(out->dims());
    dev_ctx.template Alloc<T>(&cast_out);

    const auto& runner_cast = NpuOpRunner(
        "Cast", {x}, {cast_x}, {{"dst_type", static_cast<int>(ACL_FLOAT)}});
    runner_cast.Run(stream);
  } else {
    cast_x.ShareDataWith(x);
    cast_out.ShareDataWith(*out);
  }

  if (reduce_all) {
    std::vector<int> dim_vec;
    for (int i = 0; i < x.dims().size(); i++) {
      dim_vec.push_back(i);
    }

    const auto& runner =
        NpuOpRunner("ReduceSumD",
                    {cast_x},
                    {cast_out},
                    {{"axes", dim_vec}, {"keep_dims", keep_dim}});
    runner.Run(stream);

  } else {
    const auto& runner = NpuOpRunner("ReduceSumD",
                                     {cast_x},
                                     {cast_out},
                                     {{"axes", dims}, {"keep_dims", keep_dim}});
    runner.Run(stream);
  }

  if (x.dtype() != phi::DenseTensorMeta::DataType::FLOAT32 &&
      x.dtype() != phi::DenseTensorMeta::DataType::FLOAT16) {
    auto dst_dtype = ConvertToNpuDtype(out_dtype);
    const auto& runner_cast =
        NpuOpRunner("Cast",
                    {cast_out},
                    {*out},
                    {{"dst_type", static_cast<int>(dst_dtype)}});
    runner_cast.Run(stream);
  }
}

template <typename T, typename Context>
void SumKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const std::vector<int64_t>& dims,
               phi::DenseTensorMeta::DataType out_dtype,
               bool keep_dim,
               phi::DenseTensor* out) {
  bool reduce_all = false;
  SumRawKernel<T>(dev_ctx, x, dims, keep_dim, reduce_all, out_dtype, out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(sum_raw,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::SumRawKernel,
                          int32_t,
                          int64_t,
                          phi::dtype::float16,
                          float) {}
