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
void TopkKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::Scalar& k_scalar,
                int axis,
                bool largest,
                bool sorted,
                phi::DenseTensor* out,
                phi::DenseTensor* indices) {
  if (axis < 0) {
    axis += x.dims().size();
  }

  int k = k_scalar.to<int>();

  phi::DDim output_dims = x.dims();
  output_dims[axis] = k;

  out->Resize(output_dims);
  indices->Resize(output_dims);

  dev_ctx.template Alloc<T>(out);
  dev_ctx.template Alloc<int64_t>(indices);

  // Support 0D
  if (output_dims.size() == 0) {
    TensorCopy(dev_ctx, x, true, out);
    FillNpuTensorWithConstant<int64_t>(
        indices, dev_ctx, static_cast<int64_t>(0.0));
    indices->Resize(output_dims);
    return;
  }

  phi::DenseTensor indices_int32;
  phi::DenseTensorMeta indices_int32_meta = {phi::DataType::INT32, output_dims};
  indices_int32.set_meta(indices_int32_meta);
  dev_ctx.template Alloc<int32_t>(&indices_int32);

  auto npu_stream = dev_ctx.stream();

  NpuOpRunner npu_op_runner_topkv2;
  npu_op_runner_topkv2.SetType("TopKV2")
      .AddInput(x)
      .AddInput(dev_ctx, std::vector<int32_t>{k})
      .AddOutput(*out)
      .AddOutput(indices_int32)
      .AddAttr("sorted", sorted)
      .AddAttr("dim", axis)
      .AddAttr("largest", largest)
      .Run(npu_stream);

  // Cast 'indices_int32' to 'indices', from INT32 to INT64
  auto dst_dtype = ConvertToNpuDtype(indices->dtype());
  const auto& npu_op_runner_cast =
      NpuOpRunner("Cast",
                  {indices_int32},
                  {*indices},
                  {{"dst_type", static_cast<int>(dst_dtype)}});
  npu_op_runner_cast.Run(npu_stream);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(topk,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::TopkKernel,
                          phi::dtype::float16,
                          float,
                          double,
                          int,
                          int64_t) {}
