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
#include "kernels/funcs/op_command.h"

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

  auto k = k_scalar.to<int>();
  phi::DDim output_dims = x.dims();
  output_dims[axis] = k;

  out->Resize(output_dims);
  indices->Resize(output_dims);

  dev_ctx.template Alloc<T>(out);
  dev_ctx.template Alloc<int64_t>(indices);

  phi::DenseTensor indices_int32;
  phi::DenseTensorMeta indices_int32_meta = {phi::DataType::INT32, output_dims};
  indices_int32.set_meta(indices_int32_meta);
  dev_ctx.template Alloc<int32_t>(&indices_int32);

  phi::DenseTensor k_tensor;
  k_tensor.Resize({1});
  dev_ctx.template HostAlloc<int32_t>(&k_tensor);
  *(k_tensor.data<int32_t>()) = k;

  experimental::OpCommand("TopKV2")
      .Input(x,
             experimental::TensorDescMaker("x").FromTensor(x).SetDataLayout(
                 phi::DataLayout::ANY))
      .Input(
          k_tensor,
          experimental::TensorDescMaker("k").FromTensor(k_tensor).SetDataLayout(
              phi::DataLayout::ANY))
      .Output(*out)
      .Output(indices_int32)
      .Attr("sorted", sorted)
      .Attr("dim", axis)
      .Attr("largest", largest)
      .Run(dev_ctx);

  experimental::OpCommand("Cast")
      .Input(indices_int32)
      .Output(*indices)
      .Attr("dst_type", static_cast<int>(ConvertToNpuDtype(indices->dtype())))
      .Run(dev_ctx);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(top_k,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::TopkKernel,
                          phi::dtype::float16,
                          float,
                          double,
                          int,
                          int64_t) {}
