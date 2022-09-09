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

#include <set>

#include "kernels/funcs/reduce_op.h"

namespace custom_kernel {

template <typename T, typename Context>
void SumRawKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::IntArray& axes,
                  bool keep_dim,
                  bool reduce_all,
                  phi::DenseTensorMeta::DataType out_dtype,
                  phi::DenseTensor* out) {
  MLUReduceOp<T>(
      dev_ctx, x, axes.GetData(), keep_dim, reduce_all, "reduce_sum", out);
}

template <typename T, typename Context>
void SumKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::IntArray& dims,
               phi::DenseTensorMeta::DataType out_dtype,
               bool keep_dim,
               phi::DenseTensor* out) {
  bool reduce_all = false;
  if (dims.size() == 0) {
    reduce_all = true;
  }
  custom_kernel::SumRawKernel<T>(
      dev_ctx, x, dims, keep_dim, reduce_all, out_dtype, out);
}

template <typename T, typename Context>
void SumGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& out_grad,
                   const phi::IntArray& dims_array,
                   bool keep_dim,
                   bool reduce_all,
                   phi::DenseTensor* x_grad) {
  auto reduce_dims = dims_array.GetData();
  dev_ctx.template Alloc<T>(x_grad);

  auto in_dims = phi::vectorize(x.dims());
  if (reduce_all) {
    reduce_dims.clear();
    for (size_t d = 0; d < in_dims.size(); ++d) {
      reduce_dims.push_back(static_cast<int>(d));
    }
  }
  for (auto& d : reduce_dims) {
    if (d < 0) {
      d = d + in_dims.size();
    }
  }

  Tensor tmp_out;
  auto tmp_output_dims = in_dims;
  for (auto d : reduce_dims) {
    tmp_output_dims[d] = 1;
  }
  tmp_out = out_grad;
  tmp_out.Resize(phi::make_ddim(tmp_output_dims));

  MLUCnnlTensorDesc out_desc(tmp_out, CNNL_LAYOUT_ARRAY, ToCnnlDataType<T>());
  MLUCnnlTensorDesc in_grad_desc(
      *x_grad, CNNL_LAYOUT_ARRAY, ToCnnlDataType<T>());

  MLUCnnl::BroadcastTo(dev_ctx,
                       out_desc.get(),
                       GetBasePtr(&tmp_out),
                       in_grad_desc.get(),
                       GetBasePtr(x_grad));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(sum_raw,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::SumRawKernel,
                          int32_t,
                          phi::dtype::float16,
                          float) {
  kernel->OutputAt(0).SetDataType(paddle::experimental::DataType::UNDEFINED);
}

PD_REGISTER_PLUGIN_KERNEL(sum,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::SumKernel,
                          int32_t,
                          phi::dtype::float16,
                          float) {
  kernel->OutputAt(0).SetDataType(paddle::experimental::DataType::UNDEFINED);
}

PD_REGISTER_PLUGIN_KERNEL(sum_grad,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::SumGradKernel,
                          phi::dtype::float16,
                          float) {}
