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

#include "kernels/funcs/reduce_op.h"

namespace custom_kernel {

template <typename T, typename Context>
void MeanRawKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::IntArray& axes,
                   bool keep_dim,
                   bool reduce_all,
                   phi::DenseTensor* out) {
  MLUReduceOp<T>(
      dev_ctx, x, axes.GetData(), keep_dim, reduce_all, "reduce_mean", out);
}

template <typename T, typename Context>
void MeanKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::IntArray& dims,
                bool keep_dim,
                phi::DenseTensor* out) {
  bool reduce_all = false;
  custom_kernel::MeanRawKernel<T>(dev_ctx, x, dims, keep_dim, reduce_all, out);
}

template <typename T, typename Context>
void MeanGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& out_grad,
                    const phi::IntArray& axes,
                    bool keep_dim,
                    bool reduce_all,
                    phi::DenseTensor* x_grad) {
  dev_ctx.template Alloc<T>(x_grad);

  auto reduce_dims = axes.GetData();
  auto input_dims = phi::vectorize(x.dims());

  if (!keep_dim && reduce_dims.size() == 0) reduce_all = true;
  int reduce_numel = 1;
  if (reduce_all) {
    reduce_dims.clear();
    for (size_t d = 0; d < input_dims.size(); ++d) {
      reduce_dims.push_back(static_cast<int>(d));
    }
  }
  for (auto& d : reduce_dims) {
    if (d < 0) {
      d = d + input_dims.size();
    }
    reduce_numel *= input_dims[d];
  }

  Tensor tmp_output_grad;
  auto tmp_output_dims = input_dims;
  for (auto d : reduce_dims) {
    tmp_output_dims[d] = 1;
  }
  tmp_output_grad = out_grad;
  tmp_output_grad.Resize(phi::make_ddim(tmp_output_dims));

  MLUCnnlTensorDesc output_grad_desc(tmp_output_grad,
                                     CNNL_LAYOUT_ARRAY,
                                     ToCnnlDataType(tmp_output_grad.dtype()));
  MLUCnnlTensorDesc input_grad_desc(
      *x_grad, CNNL_LAYOUT_ARRAY, ToCnnlDataType(x_grad->dtype()));

  auto value = static_cast<T>(1.0 / static_cast<float>(reduce_numel));
  MLUCnnl::Fill(dev_ctx,
                CNNL_POINTER_MODE_HOST,
                &value,
                input_grad_desc.get(),
                GetBasePtr(x_grad));

  MLUCnnlOpTensorDesc op_tensor_desc(
      CNNL_OP_TENSOR_MUL, ToCnnlDataType<T>(), CNNL_NOT_PROPAGATE_NAN);

  MLUCnnl::OpTensor(dev_ctx,
                    op_tensor_desc.get(),
                    output_grad_desc.get(),
                    GetBasePtr(&tmp_output_grad),
                    input_grad_desc.get(),
                    GetBasePtr(x_grad),
                    input_grad_desc.get(),
                    GetBasePtr(x_grad),
                    ToCnnlDataType<T>());
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(mean_raw,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::MeanRawKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(mean,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::MeanKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(mean_grad,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::MeanGradKernel,
                          float,
                          phi::dtype::float16) {}
