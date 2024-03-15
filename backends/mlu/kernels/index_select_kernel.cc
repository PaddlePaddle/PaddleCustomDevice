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

#include "kernels/funcs/mlu_baseop.h"
#include "kernels/funcs/mlu_funcs.h"

namespace custom_kernel {

template <typename T, typename Context>
void IndexSelectKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& index,
                       int dim,
                       phi::DenseTensor* output) {
  dev_ctx.template Alloc<T>(output);
  MLUCnnlTensorDesc x_desc(x);
  MLUCnnlTensorDesc out_desc(*output);
  MLUCnnlTensorDesc index_desc(index);
  MLUCnnl::IndexSelect(dev_ctx,
                       dim,
                       x_desc.get(),
                       GetBasePtr(&x),
                       index_desc.get(),
                       GetBasePtr(&index),
                       out_desc.get(),
                       GetBasePtr(output));
}

template <typename T, typename Context>
void IndexSelectGradKernel(const Context& dev_ctx,
                           const phi::DenseTensor& x,
                           const phi::DenseTensor& index,
                           const phi::DenseTensor& out_grad,
                           int dim,
                           phi::DenseTensor* x_grad) {
  auto x_dims = x_grad->dims();
  auto out_dims = out_grad.dims();

  if (dim < 0) {
    dim += out_dims.size();
  }

  phi::DenseTensor casted_index;
  MLUCnnlTensorDesc index_desc(index);
  MLUCnnlTensorDesc out_grad_desc(out_grad);
  if (index.dtype() != phi::DataType::INT32) {
    casted_index.Resize(index.dims());
    dev_ctx.template Alloc<int32_t>(&casted_index);

    cnnlCastDataType_t cast_type =
        GetCastDataType(index.dtype(), DataType::INT32);
    MLUCnnlTensorDesc casted_index_desc(casted_index);
    MLUCnnl::Cast(dev_ctx,
                  cast_type,
                  index_desc.get(),
                  GetBasePtr(&index),
                  casted_index_desc.get(),
                  GetBasePtr(&casted_index));
  } else {
    casted_index = index;
  }

  if (dim == 0) {
    dev_ctx.template Alloc<T>(x_grad);
    MLUCnnlTensorDesc casted_index_desc(casted_index);
    MLUCnnlTensorDesc x_grad_desc(*x_grad);
    auto value_t = static_cast<T>(0.0f);
    MLUCnnl::Fill(dev_ctx,
                  CNNL_POINTER_MODE_HOST,
                  &value_t,
                  x_grad_desc.get(),
                  GetBasePtr(x_grad));

    MLUCnnl::UnsortedSegmentSum(
        dev_ctx,
        out_grad_desc.get(),
        GetBasePtr(&out_grad),
        casted_index_desc.get(),
        static_cast<const int*>(GetBasePtr(&casted_index)),
        x_grad_desc.get(),
        GetBasePtr(x_grad));
  } else {
    phi::DenseTensor transed_out_grad;
    std::vector<int> in_trans_perm;
    in_trans_perm.push_back(dim);
    for (int i = 0; i < out_dims.size(); ++i) {
      if (i == dim) continue;
      in_trans_perm.push_back(i);
    }
    phi::DDim transed_out_dims(out_dims);
    for (size_t i = 0; i < in_trans_perm.size(); ++i) {
      transed_out_dims[i] = out_dims[in_trans_perm[i]];
    }
    transed_out_grad.Resize(transed_out_dims);
    dev_ctx.template Alloc<T>(&transed_out_grad);
    MLUCnnlTensorDesc transed_out_grad_desc(transed_out_grad);
    const int in_trans_dim_size = in_trans_perm.size();
    MLUCnnl::Transpose(dev_ctx,
                       in_trans_perm,
                       in_trans_dim_size,
                       out_grad_desc.get(),
                       GetBasePtr(&out_grad),
                       transed_out_grad_desc.get(),
                       GetBasePtr(&transed_out_grad));

    phi::DenseTensor sum_out;
    phi::DDim sum_dims(x_dims);
    sum_dims[0] = x_dims[dim];
    auto idx = 1;
    for (int i = 0; i < x_dims.size(); ++i) {
      if (i == dim) continue;
      sum_dims[idx++] = x_dims[i];
    }
    sum_out.Resize(sum_dims);
    dev_ctx.template Alloc<T>(&sum_out);
    auto value_t = static_cast<T>(0.0f);
    MLUCnnlTensorDesc sum_out_desc(sum_out);
    MLUCnnl::Fill(dev_ctx,
                  CNNL_POINTER_MODE_HOST,
                  &value_t,
                  sum_out_desc.get(),
                  GetBasePtr(&sum_out));
    MLUCnnlTensorDesc casted_index_desc(casted_index);
    MLUCnnl::UnsortedSegmentSum(
        dev_ctx,
        transed_out_grad_desc.get(),
        GetBasePtr(&transed_out_grad),
        casted_index_desc.get(),
        static_cast<const int*>(GetBasePtr(&casted_index)),
        sum_out_desc.get(),
        GetBasePtr(&sum_out));
    std::vector<int> out_trans_perm;
    for (int i = 1; i < 1 + dim; ++i) {
      out_trans_perm.push_back(i);
    }
    out_trans_perm.push_back(0);
    for (int i = 1 + dim; i < x_dims.size(); ++i) {
      out_trans_perm.push_back(i);
    }
    dev_ctx.template Alloc<T>(x_grad);
    MLUCnnlTensorDesc x_grad_desc(*x_grad);
    const int out_trans_dim_size = out_trans_perm.size();
    MLUCnnl::Transpose(dev_ctx,
                       out_trans_perm,
                       out_trans_dim_size,
                       sum_out_desc.get(),
                       GetBasePtr(&sum_out),
                       x_grad_desc.get(),
                       GetBasePtr(x_grad));
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(index_select,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::IndexSelectKernel,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(index_select_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::IndexSelectGradKernel,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16) {}
