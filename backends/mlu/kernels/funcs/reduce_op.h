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

#pragma once

#include "kernels/funcs/mlu_baseop.h"
#include "kernels/funcs/mlu_funcs.h"

namespace custom_kernel {

template <typename T, typename Context>
void MLUReduceOp(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const std::vector<int64_t>& axes,
                 bool keep_dim,
                 bool reduce_all,
                 const std::string& reduce_name,
                 phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  if (x.dims().size() == 0) {
    TensorCopy(dev_ctx, x, true, out);
    return;
  }

  auto dims = axes;
  auto input_dims = phi::vectorize(x.dims());
  const auto& input_dim_size = x.dims().size();
  std::vector<int> reduce_dims;
  VLOG(3) << "ReduceOp keep_dim " << keep_dim;
  if (!keep_dim && dims.size() == 0) reduce_all = true;
  if (reduce_all) {
    for (size_t i = 0; i < input_dims.size(); i++) {
      reduce_dims.push_back(static_cast<int>(i));
    }
  } else {
    for (size_t i = 0; i < dims.size(); ++i) {
      if (dims[i] < 0) {
        reduce_dims.push_back(dims[i] + input_dim_size);
      } else {
        reduce_dims.push_back(dims[i]);
      }
    }
  }

  MLUCnnlTensorDesc input_desc(x, CNNL_LAYOUT_ARRAY, ToCnnlDataType(x.dtype()));
  MLUCnnlTensorDesc output_desc(
      *out, CNNL_LAYOUT_ARRAY, ToCnnlDataType(out->dtype()));

  cnnlReduceOp_t reduce_op = GetMLUCnnlReduceOp(reduce_name);
  MLUCnnlReduceDesc reduction_desc(reduce_dims,
                                   reduce_op,
                                   ToCnnlDataType<T>(),
                                   CNNL_NOT_PROPAGATE_NAN,
                                   CNNL_REDUCE_NO_INDICES,
                                   CNNL_32BIT_INDICES);

  MLUCnnl::Reduce(dev_ctx,
                  true /*need_workspace*/,
                  reduction_desc.get(),
                  nullptr,
                  input_desc.get(),
                  GetBasePtr(&x),
                  0 /*indices_size*/,
                  nullptr,
                  nullptr,
                  output_desc.get(),
                  GetBasePtr(out));
}

}  // namespace custom_kernel
