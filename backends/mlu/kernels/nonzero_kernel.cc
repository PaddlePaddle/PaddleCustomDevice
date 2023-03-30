/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <vector>

#include "kernels/funcs/mlu_baseop.h"
#include "kernels/funcs/mlu_funcs.h"

namespace custom_kernel {

template <typename T, typename Context>
void NonZeroKernel(const Context& dev_ctx,
                   const phi::DenseTensor& condition,
                   phi::DenseTensor* out) {
  auto dims = condition.dims();
  const int rank = dims.size();

  Tensor num_true;
  num_true.Resize({1});
  dev_ctx.template Alloc<int>(&num_true);
  MLUCnnlTensorDesc con_desc(condition);
  MLUCnnlTensorDesc num_true_desc(num_true);
  MLUCnnl::NumTrue(dev_ctx,
                   con_desc.get(),
                   GetBasePtr(&condition),
                   num_true_desc.get(),
                   GetBasePtr(&num_true));

  Tensor local_true_num;
  dev_ctx.Wait();  // add sync for fully calculated results
  TensorCopy(dev_ctx, num_true, true, &local_true_num, phi::CPUPlace());
  auto true_num = *local_true_num.data<int>();

  out->Resize(phi::make_ddim({true_num, rank}));
  dev_ctx.template Alloc<int64_t>(out);

  if (true_num == 0) {
    return;
  }

  Tensor out_int32;
  out_int32.Resize(out->dims());
  dev_ctx.template Alloc<int32_t>(&out_int32);
  MLUCnnlTensorDesc out_int32_desc(out_int32);
  MLUCnnlTensorDesc out_desc(*out);
  bool as_tuple = false;
  MLUCnnl::Where(dev_ctx,
                 con_desc.get(),
                 GetBasePtr(&condition),
                 num_true_desc.get(),
                 GetBasePtr(&num_true),
                 as_tuple,
                 out_int32_desc.get(),
                 GetBasePtr(&out_int32));
  cnnlCastDataType_t cast_type =
      GetCastDataType(DataType::INT32, DataType::INT64);
  MLUCnnl::Cast(dev_ctx,
                cast_type,
                out_int32_desc.get(),
                GetBasePtr(&out_int32),
                out_desc.get(),
                GetBasePtr(out));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    nonzero, mlu, ALL_LAYOUT, custom_kernel::NonZeroKernel, bool, int, float) {}
