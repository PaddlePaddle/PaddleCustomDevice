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
void StackKernel(const Context& dev_ctx,
                 const std::vector<const phi::DenseTensor*>& x,
                 int axis,
                 phi::DenseTensor* y) {
  if (axis < 0) axis += (x[0]->dims().size() + 1);
  int num = static_cast<int>(x.size());

  PADDLE_ENFORCE_GT(
      num, 0, phi::errors::InvalidArgument("number of input Tensor <= 0"));

  std::vector<MLUCnnlTensorDesc> x_descs;
  std::vector<cnnlTensorDescriptor_t> x_raw_descs;
  std::vector<const void*> x_ptrs;
  for (int i = 0; i < num; i++) {
    if (x[i]->dims().size() != 0) {
      std::vector<int64_t> in_dims = phi::vectorize(x[i]->dims());
      in_dims.insert(in_dims.begin() + axis, 1);
      x_descs.emplace_back(MLUCnnlTensorDesc(
          in_dims.size(), in_dims.data(), ToCnnlDataType<T>()));
    } else {
      int input_dims = 1;
      x_descs.emplace_back(
          MLUCnnlTensorDesc(1, &input_dims, ToCnnlDataType<T>()));
    }
    x_raw_descs.push_back(x_descs.back().get());
    x_ptrs.push_back(GetBasePtr(x[i]));
  }
  dev_ctx.template Alloc<T>(y);

  MLUCnnlTensorDesc y_desc(*y);
  MLUCnnl::Concat(dev_ctx,
                  num,
                  axis,
                  x_raw_descs.data(),
                  x_ptrs.data(),
                  y_desc.get(),
                  GetBasePtr(y));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(stack,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::StackKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16) {}
