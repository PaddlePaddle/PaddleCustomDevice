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
void AddNKernel(const Context& dev_ctx,
                const std::vector<const phi::DenseTensor*>& x,
                phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  int n = static_cast<int>(x.size());
  if (n == 1) {
    TensorCopy(dev_ctx, *x[0], false, out);
    return;
  }

  // MLU shoul do sth
  std::vector<const void*> inputs;
  std::vector<MLUCnnlTensorDesc> input_descs;
  std::vector<cnnlTensorDescriptor_t> desc_vector;
  for (int i = 0; i < n; i++) {
    input_descs.emplace_back(MLUCnnlTensorDesc(
        *x[i], CNNL_LAYOUT_ARRAY, ToCnnlDataType(x[i]->dtype())));
    desc_vector.push_back(input_descs.back().get());
    inputs.push_back(GetBasePtr(x[i]));
  }
  // init out tensors
  MLUCnnlTensorDesc output_desc(
      *out, CNNL_LAYOUT_ARRAY, ToCnnlDataType(out->dtype()));
  uint32_t ins_size_t = static_cast<uint32_t>(n);
  MLUCnnl::AddN(dev_ctx,
                ins_size_t,
                desc_vector.data(),
                inputs.data(),
                output_desc.get(),
                GetBasePtr(out));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(add_n,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::AddNKernel,
                          float,
                          phi::dtype::float16,
                          double) {}
