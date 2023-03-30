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

namespace custom_kernel {

template <typename T, typename Context>
void SplitKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const phi::IntArray& num_or_sections,
                 const phi::Scalar& axis_scalar,
                 std::vector<phi::DenseTensor*> outs) {
  // init parameter
  if (num_or_sections.FromTensor() || axis_scalar.FromTensor()) {
    std::vector<phi::MetaTensor> out_metas;
    out_metas.reserve(outs.size());
    std::vector<phi::MetaTensor*> out_metas_ptr;
    for (size_t i = 0; i < outs.size(); ++i) {
      out_metas.push_back(outs[i]);
      out_metas_ptr.push_back(&out_metas.back());
    }

    phi::SplitInferMeta(x, num_or_sections, axis_scalar, out_metas_ptr);

    for (size_t i = 0; i < out_metas.size(); ++i) {
      outs[i]->Resize(out_metas[i].dims());
    }
  }

  auto sections = num_or_sections.GetData();
  int num = static_cast<int>(sections.size());
  int axis = axis_scalar.to<int>();
  auto in_dims = x.dims();
  auto out_size = outs.size();
  auto num_tensor = num == 0 ? out_size : num;

  // init out tensors
  std::vector<void*> vct_tensor;
  std::vector<MLUCnnlTensorDesc> output_descs;
  std::vector<cnnlTensorDescriptor_t> desc_vector;
  for (size_t j = 0; j < outs.size(); ++j) {
    dev_ctx.template Alloc<T>(outs[j]);
    output_descs.emplace_back(MLUCnnlTensorDesc(
        *outs[j], CNNL_LAYOUT_ARRAY, ToCnnlDataType(outs[j]->dtype())));
    desc_vector.push_back(output_descs.back().get());
    vct_tensor.push_back(GetBasePtr(outs[j]));
  }

  // init in tensors
  MLUCnnlTensorDesc input_desc(x, CNNL_LAYOUT_ARRAY, ToCnnlDataType(x.dtype()));
  MLUCnnl::Split(dev_ctx,
                 num_tensor,
                 axis,
                 input_desc.get(),
                 GetBasePtr(&x),
                 desc_vector.data(),
                 vct_tensor.data());
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(split,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::SplitKernel,
                          float,
                          int64_t,
                          int,
                          bool,
                          phi::dtype::float16) {}
