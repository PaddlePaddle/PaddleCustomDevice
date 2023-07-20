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
void EmbeddingKernel(const Context& dev_ctx,
                     const phi::DenseTensor& inputx,
                     const phi::DenseTensor& weight,
                     int64_t padding_idx,
                     phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  int padding_index = static_cast<int>(padding_idx);
  MLUCnnlTensorDesc ids_desc(inputx);
  MLUCnnlTensorDesc table_desc(weight);
  MLUCnnlTensorDesc output_desc(*out);

  MLUCnnl::EmbeddingForward(dev_ctx,
                            padding_index,
                            table_desc.get(),
                            GetBasePtr(&weight),
                            ids_desc.get(),
                            static_cast<const int*>(GetBasePtr(&inputx)),
                            output_desc.get(),
                            GetBasePtr(out));
}

template <typename T, typename Context>
void EmbeddingGradKernel(const Context& dev_ctx,
                         const phi::DenseTensor& input,
                         const phi::DenseTensor& weight,
                         const phi::DenseTensor& out_grad,
                         int64_t padding_idx,
                         phi::DenseTensor* weight_grad) {
  dev_ctx.template Alloc<T>(weight_grad);

  int padding_index = static_cast<int>(padding_idx);
  int64_t ids_numel = input.numel();
  PADDLE_ENFORCE_EQ(
      ids_numel <= std::numeric_limits<int32_t>::max(),
      true,
      phi::errors::OutOfRange(
          "Number of ids greater than int32_t::max , please check "
          "number of ids in LookupTableV2GradMLUKernel."));

  Tensor ids_int32;
  if (input.dtype() != DataType::INT32) {
    ids_int32.Resize(input.dims());
    dev_ctx.template Alloc<int>(&ids_int32);
    MLUCnnlTensorDesc ids_desc(input);
    MLUCnnlTensorDesc ids_int32_desc(ids_int32);
    auto cast_type = GetCastDataType(input.dtype(), DataType::INT32);
    MLUCnnl::Cast(dev_ctx,
                  cast_type,
                  ids_desc.get(),
                  GetBasePtr(&input),
                  ids_int32_desc.get(),
                  GetBasePtr(&ids_int32));
  } else {
    ids_int32 = input;
  }

  MLUCnnlTensorDesc ids_int32_desc(ids_int32);
  MLUCnnlTensorDesc output_grad_desc(out_grad);
  MLUCnnlTensorDesc table_grad_desc(*weight_grad);

  MLUCnnl::EmbeddingBackward(dev_ctx,
                             padding_index,
                             false,
                             ids_int32_desc.get(),
                             GetBasePtr(&ids_int32),
                             output_grad_desc.get(),
                             GetBasePtr(&out_grad),
                             table_grad_desc.get(),
                             GetBasePtr(weight_grad));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(embedding,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::EmbeddingKernel,
                          float,
                          int,
                          phi::dtype::float16) {}
PD_REGISTER_PLUGIN_KERNEL(embedding_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::EmbeddingGradKernel,
                          float,
                          int,
                          phi::dtype::float16) {}
