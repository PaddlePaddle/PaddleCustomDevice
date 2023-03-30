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
void ScatterKernel(const Context &dev_ctx,
                   const phi::DenseTensor &x,
                   const phi::DenseTensor &index,
                   const phi::DenseTensor &updates,
                   bool overwrite,
                   phi::DenseTensor *out) {
  dev_ctx.template Alloc<T>(out);
  MLUCnnlTensorDesc x_desc(x);
  MLUCnnlTensorDesc index_desc(index);
  MLUCnnlTensorDesc updates_desc(updates);
  MLUCnnlTensorDesc out_desc(*out);
  cnnlScatterRefMode_t mode;
  if (overwrite) {
    mode = CNNL_SCATTERREF_UPDATE;
    MLUCnnl::ScatterRefFunctor(dev_ctx,
                               x_desc.get(),
                               GetBasePtr(&x),
                               updates_desc.get(),
                               GetBasePtr(&updates),
                               index_desc.get(),
                               GetBasePtr(&index),
                               mode);

  } else {
    Tensor tensor_zeros;
    tensor_zeros.Resize(updates.dims());
    dev_ctx.template Alloc<T>(&tensor_zeros);
    MLUCnnlTensorDesc tensor_zeros_desc(tensor_zeros);
    float value = 0.0;
    auto value_t = static_cast<T>(value);
    MLUCnnl::Fill(dev_ctx,
                  CNNL_POINTER_MODE_HOST,
                  &value_t,
                  tensor_zeros_desc.get(),
                  GetBasePtr(&tensor_zeros));

    mode = CNNL_SCATTERREF_UPDATE;
    MLUCnnl::ScatterRefFunctor(dev_ctx,
                               x_desc.get(),
                               GetBasePtr(&x),
                               tensor_zeros_desc.get(),
                               GetBasePtr(&tensor_zeros),
                               index_desc.get(),
                               GetBasePtr(&index),
                               mode);
    mode = CNNL_SCATTERREF_ADD;
    MLUCnnl::ScatterRefFunctor(dev_ctx,
                               x_desc.get(),
                               GetBasePtr(&x),
                               updates_desc.get(),
                               GetBasePtr(&updates),
                               index_desc.get(),
                               GetBasePtr(&index),
                               mode);
  }
  TensorCopy(dev_ctx, x, false, out);
}

template <typename T, typename Context>
void ScatterNdAddKernel(const Context &dev_ctx,
                        const phi::DenseTensor &x,
                        const phi::DenseTensor &index,
                        const phi::DenseTensor &updates,
                        phi::DenseTensor *out) {
  dev_ctx.template Alloc<T>(out);
  cnnlScatterNdMode_t mode = CNNL_SCATTERND_ADD;
  const auto &index_type = index.dtype();
  bool index_type_match =
      index_type == phi::DataType::INT32 || index_type == phi::DataType::INT64;
  PADDLE_ENFORCE_EQ(index_type_match,
                    true,
                    phi::errors::InvalidArgument(
                        "Index holds the wrong type, it holds [%s], but "
                        "desires to be [%s] or [%s].",
                        index_type,
                        phi::DataType::INT32,
                        phi::DataType::INT64));
  MLUCnnlTensorDesc x_desc(x);
  MLUCnnlTensorDesc index_desc(index);
  MLUCnnlTensorDesc updates_desc(updates);
  MLUCnnlTensorDesc out_desc(*out);

  MLUCnnl::ScatterNd(dev_ctx,
                     mode,
                     index_desc.get(),
                     GetBasePtr(&index),
                     updates_desc.get(),
                     GetBasePtr(&updates),
                     x_desc.get(),
                     GetBasePtr(&x),
                     out_desc.get(),
                     GetBasePtr(out));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(scatter,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::ScatterKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(scatter_nd_add,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::ScatterNdAddKernel,
                          float,
                          phi::dtype::float16,
                          int) {}
