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
void TopkKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::Scalar& k_scalar,
                int axis,
                bool largest,
                bool sorted,
                phi::DenseTensor* out,
                phi::DenseTensor* indices) {
  if (axis < 0) {
    axis += x.dims().size();
  }
  int k = k_scalar.to<int>();
  phi::DDim output_dims = x.dims();
  output_dims[axis] = k;

  out->Resize(output_dims);
  indices->Resize(output_dims);

  dev_ctx.template Alloc<T>(out);
  dev_ctx.template Alloc<int64_t>(indices);

  // Support 0D
  if (output_dims.size() == 0) {
    TensorCopy(dev_ctx, x, true, out);
    int64_t* indices_data = dev_ctx.template Alloc<int64_t>(indices);
    indices_data[0] = 0;
    return;
  }

  phi::DenseTensor indices_int32;
  phi::DenseTensorMeta indices_int32_meta = {phi::DataType::INT32,
                                             indices->dims()};
  indices_int32.set_meta(indices_int32_meta);
  dev_ctx.template Alloc<int32_t>(&indices_int32);
  MLUCnnlTensorDesc input_desc(x);
  MLUCnnlTensorDesc values_output_desc(*out);
  MLUCnnlTensorDesc indices_int32_desc(indices_int32);
  MLUCnnl::TopK(dev_ctx,
                k,
                axis,
                largest,
                sorted,
                input_desc.get(),
                GetBasePtr(&x),
                values_output_desc.get(),
                GetBasePtr(out),
                indices_int32_desc.get(),
                GetBasePtr(&indices_int32));

  // cast indices type to int64
  MLUCnnlTensorDesc cast_output_desc(*indices);
  cnnlCastDataType_t cast_type =
      GetCastDataType(DataType::INT32, DataType::INT64);
  MLUCnnl::Cast(dev_ctx,
                cast_type,
                indices_int32_desc.get(),
                GetBasePtr(&indices_int32),
                cast_output_desc.get(),
                GetBasePtr(indices));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(topk,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::TopkKernel,
                          phi::dtype::float16,
                          float) {
  kernel->OutputAt(1).SetDataType(phi::DataType::INT64);
}
