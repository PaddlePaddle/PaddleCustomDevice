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
void ArgsortKernel(const Context& dev_ctx,
                   const phi::DenseTensor& in,
                   int axis,
                   bool descending,
                   phi::DenseTensor* output,
                   phi::DenseTensor* indices) {
  const auto& sorted = true;
  // axis < 0, cacluate the real axis
  if (axis < 0) {
    const auto& in_dims = in.dims();
    axis += in_dims.size();
  }

  auto in_dims = in.dims();
  auto rank = in_dims.size();
  size_t k = in_dims[axis];

  dev_ctx.template Alloc<T>(output);
  dev_ctx.template Alloc<int64_t>(indices);

  if (rank == 0) {
    TensorCopy(dev_ctx, in, false, output);
    FillMLUTensorWithHostValue(dev_ctx, 0, indices);
    return;
  }

  // cnnl only support int32/int16 type of indices
  Tensor indices_int32;
  indices_int32.Resize(indices->dims());
  dev_ctx.template Alloc<int32_t>(&indices_int32);

  MLUCnnlTensorDesc input_desc(in);
  MLUCnnlTensorDesc values_output_desc(*output);
  MLUCnnlTensorDesc indices_int32_desc(indices_int32);
  MLUCnnl::TopK(dev_ctx,
                k,
                axis,
                descending,
                sorted,
                input_desc.get(),
                GetBasePtr(&in),
                values_output_desc.get(),
                GetBasePtr(output),
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

template <typename T, typename Context>
void ArgsortGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& indices,
                       const phi::DenseTensor& input,
                       const phi::DenseTensor& out_grad,
                       int axis,
                       bool descending,
                       phi::DenseTensor* in_grad) {
  dev_ctx.template Alloc<T>(in_grad);

  auto in_dims = indices.dims();
  auto rank = input.dims().size();
  axis = (axis < 0) ? (in_dims.size() + axis) : axis;
  FillMLUTensorWithHostValue(dev_ctx, 0., in_grad);
  if (out_grad.numel() == 0) return;

  if (rank == 0) {
    FillMLUTensorWithHostValue(dev_ctx, 1., in_grad);
    return;
  }

  MLUCnnlTensorDesc dout_desc(out_grad);
  MLUCnnlTensorDesc indices_desc(indices);
  MLUCnnlTensorDesc dx_desc(*in_grad);
  MLUCnnl::ScatterFunctor(dev_ctx,
                          dx_desc.get(),
                          GetBasePtr(in_grad),
                          dout_desc.get(),
                          GetBasePtr(&out_grad),
                          indices_desc.get(),
                          GetBasePtr(&indices),
                          axis);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(argsort,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::ArgsortKernel,
                          float,
                          int8_t,
                          uint8_t,
                          int16_t,
                          int32_t,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(argsort_grad,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::ArgsortGradKernel,
                          float,
                          int8_t,
                          uint8_t,
                          int16_t,
                          int32_t,
                          phi::dtype::float16) {}
