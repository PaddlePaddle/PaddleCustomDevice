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
void OneHotRawKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::Scalar& depth_scalar,
                     phi::DataType dtype,
                     bool allow_out_of_range,
                     phi::DenseTensor* out) {
  int depth = depth_scalar.to<int>();
  auto out_dims = out->dims();
  out_dims[out_dims.size() - 1] = depth;
  out->Resize(out_dims);

  dev_ctx.template Alloc<float>(out);

  float on_value = 1.0f, off_value = 0.0f;
  std::vector<int> in_off_dim_vec(1, 1);
  phi::DDim in_out_dims = phi::make_ddim(in_off_dim_vec);

  Tensor on_value_tensor, off_value_tensor;
  on_value_tensor.Resize(in_out_dims);
  off_value_tensor.Resize(in_out_dims);
  dev_ctx.template Alloc<T>(&on_value_tensor);
  dev_ctx.template Alloc<T>(&off_value_tensor);
  FillMLUTensorWithHostValue(dev_ctx, on_value, &on_value_tensor);
  FillMLUTensorWithHostValue(dev_ctx, off_value, &off_value_tensor);
  if (x.dtype() == phi::DataType::INT32) {
    MLUCnnlTensorDesc desc_indices(x);
    MLUCnnl::OneHot(dev_ctx,
                    desc_indices.get(),
                    GetBasePtr(&x),
                    depth,
                    GetBasePtr(&on_value_tensor),
                    GetBasePtr(&off_value_tensor),
                    -1,
                    ToCnnlDataType(out->dtype()),
                    GetBasePtr(out));
  } else {
    Tensor transformed_in;
    transformed_in.Resize(x.dims());
    dev_ctx.template Alloc<int32_t>(&transformed_in);
    // use cnnlCast to cast int64_t to int32_t then do one_hot
    MLUCnnlTensorDesc in_desc(x);
    MLUCnnlTensorDesc transformed_in_desc(transformed_in);
    cnnlCastDataType_t cast_type = GetCastDataType(x.dtype(), DataType::INT32);
    MLUCnnl::Cast(dev_ctx,
                  cast_type,
                  in_desc.get(),
                  GetBasePtr(&x),
                  transformed_in_desc.get(),
                  GetBasePtr(&transformed_in));
    MLUCnnl::OneHot(dev_ctx,
                    transformed_in_desc.get(),
                    GetBasePtr(&transformed_in),
                    depth,
                    GetBasePtr(&on_value_tensor),
                    GetBasePtr(&off_value_tensor),
                    -1,
                    ToCnnlDataType(out->dtype()),
                    GetBasePtr(out));
  }
}

template <typename T, typename Context>
void OneHotKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::Scalar& num_classes_s,
                  phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  PADDLE_THROW(phi::errors::Unimplemented("OneHotKernel is not need?"));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(one_hot_raw,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::OneHotRawKernel,
                          int32_t,
                          int64_t) {}

PD_REGISTER_PLUGIN_KERNEL(one_hot,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::OneHotKernel,
                          int32_t,
                          int64_t) {}
