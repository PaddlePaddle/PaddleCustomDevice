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

template <typename Context>
void MLULogicOp(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::DenseTensor& y,
                const std::string& logic_name,
                phi::DenseTensor* out) {
  dev_ctx.template Alloc<bool>(out);

  MLUCnnlTensorDesc input_x(x, CNNL_LAYOUT_ARRAY, ToCnnlDataType(x.dtype()));
  MLUCnnlTensorDesc input_y(y, CNNL_LAYOUT_ARRAY, ToCnnlDataType(y.dtype()));
  MLUCnnlTensorDesc output(
      *out, CNNL_LAYOUT_ARRAY, ToCnnlDataType(out->dtype()));

  cnnlLogicOp_t logic_op = GetMLUCnnlLogicOp(logic_name);
  if (x.dtype() != DataType::INT64) {
    MLUCnnl::Logic(dev_ctx,
                   logic_op,
                   input_x.get(),
                   GetBasePtr(&x),
                   input_y.get(),
                   GetBasePtr(&y),
                   output.get(),
                   GetBasePtr(out));
  } else {
    Tensor x_int32;
    Tensor y_int32;
    x_int32.Resize(x.dims());
    y_int32.Resize(y.dims());

    dev_ctx.template Alloc<int32_t>(&x_int32);
    dev_ctx.template Alloc<int32_t>(&y_int32);

    MLUCnnlTensorDesc input_x_int32(x_int32);
    MLUCnnlTensorDesc input_y_int32(y_int32);
    cnnlCastDataType_t cast_type =
        GetCastDataType(DataType::INT64, DataType::INT32);
    MLUCnnl::Cast(dev_ctx,
                  cast_type,
                  input_x.get(),
                  GetBasePtr(&x),
                  input_x_int32.get(),
                  GetBasePtr(&x_int32));
    MLUCnnl::Cast(dev_ctx,
                  cast_type,
                  input_y.get(),
                  GetBasePtr(&y),
                  input_y_int32.get(),
                  GetBasePtr(&y_int32));
    MLUCnnl::Logic(dev_ctx,
                   logic_op,
                   input_x_int32.get(),
                   GetBasePtr(&x_int32),
                   input_y_int32.get(),
                   GetBasePtr(&y_int32),
                   output.get(),
                   GetBasePtr(out));
  }
}

}  // namespace custom_kernel
