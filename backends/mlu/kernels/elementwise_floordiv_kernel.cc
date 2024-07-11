// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
void FloorDivideKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& y,
                       phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  MLUCnnlTensorDesc input_x_desc(x);
  MLUCnnlTensorDesc input_y_desc(y);
  MLUCnnlTensorDesc output_desc(*out);

  cnnlComputationPreference_t prefer = CNNL_COMPUTATION_HIGH_PRECISION;

  // when input x and input y dtype are int64
  // cast datatype to int32 for cnnlFloorDiv usage
  Tensor x_temp, y_temp, out_temp;
  x_temp.Resize(x.dims());
  y_temp.Resize(y.dims());
  out_temp.Resize(out->dims());
  if (x.dtype() != DataType::INT64 && y.dtype() != DataType::INT64) {
    MLUCnnl::FloorDiv(dev_ctx,
                      prefer,
                      input_x_desc.get(),
                      GetBasePtr(&x),
                      input_y_desc.get(),
                      GetBasePtr(&y),
                      output_desc.get(),
                      GetBasePtr(out));
  } else {
    dev_ctx.template Alloc<int32_t>(&x_temp);
    dev_ctx.template Alloc<int32_t>(&y_temp);
    dev_ctx.template Alloc<int32_t>(&out_temp);
    MLUCnnlTensorDesc x_temp_desc(x_temp);
    MLUCnnlTensorDesc y_temp_desc(y_temp);
    MLUCnnlTensorDesc out_temp_desc(out_temp);
    cnnlCastDataType_t cast_int32 = GetCastDataType(x.dtype(), DataType::INT32);

    MLUCnnl::Cast(dev_ctx,
                  cast_int32,
                  input_x_desc.get(),
                  GetBasePtr(&x),
                  x_temp_desc.get(),
                  GetBasePtr(&x_temp));

    MLUCnnl::Cast(dev_ctx,
                  cast_int32,
                  input_y_desc.get(),
                  GetBasePtr(&y),
                  y_temp_desc.get(),
                  GetBasePtr(&y_temp));

    MLUCnnl::FloorDiv(dev_ctx,
                      prefer,
                      x_temp_desc.get(),
                      GetBasePtr(&x_temp),
                      y_temp_desc.get(),
                      GetBasePtr(&y_temp),
                      out_temp_desc.get(),
                      GetBasePtr(&out_temp));

    cnnlCastDataType_t cast_int64 =
        GetCastDataType(x_temp.dtype(), DataType::INT64);

    MLUCnnl::Cast(dev_ctx,
                  cast_int64,
                  out_temp_desc.get(),
                  GetBasePtr(&out_temp),
                  output_desc.get(),
                  GetBasePtr(out));
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(floor_divide,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::FloorDivideKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16) {}
