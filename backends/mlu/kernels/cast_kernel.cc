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

#include "kernels/funcs/mlu_funcs.h"

namespace custom_kernel {

template <typename T, typename Context>
void CastKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensorMeta::DataType dtype,
                phi::DenseTensor* out) {
  if (x.dtype() == dtype) {
    dev_ctx.template Alloc<T>(out);
    TensorCopy(dev_ctx, x, false, out);
    return;
  }

  PADDLE_ENFORCE_EQ(MLUSupportsCast(x.dtype(), dtype),
                    true,
                    phi::errors::InvalidArgument(
                        "MLU not support cast [%d] to [%d]", x.dtype(), dtype));
  dev_ctx.Alloc(out, dtype);

  MLUCnnlTensorDesc x_desc(x);
  MLUCnnlTensorDesc out_desc(*out);
  cnnlCastDataType_t cast_type = GetCastDataType(x.dtype(), dtype);
  MLUCnnl::Cast(dev_ctx,
                cast_type,
                x_desc.get(),
                GetBasePtr(&x),
                out_desc.get(),
                GetBasePtr(out));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(cast,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::CastKernel,
                          phi::dtype::float16,
                          float,
                          double,
                          uint8_t,
                          int8_t,
                          int16_t,
                          int32_t,
                          int64_t,
                          bool) {}
