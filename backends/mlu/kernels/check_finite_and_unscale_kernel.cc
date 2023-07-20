/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "kernels/funcs/mlu_baseop.h"
#include "kernels/funcs/mlu_funcs.h"

namespace custom_kernel {

template <typename T, typename Context>
void CheckFiniteAndUnscale(const Context& dev_ctx,
                           const std::vector<const phi::DenseTensor*>& xs,
                           const phi::DenseTensor& t_scale,
                           std::vector<phi::DenseTensor*> outs,
                           phi::DenseTensor* found_inf) {
  using MPDType = typename MPTypeTrait<T>::Type;
  dev_ctx.template Alloc<bool>(found_inf);

  MLUCnnlTensorDesc scale_desc(t_scale);
  MLUCnnlTensorDesc found_inf_desc(
      *found_inf, CNNL_LAYOUT_ARRAY, ToCnnlDataType<bool>());

  for (size_t i = 0; i < xs.size(); ++i) {
    const auto* x = xs[i];
    auto* out = outs[i];
    dev_ctx.template Alloc<T>(out);

    // check is_finite or is_nan
    Tensor is_finite;
    if (i != 0) {
      is_finite.Resize(phi::make_ddim({1}));
      dev_ctx.template Alloc<bool>(&is_finite);
    } else {
      is_finite = *found_inf;
    }

    MLUCnnlTensorDesc x_desc(*x);
    MLUCnnlTensorDesc out_desc(*out);

    MLUCnnl::IsNanInf(
        dev_ctx, x_desc.get(), GetBasePtr(x), GetBasePtr(&is_finite));

    // save is_finite by logical_and op after checking every input
    if (i != 0) {
      MLUCnnlTensorDesc is_finite_desc(
          is_finite, CNNL_LAYOUT_ARRAY, ToCnnlDataType<bool>());
      MLUCnnl::Logic(dev_ctx,
                     CNNL_LOGIC_OP_OR,
                     found_inf_desc.get(),
                     GetBasePtr(found_inf),
                     is_finite_desc.get(),
                     GetBasePtr(&is_finite),
                     found_inf_desc.get(),
                     GetBasePtr(found_inf));
    }

    // The normal logic is :
    // out = in, if found_inf = true
    // out = in/scale, if found_inf = false
    // But when found_inf is true, the data of Out should not be used.
    // So, on MLU, we always compute out with in/scale.
    Tensor float_x;
    Tensor float_out;
    if (std::is_same<T, phi::dtype::float16>::value) {
      float_x.Resize(x->dims());
      float_out.Resize(out->dims());
      dev_ctx.template Alloc<MPDType>(&float_x);
      dev_ctx.template Alloc<MPDType>(&float_out);

      MLUCnnlTensorDesc float_x_desc(float_x);
      MLUCnnlTensorDesc float_out_desc(float_out);
      auto cast_fp16_type =
          GetCastDataType(DataType::FLOAT16, DataType::FLOAT32);
      MLUCnnl::Cast(dev_ctx,
                    cast_fp16_type,
                    x_desc.get(),
                    GetBasePtr(x),
                    float_x_desc.get(),
                    GetBasePtr(&float_x));

      MLUCnnl::Div(dev_ctx,
                   CNNL_COMPUTATION_HIGH_PRECISION,
                   float_x_desc.get(),
                   GetBasePtr(&float_x),
                   scale_desc.get(),
                   GetBasePtr(&t_scale),
                   float_out_desc.get(),
                   GetBasePtr(&float_out));

      auto cast_fp32_type =
          GetCastDataType(DataType::FLOAT32, DataType::FLOAT16);
      MLUCnnl::Cast(dev_ctx,
                    cast_fp32_type,
                    float_out_desc.get(),
                    GetBasePtr(&float_out),
                    out_desc.get(),
                    GetBasePtr(out));
    } else {
      MLUCnnl::Div(dev_ctx,
                   CNNL_COMPUTATION_HIGH_PRECISION,
                   x_desc.get(),
                   GetBasePtr(x),
                   scale_desc.get(),
                   GetBasePtr(&t_scale),
                   out_desc.get(),
                   GetBasePtr(out));
    }
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(check_finite_and_unscale,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::CheckFiniteAndUnscale,
                          float,
                          phi::dtype::float16) {}
