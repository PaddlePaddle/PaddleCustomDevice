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
void ClipKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::Scalar& min,
                const phi::Scalar& max,
                phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  auto min_val = min.to<T>();
  auto max_val = max.to<T>();

  MLUCnnlTensorDesc x_desc(x);
  MLUCnnlTensorDesc out_desc(*out);
  MLUCnnl::Clip(dev_ctx,
                x_desc.get(),
                GetBasePtr(&x),
                static_cast<const void*>(&min_val),
                static_cast<const void*>(&max_val),
                out_desc.get(),
                GetBasePtr(out));
}

template <typename T, typename Context>
void ClipGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& dout,
                    const phi::Scalar& min,
                    const phi::Scalar& max,
                    phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);

  auto min_val = min.to<T>();
  auto max_val = max.to<T>();

  MLUCnnlTensorDesc x_desc(x);
  MLUCnnlTensorDesc dx_desc(*dx);
  MLUCnnlTensorDesc dout_desc(dout);

  MLUCnnl::HardtanhBackward(dev_ctx,
                            x_desc.get(),
                            GetBasePtr(&x),
                            dout_desc.get(),
                            GetBasePtr(&dout),
                            max_val,
                            min_val,
                            dx_desc.get(),
                            GetBasePtr(dx));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(clip,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::ClipKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(clip_grad,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::ClipGradKernel,
                          float,
                          phi::dtype::float16) {}
