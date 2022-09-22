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

const int kIgnoreIndex = -100;

void CheckAttrs(bool normalize, int ignore_index) {
  // Add this check is is due to Ascend SigmoidCrossEntropyWithLogits
  // and SigmoidCrossEntropyWithLogitsGrad does't supoort
  // attr normalize and ignore_index
  PADDLE_ENFORCE_EQ(normalize,
                    false,
                    phi::errors::InvalidArgument(
                        "attr normalize must be false, but got true"));
  PADDLE_ENFORCE_EQ(ignore_index,
                    kIgnoreIndex,
                    phi::errors::InvalidArgument(
                        "attr ignore_index must be default %d, but got %d",
                        kIgnoreIndex,
                        ignore_index));
}

template <typename T, typename Context>
void SigmoidCrossEntropyWithLogitsKernel(const Context& dev_ctx,
                                         const phi::DenseTensor& x,
                                         const phi::DenseTensor& label,
                                         bool normalize,
                                         int ignore_index,
                                         phi::DenseTensor* out) {
  CheckAttrs(normalize, ignore_index);

  dev_ctx.template Alloc<T>(out);
  MLUCnnlTensorDesc x_desc(x);
  MLUCnnlTensorDesc label_desc(label);
  MLUCnnlTensorDesc out_desc(*out);
  MLUCnnl::BceWithLogits(dev_ctx,
                          CNNL_BCE_WITH_LOGITS_NONE,
                          x_desc.get(),
                          GetBasePtr(&x),
                          label_desc.get(),
                          GetBasePtr(&label),
                          nullptr,
                          nullptr,
                          nullptr,
                          nullptr,
                          out_desc.get(),
                          GetBasePtr(out));
}

template <typename T, typename Context>
void SigmoidCrossEntropyWithLogitsGradKernel(const Context& dev_ctx,
                                             const phi::DenseTensor& x,
                                             const phi::DenseTensor& label,
                                             const phi::DenseTensor& dout,
                                             bool normalize,
                                             int ignore_index,
                                             phi::DenseTensor* dx) {
  CheckAttrs(normalize, ignore_index);

  dev_ctx.template Alloc<T>(dx);
  MLUCnnlTensorDesc x_desc(x);
  MLUCnnlTensorDesc label_desc(label);
  MLUCnnlTensorDesc dout_desc(dout);
  MLUCnnl::BceWithLogitsBackward(dev_ctx,
                                  CNNL_BCE_WITH_LOGITS_NONE,
                                  dout_desc.get(),
                                  GetBasePtr(&dout),
                                  x_desc.get(),
                                  GetBasePtr(&x),
                                  label_desc.get(),
                                  GetBasePtr(&label),
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  x_desc.get(),
                                  GetBasePtr(dx));

}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(sigmoid_cross_entropy_with_logits,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::SigmoidCrossEntropyWithLogitsKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(
    sigmoid_cross_entropy_with_logits_grad,
    CustomMLU,
    ALL_LAYOUT,
    custom_kernel::SigmoidCrossEntropyWithLogitsGradKernel,
    float,
    phi::dtype::float16) {}
