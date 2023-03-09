// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
void EmptyKernel(const Context& dev_ctx,
                 const phi::IntArray& shape,
                 phi::DataType dtype,
                 phi::DenseTensor* out) {
  out->Resize(phi::make_ddim(shape.GetData()));
  dev_ctx.template Alloc<T>(out);
}

template <typename T, typename Context>
void EmptyLikeKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     phi::DataType dtype,
                     phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(empty,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::EmptyKernel,
                          float,
                          double,
                          int8_t,
                          uint8_t,
                          int16_t,
                          int,
                          int64_t,
                          bool) {}

PD_REGISTER_PLUGIN_KERNEL(empty_like,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::EmptyLikeKernel,
                          float,
                          double,
                          int8_t,
                          uint8_t,
                          int16_t,
                          int,
                          int64_t,
                          bool) {}
