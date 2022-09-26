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
void EqualKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const phi::DenseTensor& y,
                 int axis,
                 phi::DenseTensor* out) {
  dev_ctx.template Alloc<bool>(out);

  MLUCnnlTensorDesc input_x(x, CNNL_LAYOUT_ARRAY, ToCnnlDataType(x.dtype()));
  MLUCnnlTensorDesc input_y(y, CNNL_LAYOUT_ARRAY, ToCnnlDataType(y.dtype()));
  MLUCnnlTensorDesc output(
      *out, CNNL_LAYOUT_ARRAY, ToCnnlDataType(out->dtype()));
  MLUCnnl::Logic(dev_ctx,
                 CNNL_LOGIC_OP_EQ,
                 input_x.get(),
                 GetBasePtr(&x),
                 input_y.get(),
                 GetBasePtr(&y),
                 output.get(),
                 GetBasePtr(out));
}

template <typename T, typename Context>
void NotEqualKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    int axis,
                    phi::DenseTensor* out) {
  dev_ctx.template Alloc<bool>(out);

  MLUCnnlTensorDesc input_x(x, CNNL_LAYOUT_ARRAY, ToCnnlDataType(x.dtype()));
  MLUCnnlTensorDesc input_y(y, CNNL_LAYOUT_ARRAY, ToCnnlDataType(y.dtype()));
  MLUCnnlTensorDesc output(
      *out, CNNL_LAYOUT_ARRAY, ToCnnlDataType(out->dtype()));
  MLUCnnl::Logic(dev_ctx,
                 CNNL_LOGIC_OP_NE,
                 input_x.get(),
                 GetBasePtr(&x),
                 input_y.get(),
                 GetBasePtr(&y),
                 output.get(),
                 GetBasePtr(out));
}

template <typename T, typename Context>
void LessThanKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    int axis,
                    phi::DenseTensor* out) {
  dev_ctx.template Alloc<bool>(out);

  MLUCnnlTensorDesc input_x(x, CNNL_LAYOUT_ARRAY, ToCnnlDataType(x.dtype()));
  MLUCnnlTensorDesc input_y(y, CNNL_LAYOUT_ARRAY, ToCnnlDataType(y.dtype()));
  MLUCnnlTensorDesc output(
      *out, CNNL_LAYOUT_ARRAY, ToCnnlDataType(out->dtype()));
  MLUCnnl::Logic(dev_ctx,
                 CNNL_LOGIC_OP_LT,
                 input_x.get(),
                 GetBasePtr(&x),
                 input_y.get(),
                 GetBasePtr(&y),
                 output.get(),
                 GetBasePtr(out));
}

template <typename T, typename Context>
void LessEqualKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& y,
                     int axis,
                     phi::DenseTensor* out) {
  dev_ctx.template Alloc<bool>(out);

  MLUCnnlTensorDesc input_x(x, CNNL_LAYOUT_ARRAY, ToCnnlDataType(x.dtype()));
  MLUCnnlTensorDesc input_y(y, CNNL_LAYOUT_ARRAY, ToCnnlDataType(y.dtype()));
  MLUCnnlTensorDesc output(
      *out, CNNL_LAYOUT_ARRAY, ToCnnlDataType(out->dtype()));
  MLUCnnl::Logic(dev_ctx,
                 CNNL_LOGIC_OP_LE,
                 input_x.get(),
                 GetBasePtr(&x),
                 input_y.get(),
                 GetBasePtr(&y),
                 output.get(),
                 GetBasePtr(out));
}

template <typename T, typename Context>
void GreaterThanKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& y,
                       int axis,
                       phi::DenseTensor* out) {
  dev_ctx.template Alloc<bool>(out);

  MLUCnnlTensorDesc input_x(x, CNNL_LAYOUT_ARRAY, ToCnnlDataType(x.dtype()));
  MLUCnnlTensorDesc input_y(y, CNNL_LAYOUT_ARRAY, ToCnnlDataType(y.dtype()));
  MLUCnnlTensorDesc output(
      *out, CNNL_LAYOUT_ARRAY, ToCnnlDataType(out->dtype()));
  MLUCnnl::Logic(dev_ctx,
                 CNNL_LOGIC_OP_GT,
                 input_x.get(),
                 GetBasePtr(&x),
                 input_y.get(),
                 GetBasePtr(&y),
                 output.get(),
                 GetBasePtr(out));
}

template <typename T, typename Context>
void GreaterEqualKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& y,
                        int axis,
                        phi::DenseTensor* out) {
  dev_ctx.template Alloc<bool>(out);

  MLUCnnlTensorDesc input_x(x, CNNL_LAYOUT_ARRAY, ToCnnlDataType(x.dtype()));
  MLUCnnlTensorDesc input_y(y, CNNL_LAYOUT_ARRAY, ToCnnlDataType(y.dtype()));
  MLUCnnlTensorDesc output(
      *out, CNNL_LAYOUT_ARRAY, ToCnnlDataType(out->dtype()));
  MLUCnnl::Logic(dev_ctx,
                 CNNL_LOGIC_OP_GE,
                 input_x.get(),
                 GetBasePtr(&x),
                 input_y.get(),
                 GetBasePtr(&y),
                 output.get(),
                 GetBasePtr(out));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(equal,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::EqualKernel,
                          bool,
                          int16_t,
                          int,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(not_equal,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::NotEqualKernel,
                          bool,
                          int16_t,
                          int,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(less_than,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::LessThanKernel,
                          bool,
                          int16_t,
                          int,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(less_equal,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::LessEqualKernel,
                          bool,
                          int16_t,
                          int,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(greater_than,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::GreaterThanKernel,
                          bool,
                          int16_t,
                          int,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(greater_equal,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::GreaterEqualKernel,
                          bool,
                          int16_t,
                          int,
                          float,
                          phi::dtype::float16) {}
