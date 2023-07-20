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
void EqualRawKernel(const Context& dev_ctx,
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
void EqualKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const phi::DenseTensor& y,
                 phi::DenseTensor* out) {
  custom_kernel::EqualRawKernel<T, Context>(dev_ctx, x, y, -1, out);
}

template <typename T, typename Context>
void NotEqualRawKernel(const Context& dev_ctx,
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
void NotEqualKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    phi::DenseTensor* out) {
  custom_kernel::NotEqualRawKernel<T, Context>(dev_ctx, x, y, -1, out);
}

template <typename T, typename Context>
void LessThanRawKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& y,
                       int axis,
                       phi::DenseTensor* out) {
  dev_ctx.template Alloc<bool>(out);

  MLUCnnlTensorDesc input_x(x, CNNL_LAYOUT_ARRAY, ToCnnlDataType(x.dtype()));
  MLUCnnlTensorDesc input_y(y, CNNL_LAYOUT_ARRAY, ToCnnlDataType(y.dtype()));
  MLUCnnlTensorDesc output(
      *out, CNNL_LAYOUT_ARRAY, ToCnnlDataType(out->dtype()));

  if (x.dtype() != DataType::INT64) {
    MLUCnnl::Logic(dev_ctx,
                   CNNL_LOGIC_OP_LT,
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
                   CNNL_LOGIC_OP_LT,
                   input_x_int32.get(),
                   GetBasePtr(&x_int32),
                   input_y_int32.get(),
                   GetBasePtr(&y_int32),
                   output.get(),
                   GetBasePtr(out));
  }
}

template <typename T, typename Context>
void LessThanKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    phi::DenseTensor* out) {
  custom_kernel::LessThanRawKernel<T, Context>(dev_ctx, x, y, -1, out);
}

template <typename T, typename Context>
void LessEqualRawKernel(const Context& dev_ctx,
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
void LessEqualKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& y,
                     phi::DenseTensor* out) {
  custom_kernel::LessEqualRawKernel<T, Context>(dev_ctx, x, y, -1, out);
}

template <typename T, typename Context>
void GreaterThanRawKernel(const Context& dev_ctx,
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
void GreaterThanKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& y,
                       phi::DenseTensor* out) {
  custom_kernel::GreaterThanRawKernel<T, Context>(dev_ctx, x, y, -1, out);
}

template <typename T, typename Context>
void GreaterEqualRawKernel(const Context& dev_ctx,
                           const phi::DenseTensor& x,
                           const phi::DenseTensor& y,
                           int axis,
                           phi::DenseTensor* out) {
  dev_ctx.template Alloc<bool>(out);

  MLUCnnlTensorDesc input_x(x, CNNL_LAYOUT_ARRAY, ToCnnlDataType(x.dtype()));
  MLUCnnlTensorDesc input_y(y, CNNL_LAYOUT_ARRAY, ToCnnlDataType(y.dtype()));
  MLUCnnlTensorDesc output(
      *out, CNNL_LAYOUT_ARRAY, ToCnnlDataType(out->dtype()));

  if (x.dtype() != DataType::INT64) {
    MLUCnnl::Logic(dev_ctx,
                   CNNL_LOGIC_OP_GE,
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
                   CNNL_LOGIC_OP_GE,
                   input_x_int32.get(),
                   GetBasePtr(&x_int32),
                   input_y_int32.get(),
                   GetBasePtr(&y_int32),
                   output.get(),
                   GetBasePtr(out));
  }
}

template <typename T, typename Context>
void GreaterEqualKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& y,
                        phi::DenseTensor* out) {
  custom_kernel::GreaterEqualRawKernel<T, Context>(dev_ctx, x, y, -1, out);
}

}  // namespace custom_kernel

#define PD_REGISTER_COMPARE_KERNEL(name, func)              \
  PD_REGISTER_PLUGIN_KERNEL(name,                           \
                            mlu,                            \
                            ALL_LAYOUT,                     \
                            custom_kernel::func##Kernel,    \
                            bool,                           \
                            int16_t,                        \
                            int,                            \
                            float,                          \
                            phi::dtype::float16) {          \
    kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);   \
  }                                                         \
  PD_REGISTER_PLUGIN_KERNEL(name##_raw,                     \
                            mlu,                            \
                            ALL_LAYOUT,                     \
                            custom_kernel::func##RawKernel, \
                            bool,                           \
                            int16_t,                        \
                            int,                            \
                            float,                          \
                            phi::dtype::float16) {          \
    kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);   \
  }

PD_REGISTER_COMPARE_KERNEL(less_equal, LessEqual)
PD_REGISTER_COMPARE_KERNEL(greater_than, GreaterThan)
PD_REGISTER_COMPARE_KERNEL(equal, Equal)
PD_REGISTER_COMPARE_KERNEL(not_equal, NotEqual)

PD_REGISTER_PLUGIN_KERNEL(less_than,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::LessThanKernel,
                          bool,
                          int16_t,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_REGISTER_PLUGIN_KERNEL(less_than_raw,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::LessThanRawKernel,
                          bool,
                          int16_t,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_REGISTER_PLUGIN_KERNEL(greater_equal,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::GreaterEqualKernel,
                          bool,
                          int16_t,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_REGISTER_PLUGIN_KERNEL(greater_equal_raw,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::GreaterEqualRawKernel,
                          bool,
                          int16_t,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}
