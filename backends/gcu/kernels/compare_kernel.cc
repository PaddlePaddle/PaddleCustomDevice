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

#include "kernels/common_ops/common_ops.h"
#include "kernels/common_ops/elementwise_ops.h"
#include "kernels/funcs/gcu_kernel_funcs.h"
#include "kernels/funcs/gcu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void CompareBaseKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& y,
                       int axis,
                       phi::DenseTensor* out,
                       const std::string& op_type) {
  dev_ctx.template Alloc<bool>(out);

  TensorNameMap input_names;
  input_names["X"] = {"x"};
  input_names["Y"] = {"y"};

  TensorValueMap inputs;
  inputs["X"] = {const_cast<DenseTensor*>(&x)};
  inputs["Y"] = {const_cast<DenseTensor*>(&y)};

  TensorNameMap output_names;
  output_names["Out"] = {"out"};

  TensorValueMap outputs;
  outputs["Out"] = {out};

  GcuAttributeMap attrs;
  attrs["axis"] = axis;

  GcuRunner(
      input_names, inputs, output_names, outputs, attrs, op_type, dev_ctx);
}

template <typename T, typename Context>
void EqualKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const phi::DenseTensor& y,
                 phi::DenseTensor* out) {
  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "equal", equal);
    dev_ctx.template Alloc<bool>(out);
    phi::DenseTensor tmp_x = x;
    phi::DenseTensor tmp_y = y;
    if (x.dtype() == phi::DataType::INT64) {
      tmp_x = cast(dev_ctx, x, phi::DataType::INT32);
    }
    if (y.dtype() == phi::DataType::INT64) {
      tmp_y = cast(dev_ctx, y, phi::DataType::INT32);
    }
    equal_compute(
        static_cast<const phi::CustomContext&>(dev_ctx), tmp_x, tmp_y, out);
    PADDLE_GCU_KERNEL_END("equal", equal);
  } else {
    CompareBaseKernel<T, Context>(dev_ctx, x, y, -1, out, "equal");
  }
}

template <typename T, typename Context>
void NotEqualKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    phi::DenseTensor* out) {
  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "not_equal", not_equal);
    dev_ctx.template Alloc<bool>(out);
    phi::DenseTensor tmp_x = x;
    phi::DenseTensor tmp_y = y;
    if (x.dtype() == phi::DataType::INT64) {
      tmp_x = cast(dev_ctx, x, phi::DataType::INT32);
    }
    if (y.dtype() == phi::DataType::INT64) {
      tmp_y = cast(dev_ctx, y, phi::DataType::INT32);
    }
    not_equal_compute(
        static_cast<const phi::CustomContext&>(dev_ctx), tmp_x, tmp_y, out);
    PADDLE_GCU_KERNEL_END("not_equal", not_equal);
  } else {
    CompareBaseKernel<T, Context>(dev_ctx, x, y, -1, out, "not_equal");
  }
}

template <typename T, typename Context>
void LessEqualKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& y,
                     phi::DenseTensor* out) {
  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "less_equal", less_equal);
    dev_ctx.template Alloc<bool>(out);
    phi::DenseTensor tmp_x = x;
    phi::DenseTensor tmp_y = y;
    if (x.dtype() == phi::DataType::INT64) {
      tmp_x = cast(dev_ctx, x, phi::DataType::INT32);
    }
    if (y.dtype() == phi::DataType::INT64) {
      tmp_y = cast(dev_ctx, y, phi::DataType::INT32);
    }
    less_equal_compute(
        static_cast<const phi::CustomContext&>(dev_ctx), tmp_x, tmp_y, out);
    PADDLE_GCU_KERNEL_END("less_equal", less_equal);
  } else {
    CompareBaseKernel<T, Context>(dev_ctx, x, y, -1, out, "less_equal");
  }
}

template <typename T, typename Context>
void LessThanKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    phi::DenseTensor* out) {
  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "less_than", less_than);
    dev_ctx.template Alloc<bool>(out);
    phi::DenseTensor tmp_x = x;
    phi::DenseTensor tmp_y = y;
    if (x.dtype() == phi::DataType::INT64) {
      tmp_x = cast(dev_ctx, x, phi::DataType::INT32);
    }
    if (y.dtype() == phi::DataType::INT64) {
      tmp_y = cast(dev_ctx, y, phi::DataType::INT32);
    }
    less_than_compute(
        static_cast<const phi::CustomContext&>(dev_ctx), tmp_x, tmp_y, out);
    PADDLE_GCU_KERNEL_END("less_than", less_than);
  } else {
    CompareBaseKernel<T, Context>(dev_ctx, x, y, -1, out, "less_than");
  }
}

template <typename T, typename Context>
void GreaterEqualKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& y,
                        phi::DenseTensor* out) {
  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "greater_equal", greater_equal);
    dev_ctx.template Alloc<bool>(out);
    phi::DenseTensor tmp_x = x;
    phi::DenseTensor tmp_y = y;
    if (x.dtype() == phi::DataType::INT64) {
      tmp_x = cast(dev_ctx, x, phi::DataType::INT32);
    }
    if (y.dtype() == phi::DataType::INT64) {
      tmp_y = cast(dev_ctx, y, phi::DataType::INT32);
    }
    greater_equal_compute(
        static_cast<const phi::CustomContext&>(dev_ctx), tmp_x, tmp_y, out);
    PADDLE_GCU_KERNEL_END("greater_equal", greater_equal);
  } else {
    CompareBaseKernel<T, Context>(dev_ctx, x, y, -1, out, "greater_equal");
  }
}

template <typename T, typename Context>
void GreaterThanKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& y,
                       phi::DenseTensor* out) {
  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "greater_than", greater_than);
    dev_ctx.template Alloc<bool>(out);
    phi::DenseTensor tmp_x = x;
    phi::DenseTensor tmp_y = y;
    if (x.dtype() == phi::DataType::INT64) {
      tmp_x = cast(dev_ctx, x, phi::DataType::INT32);
    }
    if (y.dtype() == phi::DataType::INT64) {
      tmp_y = cast(dev_ctx, y, phi::DataType::INT32);
    }
    greater_than_compute(
        static_cast<const phi::CustomContext&>(dev_ctx), tmp_x, tmp_y, out);
    PADDLE_GCU_KERNEL_END("greater_than", greater_than);
  } else {
    CompareBaseKernel<T, Context>(dev_ctx, x, y, -1, out, "greater_than");
  }
}

}  // namespace custom_kernel

#define PD_REGISTER_COMPARE_KERNEL(name, func)            \
  PD_REGISTER_PLUGIN_KERNEL(name,                         \
                            gcu,                          \
                            ALL_LAYOUT,                   \
                            custom_kernel::func##Kernel,  \
                            bool,                         \
                            int16_t,                      \
                            int,                          \
                            int64_t,                      \
                            float,                        \
                            phi::dtype::float16,          \
                            double) {                     \
    kernel->OutputAt(0).SetDataType(phi::DataType::BOOL); \
  }

PD_REGISTER_COMPARE_KERNEL(less_than, LessThan)
PD_REGISTER_COMPARE_KERNEL(less_equal, LessEqual)
PD_REGISTER_COMPARE_KERNEL(greater_than, GreaterThan)
PD_REGISTER_COMPARE_KERNEL(greater_equal, GreaterEqual)
PD_REGISTER_COMPARE_KERNEL(equal, Equal)
PD_REGISTER_COMPARE_KERNEL(not_equal, NotEqual)
