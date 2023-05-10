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

#include "kernels/funcs/npu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void CastKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DataType dtype,
                phi::DenseTensor* out);

template <typename T, typename Context>
void EqualRawKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    int axis,
                    phi::DenseTensor* out) {
  auto stream = dev_ctx.stream();
  dev_ctx.template Alloc<bool>(out);

  phi::DenseTensor transformed_x(x), transformed_y;
  if (x.dtype() != y.dtype()) {
    phi::DenseTensorMeta meta = {x.dtype(), y.dims()};
    transformed_y.set_meta(meta);
    custom_kernel::CastKernel<T, Context>(
        dev_ctx, y, x.dtype(), &transformed_y);
  } else {
    transformed_y = y;
  }

  const auto& runner =
      NpuOpRunner("Equal", {transformed_x, transformed_y}, {*out}, {});
  runner.Run(stream);
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
  if (x.dtype() == phi::DataType::BOOL || y.dtype() == phi::DataType::BOOL) {
    auto op_func = [](const std::vector<phi::DenseTensor>& inputs,
                      const std::vector<phi::DenseTensor>& outputs,
                      const NPUAttributeMap& attrs,
                      const Context& dev_ctx) {
      const auto& runner = NpuOpRunner("NotEqual", inputs, outputs, attrs);
      runner.Run(dev_ctx.stream());
    };
    NPUAttributeMap attr_input = {};
    NpuOpRunner::TypeAdapter({x, y},
                             {*out},
                             attr_input,
                             dev_ctx,
                             op_func,
                             {phi::DataType::UINT8, phi::DataType::UINT8},
                             {phi::DataType::BOOL});
  } else {
    const auto& runner = NpuOpRunner("NotEqual", {x, y}, {*out}, {});
    auto stream = dev_ctx.stream();
    runner.Run(stream);
  }
}

template <typename T, typename Context>
void NotEqualKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    phi::DenseTensor* out) {
  custom_kernel::NotEqualRawKernel<T, Context>(dev_ctx, x, y, -1, out);
}

template <typename T, typename Context>
void LessEqualRawKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& y,
                        int axis,
                        phi::DenseTensor* out) {
  dev_ctx.template Alloc<bool>(out);
  if (x.dtype() == phi::DataType::BOOL || y.dtype() == phi::DataType::BOOL) {
    auto op_func = [](const std::vector<phi::DenseTensor>& inputs,
                      const std::vector<phi::DenseTensor>& outputs,
                      const NPUAttributeMap& attrs,
                      const Context& dev_ctx) {
      const auto& runner = NpuOpRunner("LessEqual", inputs, outputs, attrs);
      runner.Run(dev_ctx.stream());
    };
    NPUAttributeMap attr_input = {};
    NpuOpRunner::TypeAdapter({x, y},
                             {*out},
                             attr_input,
                             dev_ctx,
                             op_func,
                             {phi::DataType::UINT8, phi::DataType::UINT8},
                             {phi::DataType::BOOL});
  } else {
    auto stream = dev_ctx.stream();
    const auto& runner = NpuOpRunner("LessEqual", {x, y}, {*out}, {});
    runner.Run(stream);
  }
}

template <typename T, typename Context>
void LessEqualKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& y,
                     phi::DenseTensor* out) {
  custom_kernel::LessEqualRawKernel<T, Context>(dev_ctx, x, y, -1, out);
}

template <typename T, typename Context>
void LessThanRawKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& y,
                       int axis,
                       phi::DenseTensor* out) {
  dev_ctx.template Alloc<bool>(out);
  if (x.dtype() == phi::DataType::BOOL || y.dtype() == phi::DataType::BOOL) {
    auto op_func = [](const std::vector<phi::DenseTensor>& inputs,
                      const std::vector<phi::DenseTensor>& outputs,
                      const NPUAttributeMap& attrs,
                      const Context& dev_ctx) {
      const auto& runner = NpuOpRunner("Less", inputs, outputs, attrs);
      runner.Run(dev_ctx.stream());
    };
    NPUAttributeMap attr_input = {};
    NpuOpRunner::TypeAdapter({x, y},
                             {*out},
                             attr_input,
                             dev_ctx,
                             op_func,
                             {phi::DataType::UINT8, phi::DataType::UINT8},
                             {phi::DataType::BOOL});
  } else {
    const auto& runner = NpuOpRunner("Less", {x, y}, {*out}, {});
    auto stream = dev_ctx.stream();
    runner.Run(stream);
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
void GreaterEqualRawKernel(const Context& dev_ctx,
                           const phi::DenseTensor& x,
                           const phi::DenseTensor& y,
                           int axis,
                           phi::DenseTensor* out) {
  dev_ctx.template Alloc<bool>(out);
  if (x.dtype() == phi::DataType::BOOL || y.dtype() == phi::DataType::BOOL) {
    auto op_func = [](const std::vector<phi::DenseTensor>& inputs,
                      const std::vector<phi::DenseTensor>& outputs,
                      const NPUAttributeMap& attrs,
                      const Context& dev_ctx) {
      const auto& runner = NpuOpRunner("GreaterEqual", inputs, outputs, attrs);
      runner.Run(dev_ctx.stream());
    };
    NPUAttributeMap attr_input = {};
    NpuOpRunner::TypeAdapter({x, y},
                             {*out},
                             attr_input,
                             dev_ctx,
                             op_func,
                             {phi::DataType::UINT8, phi::DataType::UINT8},
                             {phi::DataType::BOOL});
  } else {
    const auto& runner = NpuOpRunner("GreaterEqual", {x, y}, {*out}, {});
    auto stream = dev_ctx.stream();
    runner.Run(stream);
  }
}

template <typename T, typename Context>
void GreaterEqualKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& y,
                        phi::DenseTensor* out) {
  custom_kernel::GreaterEqualRawKernel<T, Context>(dev_ctx, x, y, -1, out);
}

template <typename T, typename Context>
void GreaterThanRawKernel(const Context& dev_ctx,
                          const phi::DenseTensor& x,
                          const phi::DenseTensor& y,
                          int axis,
                          phi::DenseTensor* out) {
  dev_ctx.template Alloc<bool>(out);
  if (x.dtype() == phi::DataType::BOOL || y.dtype() == phi::DataType::BOOL) {
    auto op_func = [](const std::vector<phi::DenseTensor>& inputs,
                      const std::vector<phi::DenseTensor>& outputs,
                      const NPUAttributeMap& attrs,
                      const Context& dev_ctx) {
      const auto& runner = NpuOpRunner("Greater", inputs, outputs, attrs);
      runner.Run(dev_ctx.stream());
    };
    NPUAttributeMap attr_input = {};
    NpuOpRunner::TypeAdapter({x, y},
                             {*out},
                             attr_input,
                             dev_ctx,
                             op_func,
                             {phi::DataType::UINT8, phi::DataType::UINT8},
                             {phi::DataType::BOOL});
  } else {
    const auto& runner = NpuOpRunner("Greater", {x, y}, {*out}, {});
    auto stream = dev_ctx.stream();
    runner.Run(stream);
  }
}

template <typename T, typename Context>
void GreaterThanKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& y,
                       phi::DenseTensor* out) {
  custom_kernel::GreaterThanRawKernel<T, Context>(dev_ctx, x, y, -1, out);
}

}  // namespace custom_kernel

#define PD_REGISTER_COMPARE_KERNEL(name, func)              \
  PD_REGISTER_PLUGIN_KERNEL(name,                           \
                            npu,                            \
                            ALL_LAYOUT,                     \
                            custom_kernel::func##Kernel,    \
                            bool,                           \
                            int16_t,                        \
                            int,                            \
                            int64_t,                        \
                            float,                          \
                            phi::dtype::float16,            \
                            double) {                       \
    kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);   \
  }                                                         \
  PD_REGISTER_PLUGIN_KERNEL(name##_raw,                     \
                            npu,                            \
                            ALL_LAYOUT,                     \
                            custom_kernel::func##RawKernel, \
                            bool,                           \
                            int16_t,                        \
                            int,                            \
                            int64_t,                        \
                            float,                          \
                            phi::dtype::float16,            \
                            double) {                       \
    kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);   \
  }

PD_REGISTER_COMPARE_KERNEL(less_than, LessThan)
PD_REGISTER_COMPARE_KERNEL(less_equal, LessEqual)
PD_REGISTER_COMPARE_KERNEL(greater_than, GreaterThan)
PD_REGISTER_COMPARE_KERNEL(greater_equal, GreaterEqual)
PD_REGISTER_COMPARE_KERNEL(equal, Equal)
PD_REGISTER_COMPARE_KERNEL(not_equal, NotEqual)
