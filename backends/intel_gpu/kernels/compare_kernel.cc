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

#include "kernels/dnn_support.hpp"
#include "kernels/phi_funcs.h"
#include "paddle/phi/capi/all.h"

namespace custom_kernel {

template <typename T, typename F, typename FF>
void RawCompareKernelSycl(const phi::Context& dev_ctx,
                          std::string kernel_name,
                          const phi::DenseTensor& x,
                          const phi::DenseTensor& y,
                          int axis,
                          phi::DenseTensor* out,
                          const F& func,
                          const FF& float_func) {
  show_kernel(kernel_name << "-SYCL type="
                          << dnn_support::type2String<T>::name());

  auto x_dims = x.dims();
  auto y_dims = y.dims();
  auto dst_dims = phi::BroadcastDims(axis, x_dims, y_dims);

  auto x_data = x.data<T>();
  auto y_data = y.data<T>();
  auto out_data = dev_ctx.template Alloc<bool>(out);
  auto numel = out->numel();

  auto* q = static_cast<sycl::queue*>(dev_ctx.stream());
  // if float_func == func only func is to be calculated
  if (float_func != func && std::is_floating_point<T>::value) {
    q->parallel_for(numel,
                    [=](auto& i) { float_func(x_data, y_data, out_data, i); });
  } else {
    q->parallel_for(numel, [=](auto& i) { func(x_data, y_data, out_data, i); });
  }
  q->wait();
}

template <typename T>
void RawCompareKernelDNN(const phi::Context& dev_ctx,
                         std::string kernel_name,
                         dnnl::algorithm binary_type,
                         const phi::DenseTensor& x,
                         const phi::DenseTensor& y,
                         int axis,
                         phi::DenseTensor* out) {
  show_kernel(kernel_name << "-DNN type="
                          << dnn_support::type2String<T>::name());

  void* stream = const_cast<void*>(dev_ctx.stream());
  auto* q = static_cast<sycl::queue*>(const_cast<void*>(dev_ctx.stream()));

  using tag = dnnl::memory::format_tag;
  using dt = dnnl::memory::data_type;

  auto eng = dnnl::sycl_interop::make_engine(q->get_device(), q->get_context());
  auto engine_stream = dnnl::sycl_interop::make_stream(eng, *q);

  dnnl::memory::dims dims_x = x.dims();
  dnnl::memory::dims dims_y = y.dims();
  dnnl::memory::dims dims_out = out->dims();

  phi::update_broadcast(dims_x, dims_y, axis);

  auto md_x = dnnl::memory::desc(
      dims_x, dnn_support::toDnnType<T>::type, dnn_support::dims2Tag(dims_x));

  auto md_y = dnnl::memory::desc(
      dims_y, dnn_support::toDnnType<T>::type, dnn_support::dims2Tag(dims_y));
  auto md_out = dnnl::memory::desc(dims_out,
                                   dnn_support::toDnnType<T>::type,
                                   dnn_support::dims2Tag(dims_out));

  auto x_mem = dnnl::memory(md_x, eng, x.data<T>());
  auto y_mem = dnnl::memory(md_y, eng, y.data<T>());

  auto out_data = dev_ctx.template Alloc<T>(out);

  auto out_mem = dnnl::memory(md_out, eng, out_data);

  auto oper_desc = dnnl::binary::desc(binary_type, md_x, md_y, md_out);
  auto prim_desc = dnnl::binary::primitive_desc(oper_desc, eng);
  auto prim = dnnl::binary(prim_desc);

  std::unordered_map<int, dnnl::memory> binary_args;
  binary_args.insert({DNNL_ARG_SRC_0, x_mem});
  binary_args.insert({DNNL_ARG_SRC_1, y_mem});
  binary_args.insert({DNNL_ARG_DST, out_mem});

  prim.execute(engine_stream, binary_args);
  engine_stream.wait();
}

template <typename T, typename F, typename FF>
void EqualityKernel(const phi::Context& dev_ctx,
                    std::string kernel_name,
                    dnnl::algorithm binary_type,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    int axis,
                    phi::DenseTensor* out,
                    const F& func,
                    const FF& float_func) {
  if constexpr (std::is_same<T, float>::value) {
    RawCompareKernelDNN<T>(dev_ctx, kernel_name, binary_type, x, y, axis, out);
  } else {
    RawCompareKernelSycl<T>(
        dev_ctx, kernel_name, x, y, axis, out, float_func, func);
  }
}

template <typename T, typename F>
void CompareKernel(const phi::Context& dev_ctx,
                   std::string kernel_name,
                   dnnl::algorithm binary_type,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& y,
                   int axis,
                   phi::DenseTensor* out,
                   const F& func) {
  if constexpr (std::is_same<T, float>::value) {
    RawCompareKernelDNN<T>(dev_ctx, kernel_name, binary_type, x, y, axis, out);
  } else {
    RawCompareKernelSycl<T>(dev_ctx, kernel_name, x, y, axis, out, func, func);
  }
}

template <typename T>
void NotEqualKernel(const phi::Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    int axis,
                    phi::DenseTensor* out) {
  EqualityKernel<T>(
      dev_ctx,
      "NotEqual",
      dnnl::algorithm::binary_ne,
      x,
      y,
      axis,
      out,
      [](T* x_data, T* y_data, bool* out_data, int64_t i) {
        out_data[i] = x_data[i] != y_data[i];
      },
      [](T* x_data, T* y_data, bool* out_data, int64_t i) {
        out_data[i] = static_cast<bool>(
            std::fabs(static_cast<double>(x_data[i] - y_data[i])) >= 1e-8);
      });
}

template <typename T>
void EqualKernel(const phi::Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const phi::DenseTensor& y,
                 int axis,
                 phi::DenseTensor* out) {
  EqualityKernel<T>(
      dev_ctx,
      "Equal",
      dnnl::algorithm::binary_eq,
      x,
      y,
      axis,
      out,
      [](T* x_data, T* y_data, bool* out_data, int64_t i) {
        out_data[i] = x_data[i] == y_data[i];
      },
      [](T* x_data, T* y_data, bool* out_data, int64_t i) {
        out_data[i] = static_cast<bool>(
            std::fabs(static_cast<double>(x_data[i] - y_data[i])) < 1e-8);
      });
}

template <typename T>
void LessThanKernel(const phi::Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    int axis,
                    phi::DenseTensor* out) {
  CompareKernel<T>(dev_ctx,
                   "LessThanKernel",
                   dnnl::algorithm::binary_lt,
                   x,
                   y,
                   axis,
                   out,
                   [](T* x_data, T* y_data, bool* out_data, int64_t i) {
                     out_data[i] = x_data[i] < y_data[i];
                   });
}

template <typename T>
void LessEqualKernel(const phi::Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& y,
                     int axis,
                     phi::DenseTensor* out) {
  CompareKernel<T>(dev_ctx,
                   "LessEqual",
                   dnnl::algorithm::binary_le,
                   x,
                   y,
                   axis,
                   out,
                   [](T* x_data, T* y_data, bool* out_data, int64_t i) {
                     out_data[i] = x_data[i] <= y_data[i];
                   });
}

template <typename T>
void GreaterThanKernel(const phi::Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& y,
                       int axis,
                       phi::DenseTensor* out) {
  CompareKernel<T>(dev_ctx,
                   "GreaterThan",
                   dnnl::algorithm::binary_gt,
                   x,
                   y,
                   axis,
                   out,
                   [](T* x_data, T* y_data, bool* out_data, int64_t i) {
                     out_data[i] = x_data[i] > y_data[i];
                   });
}

template <typename T>
void GreaterEqualKernel(const phi::Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& y,
                        int axis,
                        phi::DenseTensor* out) {
  CompareKernel<T>(dev_ctx,
                   "GreaterEqual",
                   dnnl::algorithm::binary_ge,
                   x,
                   y,
                   axis,
                   out,
                   [](T* x_data, T* y_data, bool* out_data, int64_t i) {
                     out_data[i] = x_data[i] >= y_data[i];
                   });
}

}  // namespace custom_kernel

#define PD_REGISTER_COMPARE_KERNEL(name, func)            \
  PD_BUILD_PHI_KERNEL(name,                               \
                      intel_gpu,                          \
                      ALL_LAYOUT,                         \
                      custom_kernel::func##Kernel,        \
                      float,                              \
                      double,                             \
                      uint8_t,                            \
                      int16_t,                            \
                      int32_t,                            \
                      int64_t,                            \
                      bool) {                             \
    kernel->OutputAt(0).SetDataType(phi::DataType::BOOL); \
  }
PD_REGISTER_COMPARE_KERNEL(less_equal, LessEqual)
PD_REGISTER_COMPARE_KERNEL(greater_than, GreaterThan)
PD_REGISTER_COMPARE_KERNEL(greater_equal, GreaterEqual)
PD_REGISTER_COMPARE_KERNEL(equal, Equal)
PD_REGISTER_COMPARE_KERNEL(not_equal, NotEqual)
