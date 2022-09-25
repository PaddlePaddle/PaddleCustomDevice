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

#include <cmath>

#include "glog/logging.h"
#include "paddle/phi/capi/all.h"
#include "phi_funcs.h"
#include <CL/sycl.hpp>

namespace custom_kernel {

// template <typename T, typename F>
template <typename T, typename F, typename FF>
void RawCompareKernel(const phi::Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const phi::DenseTensor& y,
                 int axis,
                 phi::DenseTensor* out,
                 const F& func,
                 const FF& float_func) {
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  auto dst_dims = phi::BroadcastDims(axis, x_dims, y_dims);

// TODO BroadcastTo gives segfault
  // phi::DenseTensor tmp_x, tmp_y;
  // phi::BroadcastTo<T>(dev_ctx, x, dst_dims, axis, &tmp_x);
  // phi::BroadcastTo<T>(dev_ctx, y, dst_dims, axis, &tmp_y);

  // auto x_data = tmp_x.data<T>();
  // auto y_data = tmp_y.data<T>();
  auto x_data = x.data<T>();
  auto y_data = y.data<T>();
  auto out_data = dev_ctx.template Alloc<bool>(out);
  auto numel = out->numel();
  
  auto* q = static_cast<sycl::queue*>(dev_ctx.stream());
  // if float_func == func only func is to be calculated
  if (float_func != func && std::is_floating_point<T>::value) {
    q->parallel_for(numel, [=](auto& i){
      float_func(x_data, y_data, out_data, i);
    });
  } else {
    q->parallel_for(numel, [=](auto& i){
      func(x_data, y_data, out_data, i);
    });
  }  
  q->wait();
}

template <typename T, typename F, typename FF>
void EqualityKernel(const phi::Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const phi::DenseTensor& y,
                 int axis,
                 phi::DenseTensor* out,
                 const F& func,
                 const FF& float_func) {
  RawCompareKernel<T>(dev_ctx, x, y, axis, out, float_func, func);
}

template <typename T, typename F>
void CompareKernel(const phi::Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const phi::DenseTensor& y,
                 int axis,
                 phi::DenseTensor* out,
                 const F& func) {
  RawCompareKernel<T>(dev_ctx, x, y, axis, out, func, func);
}


template <typename T>
void NotEqualKernel(const phi::Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    int axis,
                    phi::DenseTensor* out) {
  VLOG(3) << "NotEqualKernel-SYCL";
  show_kernel("NotEqual-SYCL");
  
  EqualityKernel<T>(dev_ctx, x, y, axis, out, 
    [](T* x_data, T* y_data, bool* out_data, long i){  
          out_data[i] = static_cast<bool>(
              std::fabs(static_cast<double>(x_data[i] - y_data[i])) >= 1e-8);
    },
    [](T* x_data, T* y_data, bool* out_data, long i){
              out_data[i] = x_data[i] != y_data[i];
    });

  // for (auto i = 0; i < numel; ++i) {
  //   if (std::is_floating_point<T>::value) {
  //     out_data[i] = static_cast<bool>(
  //         fabs(static_cast<double>(x_data[i] - y_data[i])) >= 1e-8);
  //   } else {
  //     out_data[i] = x_data[i] != y_data[i];
  //   }
  // }
}

template <typename T>
void EqualKernel(const phi::Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const phi::DenseTensor& y,
                 int axis,
                 phi::DenseTensor* out) {
  VLOG(3) << "EqualKernel-SYCL";
  show_kernel("Equal-SYCL");

  EqualityKernel<T>(dev_ctx, x, y, axis, out, 
    [](T* x_data, T* y_data, bool* out_data, long i){  
          out_data[i] = static_cast<bool>(
              std::fabs(static_cast<double>(x_data[i] - y_data[i])) < 1e-8);
    },
    [](T* x_data, T* y_data, bool* out_data, long i){
              out_data[i] = x_data[i] == y_data[i];
    });

  // for (auto i = 0; i < numel; ++i) {
  //   if (std::is_floating_point<T>::value) {
  //     out_data[i] = static_cast<bool>(
  //         fabs(static_cast<double>(x_data[i] - y_data[i])) < 1e-8);
  //   } else {
  //     out_data[i] = x_data[i] == y_data[i];
  //   }
  // }
}

template <typename T>
void LessThanKernel(const phi::Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    int axis,
                    phi::DenseTensor* out) {
  VLOG(3) << "Tutaj LessThanKernel";
  show_kernel("LessThan-SYCL");

  CompareKernel<T>(dev_ctx, x, y, axis, out, 
    [](T* x_data, T* y_data, bool* out_data, long i){
        out_data[i] = x_data[i] < y_data[i];
    }
  );

  // for (auto i = 0; i < numel; ++i) {
  //   out_data[i] = x_data[i] < y_data[i];
  // }
}

template <typename T>
void LessEqualKernel(const phi::Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& y,
                     int axis,
                     phi::DenseTensor* out) {
  VLOG(3) << "Tutaj LessEqualKernel";
  show_kernel("LessEqual-SYCL");

  CompareKernel<T>(dev_ctx, x, y, axis, out, 
    [](T* x_data, T* y_data, bool* out_data, long i){
      out_data[i] = x_data[i] <= y_data[i];
    }
  );

  // for (auto i = 0; i < numel; ++i) {
  //   out_data[i] = x_data[i] <= y_data[i];
  // }
}

template <typename T>
void GreaterThanKernel(const phi::Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& y,
                       int axis,
                       phi::DenseTensor* out) {
  VLOG(3) << "Tutaj GreaterThanKernel";
  show_kernel("GreaterThan-SYCL");

  CompareKernel<T>(dev_ctx, x, y, axis, out, 
    [](T* x_data, T* y_data, bool* out_data, long i){
      out_data[i] = x_data[i] > y_data[i];
    }
  );

  // for (auto i = 0; i < numel; ++i) {
  //   out_data[i] = x_data[i] > y_data[i];
  // }
}

template <typename T>
void GreaterEqualKernel(const phi::Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& y,
                        int axis,
                        phi::DenseTensor* out) {
  VLOG(3) << "Tutaj GreaterEqualKernel";
  show_kernel("GreaterEqual-SYCL");

  CompareKernel<T>(dev_ctx, x, y, axis, out, 
    [](T* x_data, T* y_data, bool* out_data, long i){
      out_data[i] = x_data[i] >= y_data[i];
    }
  );

  // for (auto i = 0; i < numel; ++i) {
  //   out_data[i] = x_data[i] >= y_data[i];
  // }
}

}  // namespace custom_kernel

PD_BUILD_PHI_KERNEL(not_equal,
                    intel_gpu,
                    ALL_LAYOUT,
                    custom_kernel::NotEqualKernel,
                    float,
                    double,
                    uint8_t,
                    int16_t,
                    int32_t,
                    int64_t,
                    bool) {}

PD_BUILD_PHI_KERNEL(equal,
                    intel_gpu,
                    ALL_LAYOUT,
                    custom_kernel::EqualKernel,
                    float,
                    double,
                    uint8_t,
                    int16_t,
                    int32_t,
                    int64_t,
                    bool) {}

PD_BUILD_PHI_KERNEL(less_than,
                    intel_gpu,
                    ALL_LAYOUT,
                    custom_kernel::LessThanKernel,
                    float,
                    double,
                    uint8_t,
                    int16_t,
                    int32_t,
                    int64_t,
                    bool) {}

PD_BUILD_PHI_KERNEL(less_equal,
                    intel_gpu,
                    ALL_LAYOUT,
                    custom_kernel::LessEqualKernel,
                    float,
                    double,
                    uint8_t,
                    int16_t,
                    int32_t,
                    int64_t,
                    bool) {}

PD_BUILD_PHI_KERNEL(greater_than,
                    intel_gpu,
                    ALL_LAYOUT,
                    custom_kernel::GreaterThanKernel,
                    float,
                    double,
                    uint8_t,
                    int16_t,
                    int32_t,
                    int64_t,
                    bool) {}

PD_BUILD_PHI_KERNEL(greater_equal,
                    intel_gpu,
                    ALL_LAYOUT,
                    custom_kernel::GreaterEqualKernel,
                    float,
                    double,
                    uint8_t,
                    int16_t,
                    int32_t,
                    int64_t,
                    bool) {}
