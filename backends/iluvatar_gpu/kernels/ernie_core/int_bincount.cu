// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

// #include "paddle/extension.h"
#include <cstdint>
#include <vector>

#include "cub/device/device_histogram.cuh"
#include "paddle/common/flags.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"  // NOLINT

COMMON_DECLARE_bool(enable_pir_api);

namespace phi {
static phi::DataType TransToDataType(int64_t dtype) {
  if (FLAGS_enable_pir_api) {
    return static_cast<phi::DataType>(dtype);
  } else {
    return phi::TransToPhiDataType(dtype);
  }
}

std::vector<std::vector<int64_t>> IntBincountInferShape(
    std::vector<int64_t> x_shape,
    int64_t min_value,
    int64_t max_value,
    int64_t out_dtype) {
  return {{max_value - min_value}};
}

std::vector<phi::DataType> IntBincountInferDType(phi::DataType x_dtype,
                                                 int64_t min_value,
                                                 int64_t max_value,
                                                 int64_t out_dtype) {
  return {TransToDataType(out_dtype)};
}

template <typename T, typename BinsT, typename Context>
void IntBincountImpl(const Context &dev_ctx,
                     const T *x,
                     int64_t n,
                     T min_v,
                     T max_v,
                     BinsT *bins) {
  DenseTensor workspace;
  void *workspace_ptr = nullptr;
  size_t workspace_size = 0;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    if (workspace_size > 0) {
      workspace =
          phi::Empty<uint8_t>(dev_ctx, {static_cast<int64_t>(workspace_size)});
      workspace_ptr = workspace.data();
    }
    auto err = cub::DeviceHistogram::HistogramEven(workspace_ptr,
                                                   workspace_size,
                                                   x,
                                                   bins,
                                                   max_v - min_v + 1,
                                                   min_v,
                                                   max_v,
                                                   n,
                                                   dev_ctx.stream());
    PD_CHECK(
        err == cudaSuccess, "HistogramEven error: %s", cudaGetErrorString(err));
  }
}

// T is x's input type and out_dtype is in args
template <typename T, typename Context>
void IntBincount(const Context &dev_ctx,
                 const DenseTensor &x,
                 int64_t low,
                 int64_t high,
                 int64_t out_dtype,
                 DenseTensor *out) {
  PD_CHECK(low < high);
  int64_t bins_width = high - low;
  PD_CHECK(bins_width + 1 < std::numeric_limits<int>::max());

  auto bins_dtype = TransToDataType(out_dtype);

  // auto x_dytpe = x.dtype();
  auto low_v = static_cast<T>(low);
  auto high_v = static_cast<T>(high);
  PD_CHECK(static_cast<int64_t>(low_v) == low);
  PD_CHECK(static_cast<int64_t>(high_v) == high);
  const auto *x_data = x.data<T>();
  int64_t n = x.numel();
  if (bins_dtype == phi::DataType::INT32) {
    dev_ctx.template Alloc<int32_t>(out);
    uint32_t *out_ptr = static_cast<uint32_t *>(out->data());
    IntBincountImpl<T, uint32_t, Context>(
        dev_ctx, x_data, n, low_v, high_v, out_ptr);
  } else if (bins_dtype == phi::DataType::INT64) {
    // HistogramEven not supoort int64_t now, use int32_t as temp
    dev_ctx.template Alloc<int64_t>(out);
    DenseTensor temp_out;
    temp_out.Resize(out->dims());
    temp_out.set_type(phi::DataType::INT32);
    dev_ctx.template Alloc<int32_t>(&temp_out);
    uint32_t *temp_ptr = static_cast<uint32_t *>(temp_out.data());
    IntBincountImpl<T, uint32_t, Context>(
        dev_ctx, x_data, n, low_v, high_v, temp_ptr);
    phi::CastKernel<int32_t, Context>(
        dev_ctx, temp_out, phi::DataType::INT64, out);
  } else {
    PD_THROW("Only support INT32 and INT64, but got %s", bins_dtype);
  }
}
}  // namespace phi

PD_CUSTOM_KERNEL_REGISTER(
    int_bincount, iluvatar_gpu, ALL_LAYOUT, phi::IntBincount, int64_t, int) {}
