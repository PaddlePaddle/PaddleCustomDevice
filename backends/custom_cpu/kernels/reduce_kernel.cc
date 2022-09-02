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

#include "paddle/phi/capi/all.h"
#include "phi_funcs.h"

namespace custom_kernel {

template <typename T>
void MeanRawKernel(const phi::Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::IntArray& dims,
                   bool keep_dim,
                   bool reduce_all,
                   phi::DenseTensor* out) {
  auto x_dims = x.dims();
  auto reduce_dims = dims.GetData();
  if (reduce_all) {
    reduce_dims.clear();
    for (auto i = 0; i < x_dims.size(); ++i) {
      reduce_dims.push_back(i);
    }
  }

  auto x_data = x.data<T>();
  auto out_data = dev_ctx.template Alloc<T>(out);

  auto numel = x.numel();
  std::vector<size_t> index(x_dims.size(), 0);
  std::vector<size_t> step(x_dims.size(), 1);
  for (auto i = x_dims.size() - 1; i > 0; --i) {
    step[i - 1] = step[i] * x_dims[i];
  }
  std::vector<size_t> dst_step = step;
  for (auto d : reduce_dims) {
    dst_step[d] = 0;
  }

  memset(out_data, 0, sizeof(T) * out->numel());
  auto reduce_numel = static_cast<T>(phi::product(reduce_dims));
  for (auto i = 0; i < numel; ++i) {
    out_data[phi::vec_product(dst_step, index)] +=
        x_data[phi::vec_product(step, index)] / reduce_numel;
    index.back()++;
    for (auto j = index.size() - 1; j > 0; --j) {
      if (index[j] >= x_dims[j]) {
        index[j] = 0;
        index[j - 1]++;
      } else {
        break;
      }
    }
  }
}

template <typename T>
void MeanKernel(const phi::Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::IntArray& dims,
                bool keep_dim,
                phi::DenseTensor* out) {
  bool reduce_all = false;
  MeanRawKernel<T>(dev_ctx, x, dims, keep_dim, reduce_all, out);
}

template <typename T>
void SumRawKernel(const phi::Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::IntArray& dims,
                  bool keep_dim,
                  bool reduce_all,
                  phi::DataType out_dtype,
                  phi::DenseTensor* out) {
  auto x_dims = x.dims();
  auto reduce_dims = dims.GetData();
  if (reduce_all) {
    reduce_dims.clear();
    for (auto i = 0; i < x_dims.size(); ++i) {
      reduce_dims.push_back(i);
    }
  }

  auto x_data = x.data<T>();
  auto out_data = dev_ctx.template Alloc<T>(out);

  auto numel = x.numel();
  std::vector<size_t> index(x_dims.size(), 0);
  std::vector<size_t> step(x_dims.size(), 1);
  for (auto i = x_dims.size() - 1; i > 0; --i) {
    step[i - 1] = step[i] * x_dims[i];
  }
  std::vector<size_t> dst_step = step;
  for (auto d : reduce_dims) {
    dst_step[d] = 0;
  }

  memset(out_data, 0, sizeof(T) * out->numel());
  for (auto i = 0; i < numel; ++i) {
    out_data[phi::vec_product(dst_step, index)] +=
        x_data[phi::vec_product(step, index)];
    index.back()++;
    for (auto j = index.size() - 1; j > 0; --j) {
      if (index[j] >= x_dims[j]) {
        index[j] = 0;
        index[j - 1]++;
      } else {
        break;
      }
    }
  }
}

template <typename T>
void SumKernel(const phi::Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::IntArray& dims,
               phi::DataType out_dtype,
               bool keep_dim,
               phi::DenseTensor* out) {
  bool reduce_all = false;
  SumRawKernel<T>(dev_ctx, x, dims, keep_dim, reduce_all, out_dtype, out);
}

template <typename T>
void MinRawKernel(const phi::Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::IntArray& dims,
                  bool keep_dim,
                  bool reduce_all,
                  phi::DenseTensor* out) {
  auto x_dims = x.dims();
  auto reduce_dims = dims.GetData();
  if (reduce_all) {
    reduce_dims.clear();
    for (auto i = 0; i < x_dims.size(); ++i) {
      reduce_dims.push_back(i);
    }
  }

  auto x_data = x.data<T>();
  auto out_data = dev_ctx.template Alloc<T>(out);

  auto numel = x.numel();
  std::vector<size_t> index(x_dims.size(), 0);
  std::vector<size_t> step(x_dims.size(), 1);

  for (auto i = x_dims.size() - 1; i > 0; --i) {
    step[i - 1] = step[i] * x_dims[i];
  }
  std::vector<size_t> dst_step = step;
  for (auto d : reduce_dims) {
    dst_step[d] = 0;
  }

  for (auto i = 0; i < numel; ++i) {
    out_data[phi::vec_product(dst_step, index)] =
        x_data[phi::vec_product(step, index)];
  }
  for (auto i = 0; i < numel; ++i) {
    out_data[phi::vec_product(dst_step, index)] =
        std::min(out_data[phi::vec_product(dst_step, index)],
                 x_data[phi::vec_product(step, index)]);
    index.back()++;
    for (auto j = index.size() - 1; j > 0; --j) {
      if (index[j] >= x_dims[j]) {
        index[j] = 0;
        index[j - 1]++;
      } else {
        break;
      }
    }
  }
}

template <typename T>
void MinKernel(const phi::Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::IntArray& dims,
               bool keep_dim,
               phi::DenseTensor* out) {
  bool reduce_all = false;
  MinRawKernel<T>(dev_ctx, x, dims, keep_dim, reduce_all, out);
}

template <typename T>
void MaxRawKernel(const phi::Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::IntArray& dims,
                  bool keep_dim,
                  bool reduce_all,
                  phi::DenseTensor* out) {
  auto x_dims = x.dims();
  auto reduce_dims = dims.GetData();
  if (reduce_all) {
    reduce_dims.clear();
    for (auto i = 0; i < x_dims.size(); ++i) {
      reduce_dims.push_back(i);
    }
  }

  auto x_data = x.data<T>();
  auto out_data = dev_ctx.template Alloc<T>(out);

  auto numel = x.numel();
  std::vector<size_t> index(x_dims.size(), 0);
  std::vector<size_t> step(x_dims.size(), 1);
  for (auto i = x_dims.size() - 1; i > 0; --i) {
    step[i - 1] = step[i] * x_dims[i];
  }
  std::vector<size_t> dst_step = step;
  for (auto d : reduce_dims) {
    dst_step[d] = 0;
  }

  for (auto i = 0; i < numel; ++i) {
    out_data[phi::vec_product(dst_step, index)] =
        x_data[phi::vec_product(step, index)];
  }
  for (auto i = 0; i < numel; ++i) {
    out_data[phi::vec_product(dst_step, index)] =
        std::max(out_data[phi::vec_product(dst_step, index)],
                 x_data[phi::vec_product(step, index)]);
    index.back()++;
    for (auto j = index.size() - 1; j > 0; --j) {
      if (index[j] >= x_dims[j]) {
        index[j] = 0;
        index[j - 1]++;
      } else {
        break;
      }
    }
  }
}

template <typename T>
void MaxKernel(const phi::Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::IntArray& dims,
               bool keep_dim,
               phi::DenseTensor* out) {
  bool reduce_all = false;
  MaxRawKernel<T>(dev_ctx, x, dims, keep_dim, reduce_all, out);
}

}  // namespace custom_kernel

PD_BUILD_PHI_KERNEL(mean_raw,
                    custom_cpu,
                    ALL_LAYOUT,
                    custom_kernel::MeanRawKernel,
                    float,
                    double) {}

PD_BUILD_PHI_KERNEL(
    mean, custom_cpu, ALL_LAYOUT, custom_kernel::MeanKernel, float, double) {}

PD_BUILD_PHI_KERNEL(sum_raw,
                    custom_cpu,
                    ALL_LAYOUT,
                    custom_kernel::SumRawKernel,
                    float,
                    double) {}

PD_BUILD_PHI_KERNEL(
    sum, custom_cpu, ALL_LAYOUT, custom_kernel::SumKernel, float, double) {}

PD_BUILD_PHI_KERNEL(min_raw,
                    custom_cpu,
                    ALL_LAYOUT,
                    custom_kernel::MinRawKernel,
                    int32_t,
                    int64_t,
                    float,
                    double) {}

PD_BUILD_PHI_KERNEL(min,
                    custom_cpu,
                    ALL_LAYOUT,
                    custom_kernel::MinKernel,
                    int32_t,
                    int64_t,
                    float,
                    double) {}

PD_BUILD_PHI_KERNEL(max_raw,
                    custom_cpu,
                    ALL_LAYOUT,
                    custom_kernel::MaxRawKernel,
                    int32_t,
                    int64_t,
                    float,
                    double) {}

PD_BUILD_PHI_KERNEL(max,
                    custom_cpu,
                    ALL_LAYOUT,
                    custom_kernel::MaxKernel,
                    int32_t,
                    int64_t,
                    float,
                    double) {}
