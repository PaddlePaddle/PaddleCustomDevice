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

#include <cmath>

#include "kernels/phi_funcs.h"
#include "paddle/phi/capi/all.h"

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
  for (auto& d : reduce_dims) {
    // handle negative dims, f.e. "-1" means rightmost dimension
    if (d < 0) {
      d = d + x_dims.size();
    }
  }
  auto x_data = x.data<T>();
  auto out_data = dev_ctx.template Alloc<T>(out);

  auto numel = x.numel();
  auto out_dims(x_dims);
  for (auto d : reduce_dims) {
    out_dims[d] = 1;
  }
  std::vector<size_t> index(x_dims.size(), 0);
  std::vector<size_t> step(x_dims.size(), 1);
  std::vector<size_t> dst_step = step;
  for (auto i = x_dims.size() - 1; i > 0; --i) {
    step[i - 1] = step[i] * x_dims[i];
    dst_step[i - 1] = dst_step[i] * out_dims[i];
  }
  for (auto d : reduce_dims) {
    dst_step[d] = 0;
  }
  memset(out_data, 0, sizeof(T) * out->numel());
  size_t reduce_numel = 1;
  for (auto d : reduce_dims) {
    reduce_numel *= x_dims[d];
  }
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
  if (dims.size() == 0) {
    reduce_all = true;
  }
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
  if (reduce_dims.size() == 0) {
    reduce_all = true;
  }
  if (reduce_all) {
    reduce_dims.clear();
    for (auto i = 0; i < x_dims.size(); ++i) {
      reduce_dims.push_back(i);
    }
  }

  for (auto& d : reduce_dims) {
    // handle negative dims, f.e. "-1" means rightmost dimension
    if (d < 0) {
      d = d + x_dims.size();
    }
  }
  auto x_data = x.data<T>();
  auto out_data = dev_ctx.template Alloc<T>(out);

  auto numel = x.numel();
  auto out_dims(x_dims);
  for (auto d : reduce_dims) {
    out_dims[d] = 1;
  }
  std::vector<size_t> index(x_dims.size(), 0);
  std::vector<size_t> step(x_dims.size(), 1);
  std::vector<size_t> dst_step = step;
  for (auto i = x_dims.size() - 1; i > 0; --i) {
    step[i - 1] = step[i] * x_dims[i];
    dst_step[i - 1] = dst_step[i] * out_dims[i];
  }
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
  if (dims.size() == 0) {
    reduce_all = true;
  }
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
  if (reduce_dims.size() == 0) {
    reduce_all = true;
  }
  if (reduce_all) {
    reduce_dims.clear();
    for (auto i = 0; i < x_dims.size(); ++i) {
      reduce_dims.push_back(i);
    }
  }

  for (auto& d : reduce_dims) {
    // handle negative dims, f.e. "-1" means rightmost dimension
    if (d < 0) {
      d = d + x_dims.size();
    }
  }
  auto x_data = x.data<T>();
  auto out_data = dev_ctx.template Alloc<T>(out);

  auto numel = x.numel();
  auto out_dims(x_dims);
  for (auto d : reduce_dims) {
    out_dims[d] = 1;
  }
  std::vector<size_t> index(x_dims.size(), 0);
  std::vector<size_t> step(x_dims.size(), 1);
  std::vector<size_t> dst_step = step;
  for (auto i = x_dims.size() - 1; i > 0; --i) {
    step[i - 1] = step[i] * x_dims[i];
    dst_step[i - 1] = dst_step[i] * out_dims[i];
  }
  for (auto d : reduce_dims) {
    dst_step[d] = 0;
  }
  std::fill(out_data, out_data + out->numel(), std::numeric_limits<T>::max());
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
  if (dims.size() == 0) {
    reduce_all = true;
  }
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
  for (auto& d : reduce_dims) {
    // handle negative dims, f.e. "-1" means rightmost dimension
    if (d < 0) {
      d = d + x_dims.size();
    }
  }
  auto x_data = x.data<T>();
  auto out_data = dev_ctx.template Alloc<T>(out);

  auto numel = x.numel();
  auto out_dims(x_dims);
  for (auto d : reduce_dims) {
    out_dims[d] = 1;
  }
  std::vector<size_t> index(x_dims.size(), 0);
  std::vector<size_t> step(x_dims.size(), 1);
  std::vector<size_t> dst_step = step;
  for (auto i = x_dims.size() - 1; i > 0; --i) {
    step[i - 1] = step[i] * x_dims[i];
    dst_step[i - 1] = dst_step[i] * out_dims[i];
  }
  for (auto d : reduce_dims) {
    dst_step[d] = 0;
  }
  std::fill(
      out_data, out_data + out->numel(), std::numeric_limits<T>::lowest());
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
  if (dims.size() == 0) {
    reduce_all = true;
  }
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
