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

#include "dnn_support.hpp"
#include "kernels/kernels.h"
#include "kernels/phi_funcs.h"
#include "paddle/phi/capi/all.h"

namespace custom_kernel {

namespace gpu {

template <typename T>
void Transpose(const phi::Context& ctx,
               const phi::DenseTensor& x,
               const std::vector<int64_t>& axis,
               T* out_data,
               std::vector<int64_t>& out_dims,
               int64_t out_numel) {
  auto x_dims = x.dims();
  auto x_data = x.data<T>();
  show_kernel("TransposeKernel");
  show_debug("x{dims}=" << x.dims() << " x{rank}=" << x_dims.size()
                        << " out{dims}=" << out_dims);

  if (out_numel == 0) {
    return;
  }
  auto rank = x_dims.size();
  if (rank == 1) {
    memcpy(out_data, x_data, x.numel() * sizeof(T));
  }
  PD_CHECK(axis.size() == rank,
           "axis.size (%d) must be equal the rank of input (%d).",
           axis.size(),
           rank);

  std::vector<size_t> step(out_dims.size(), 1);
  for (auto i = out_dims.size() - 1; i > 0; --i) {
    step[i - 1] = step[i] * out_dims[i];
  }

  std::vector<size_t> index(rank, 0);
  for (auto i = 0; i < x.numel(); ++i) {
    std::vector<size_t> dst_index(rank, 0);
    for (auto j = 0; j < rank; ++j) {
      dst_index[j] = index[axis[j]];
    }
    out_data[phi::vec_product(dst_index, step)] = x_data[i];

    index.back()++;
    for (auto j = rank - 1; j > 0; --j) {
      if (index[j] >= x_dims[j]) {
        index[j - 1]++;
        index[j] = 0;
      } else {
        break;
      }
    }
  }
}

template <typename T>
void FullSort(int input_height,
              int input_width,
              int input_dim,
              T* input,
              T* t_out,
              int64_t* t_indices,
              bool descending) {
  for (int i = 0; i < input_height; ++i) {
    std::vector<std::pair<T, int>> col_vec;
    col_vec.reserve(input_width);
    if (input_dim == 1) {
      for (int j = 0; j < input_width; ++j) {
        col_vec.push_back(std::pair<T, int>(input[j], j));
      }
    } else {
      for (int j = 0; j < input_width; ++j) {
        col_vec.push_back(std::pair<T, int>(input[i * input_width + j], j));
      }
    }
    std::sort(col_vec.begin(),
              col_vec.end(),
              [&](const std::pair<T, int>& l, const std::pair<T, int>& r) {
                if (descending)
                  // TODO(Zhiwei35):comparison with NaN always evaluates to
                  // false in fast floating point modes and need to enhance
                  return (std::isnan(static_cast<double>(l.first)) &&
                          !std::isnan(static_cast<double>(r.first))) ||
                         (l.first > r.first);
                else
                  return (!std::isnan(static_cast<double>(l.first)) &&
                          std::isnan(static_cast<double>(r.first))) ||
                         (l.first < r.first);
              });
    for (int j = 0; j < input_width; ++j) {
      t_out[i * input_width + j] = col_vec[j].first;
      t_indices[i * input_width + j] = col_vec[j].second;
    }
  }
}

template <typename T>
void ArgsortKernel(const phi::Context& dev_ctx,
                   const phi::DenseTensor& input,
                   int axis,
                   bool descending,
                   phi::DenseTensor* output,
                   phi::DenseTensor* indices) {
  auto in_dims = input.dims();
  auto out_dims = output->dims();
  auto out_size = output->numel();
  auto ids_size = indices->numel();
  auto out_mem_size = out_size * sizeof(T);
  auto ids_mem_size = ids_size * sizeof(int64_t);
  axis = (axis < 0) ? (in_dims.size() + axis) : axis;
  T* out_data = dev_ctx.template Alloc<T>(output);
  int64_t* ids_data = dev_ctx.template Alloc<int64_t>(indices);

  show_kernel("argsort in_dims=" << in_dims << " axis=" << axis << " type="
                                 << dnn_support::type2String<T>::name()
                                 << " desc=" << descending);
  // TODO(Zhiwei35): support argsort with dims >=3
  PD_CHECK(in_dims.size() < 3, "PoC Lenet/Mnist use case only");
  auto* q = static_cast<sycl::queue*>(const_cast<void*>(dev_ctx.stream()));
  size_t n = 1;
  size_t m = in_dims[0];

  if (in_dims.size() == 2) {
    n = in_dims[0];
    m = in_dims[1];
  }
  phi::DenseTensor cpu_input;
  cpu_input.Resize(std::vector<int64_t>(in_dims));
  cpu_input.set_dtype(input.dtype());
  auto cpu_input_data = dev_ctx.template HostAlloc<T>(&cpu_input);

  auto input_data = input.data<T>();
  q->memcpy(cpu_input_data, input_data, input.memory_size());
  q->wait();
  // cpu implement
  phi::DenseTensor cpu_output;
  cpu_output.Resize(std::vector<int64_t>(out_dims));
  cpu_output.set_dtype(output->dtype());
  auto cpu_output_dims = cpu_output.dims();
  auto cpu_output_numel = cpu_output.numel();
  auto cpu_output_data = dev_ctx.template HostAlloc<T>(&cpu_output);

  phi::DenseTensor cpu_ids;
  cpu_ids.Resize(std::vector<int64_t>(indices->dims()));
  cpu_ids.set_dtype(indices->dtype());
  auto cpu_ids_dims = cpu_ids.dims();
  auto cpu_ids_numel = cpu_ids.numel();
  auto cpu_ids_data = dev_ctx.template HostAlloc<int64_t>(&cpu_ids);
  // no need transpose
  if (axis == -1 || axis + 1 == in_dims.size()) {
    const int input_height = n;
    const int input_width = m;
    FullSort<T>(input_height,
                input_width,
                in_dims.size(),
                cpu_input_data,
                cpu_output_data,
                cpu_ids_data,
                descending);
  } else {
    // do cpu transpose
    std::vector<int64_t> trans;
    for (int i = 0; i < axis; i++) {
      trans.push_back(i);
    }
    trans.push_back(in_dims.size() - 1);
    for (int i = axis + 1; i < in_dims.size() - 1; i++) {
      trans.push_back(i);
    }
    trans.push_back(axis);
    std::vector<int64_t> trans_dims(in_dims.cbegin(), in_dims.cend());
    for (size_t i = 0; i < trans.size(); i++) {
      trans_dims[i] = in_dims[trans[i]];
    }

    phi::DenseTensor trans_inp;
    trans_inp.Resize(trans_dims);
    auto trans_input_dims = trans_inp.dims();
    auto trans_input_numel = trans_inp.numel();
    auto trans_input_data = dev_ctx.template HostAlloc<T>(&trans_inp);
    // do cpu transpose input
    Transpose<T>(dev_ctx,
                 cpu_input,
                 trans,
                 trans_input_data,
                 trans_input_dims,
                 trans_input_numel);

    const int64_t input_height = trans_dims[0];
    const int64_t input_width = trans_dims[trans_dims.size() - 1];

    phi::DenseTensor cpu_tmp_output;
    cpu_tmp_output.Resize(trans_dims);
    cpu_tmp_output.set_dtype(output->dtype());
    auto cpu_tmp_output_data = dev_ctx.template HostAlloc<T>(&cpu_tmp_output);

    phi::DenseTensor cpu_tmp_ids;
    cpu_tmp_ids.Resize(trans_dims);
    cpu_tmp_ids.set_dtype(indices->dtype());
    auto cpu_tmp_ids_data = dev_ctx.template HostAlloc<int64_t>(&cpu_tmp_ids);

    FullSort<T>(input_height,
                input_width,
                trans_dims.size(),
                trans_input_data,
                cpu_tmp_output_data,
                cpu_tmp_ids_data,
                descending);

    Transpose<int64_t>(
        dev_ctx, cpu_tmp_ids, trans, cpu_ids_data, cpu_ids_dims, cpu_ids_numel);
    // CPU transpose back
    Transpose<T>(dev_ctx,
                 cpu_tmp_output,
                 trans,
                 cpu_output_data,
                 cpu_output_dims,
                 cpu_output_numel);
  }
  // copy cpu result to intel gpu
  q->memcpy(out_data, cpu_output_data, out_mem_size);
  q->memcpy(ids_data, cpu_ids_data, ids_mem_size);
  q->wait();

}  // ArgsortKernel

}  // namespace gpu

}  // namespace custom_kernel

PD_BUILD_PHI_KERNEL(argsort,
                    intel_gpu,
                    ALL_LAYOUT,
                    custom_kernel::gpu::ArgsortKernel,
                    float,
                    double,
                    int,
                    int64_t) {}
