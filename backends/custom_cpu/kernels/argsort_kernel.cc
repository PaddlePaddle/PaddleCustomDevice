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

#include "kernels.h"
#include "paddle/phi/capi/all.h"
#include "phi_funcs.h"

namespace custom_kernel {

template <typename T, typename Type>
static void FullSort(Type input_height,
                     Type input_width,
                     int input_dim,
                     const phi::DenseTensor* input,
                     T* t_out,
                     Type* t_indices,
                     bool descending) {
  for (Type i = 0; i < input_height; ++i) {
    std::vector<std::pair<T, Type>> col_vec;
    col_vec.reserve(input_width);
    auto e_input = input->data<T>();
    if (input_dim == 1) {
      for (Type j = 0; j < input_width; ++j) {
        col_vec.push_back(std::pair<T, Type>(e_input[j], j));
      }
    } else {
      for (Type j = 0; j < input_width; ++j) {
        col_vec.push_back(std::pair<T, Type>(e_input[i * input_width + j], j));
      }
    }
    std::sort(col_vec.begin(),
              col_vec.end(),
              [&](const std::pair<T, Type>& l, const std::pair<T, Type>& r) {
                if (descending)
                  return (std::isnan(static_cast<double>(l.first)) &&
                          !std::isnan(static_cast<double>(r.first))) ||
                         (l.first > r.first);
                else
                  return (!std::isnan(static_cast<double>(l.first)) &&
                          std::isnan(static_cast<double>(r.first))) ||
                         (l.first < r.first);
              });

    for (Type j = 0; j < input_width; ++j) {
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
  auto rank = in_dims.size();
  axis = (axis < 0) ? (in_dims.size() + axis) : axis;
  T* out_data = dev_ctx.template Alloc<T>(output);

  if (rank == 0) {
    auto numel = input.numel();
    if (numel == 0) {
      return;
    }
    T* in_data = input.data<T>();
    // Actually, numel = 1 if rank == 0.
    for (auto i = 0; i < numel; ++i) {
      out_data[i] = in_data[i];
    }
    int64_t* ids_data = dev_ctx.template Alloc<int64_t>(indices);
    ids_data[0] = 0;
    return;
  }

  // Do full sort
  if (axis == -1 || axis + 1 == in_dims.size()) {
    const int64_t input_height =
        phi::product(phi::slice_ddim(in_dims, 0, in_dims.size() - 1));
    const int64_t input_width = in_dims[in_dims.size() - 1];
    int64_t* ids_data = dev_ctx.template Alloc<int64_t>(indices);
    FullSort<T, int64_t>(input_height,
                         input_width,
                         in_dims.size(),
                         &input,
                         out_data,
                         ids_data,
                         descending);
  } else {
    // If not full sort do transpose
    std::vector<int> trans;
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
    dev_ctx.template Alloc<T>(&trans_inp);
    // Do transpose
    TransposeKernel<T>(dev_ctx, input, trans, &trans_inp);

    const int64_t input_height =
        phi::product(phi::slice_ddim(trans_dims, 0, trans_dims.size() - 1));
    const int64_t input_width = trans_dims[trans_dims.size() - 1];

    phi::DenseTensor tmp_out;
    tmp_out.Resize(trans_dims);
    T* t_out = dev_ctx.template Alloc<T>(&tmp_out);

    phi::DenseTensor tmp_indices;
    tmp_indices.Resize(trans_dims);
    auto* t_ind = dev_ctx.template Alloc<int64_t>(&tmp_indices);

    FullSort<T, int64_t>(input_height,
                         input_width,
                         in_dims.size(),
                         &trans_inp,
                         t_out,
                         t_ind,
                         descending);

    dev_ctx.template Alloc<int64_t>(indices);
    TransposeKernel<int64_t>(dev_ctx, tmp_indices, trans, indices);
    // transpose back
    TransposeKernel<T>(dev_ctx, tmp_out, trans, output);
  }
}

}  // namespace custom_kernel

PD_BUILD_PHI_KERNEL(argsort,
                    custom_cpu,
                    ALL_LAYOUT,
                    custom_kernel::ArgsortKernel,
                    float,
                    double,
                    int,
                    int64_t) {}
