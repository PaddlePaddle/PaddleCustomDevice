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
  axis = (axis < 0) ? (in_dims.size() + axis) : axis;
  T* out_data = dev_ctx.template Alloc<T>(output);

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




namespace gpu {

template <typename T>
void ArgsortKernel(const phi::Context& dev_ctx,
                   const phi::DenseTensor& input,
                   int axis,
                   bool descending,
                   phi::DenseTensor* output,
                   phi::DenseTensor* indices) {
  auto in_dims = input.dims();
  axis = (axis < 0) ? (in_dims.size() + axis) : axis;
  T* out_data = dev_ctx.template Alloc<T>(output);

  int64_t* ids_data = dev_ctx.template Alloc<int64_t>(indices);
  show_kernel("argsort in_dims=" << in_dims << " axis="<< axis << " type="<< dnn_support::type2String<T>::name() << " desc="<< descending );

  PD_CHECK(in_dims.size()<3, "PoC Lenet/Mnist use case only");

  using namespace oneapi::dpl::execution;
  using namespace oneapi::dpl;
  auto* q = static_cast<sycl::queue*>(const_cast<void*>(dev_ctx.stream()));
  auto policy_e = make_device_policy(*q);

  size_t n = 1;
  size_t m = in_dims[0];

  if(in_dims.size()==2)
  {
     n = in_dims[0];
     m = in_dims[1];
  }

  auto input_data = input.data<T>();
  q->memcpy(out_data,input_data, input.memory_size() );
  q->wait();

  for(size_t i=0;i<n;i++)
  {

   q->parallel_for(m, [p_data=ids_data + i*m,m](auto& i){
           p_data[i] = i;
   });

   q->wait();

   sycl::buffer<int64_t> keys_buf{reinterpret_cast<int64_t*>(ids_data + i*m),sycl::range<1>(m)};
   sycl::buffer<T> vals_buf{reinterpret_cast<T*>(out_data + i*m),sycl::range<1>(m)};

   auto keys_begin = oneapi::dpl::begin(keys_buf);
   auto vals_begin = oneapi::dpl::begin(vals_buf);
   auto zipped_begin = dpl::make_zip_iterator(keys_begin, vals_begin);

    // gpu sort
     std::stable_sort(policy_e, zipped_begin, zipped_begin + m,[descending](auto lhs, auto rhs) {
        return (descending)? (get<1>(lhs) > get<1>(rhs)) :  (get<1>(lhs) < get<1>(rhs));
     });

   }

} // ArgsortKernel

} // gpu

}  // namespace custom_kernel

PD_BUILD_PHI_KERNEL(argsort,
                    custom_cpu,
                    ALL_LAYOUT,
                    custom_kernel::ArgsortKernel,
                    float,
                    double,
                    int,
                    int64_t) {}


PD_BUILD_PHI_KERNEL(argsort,
                    intel_gpu,
                    ALL_LAYOUT,
                    custom_kernel::gpu::ArgsortKernel,
                    float,
                    double,
                    int,
                    int64_t
                    ) {}
