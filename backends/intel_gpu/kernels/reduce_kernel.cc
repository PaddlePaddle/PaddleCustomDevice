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
#include <cmath>

#include "paddle/phi/capi/all.h"
#include "phi_funcs.h"


namespace custom_kernel {

template <typename T>
void ReduceKernel(const phi::Context& dev_ctx,
                  std::string kernel_name,
                  const phi::DenseTensor& x,
                  const std::vector<int64_t>& dims,
                  dnnl::algorithm reduction_type,
                  bool keep_dim,
                  bool reduce_all,
                  phi::DenseTensor* out) {
  auto x_dims = x.dims();
  auto reduce_dims = dims;

  if (reduce_dims.size() == 0) {
    reduce_all = true;
  }

  if (reduce_all) {
    reduce_dims = std::vector<int64_t>(x_dims.size(), 1);
  }
  else {
    auto output_dims(x_dims);
    for (size_t i = 0; i < reduce_dims.size(); ++i) {
      // handle negative dims, f.e. "-1" means rightmost dimension
      int index = (reduce_dims[i] >= 0) ? reduce_dims[i]
                                        : x_dims.size() + reduce_dims[i];
      output_dims[index] = 1;
    }
    reduce_dims = output_dims;
  }
  show_kernel(
    kernel_name << "-Sycl type="  << dnn_support::type2String<T>::name() <<
    ", reduce_all=" << reduce_all << ", x_dims=" << x_dims << ", dims="
     << dims << ", keep_dim=" << keep_dim<< ", reduce_dims=" << reduce_dims);
  void* stream = const_cast<void*>(dev_ctx.stream());
  auto* q = static_cast<sycl::queue*>(const_cast<void*>(dev_ctx.stream()));
  auto out_data = dev_ctx.template Alloc<T>(out);

  if (x_dims == reduce_dims) {
    auto x_data = x.data<T>();
    show_debug(kernel_name << " -> memcpy(to="<< std::hex<< out_data << ", from="<< x_data << ", size="<< std::dec << x.memory_size()<<")");
    q->memcpy(out_data, x_data, x.memory_size());
  }
  else {
    using namespace dnnl;
    using tag = memory::format_tag;
    using dt = memory::data_type;

    auto eng = dnnl::sycl_interop::make_engine(q->get_device(), q->get_context());
    auto engine_stream = dnnl::sycl_interop::make_stream(eng, *q);

    dnnl::memory::dims dims_x = x.dims();
    dnnl::memory::dims dims_out = reduce_dims;

    auto md_x = memory::desc(dims_x,
                              dnn_support::toDnnType<T>::type,
                              dnn_support::dims2Tag(dims_x));

    auto md_out = memory::desc(reduce_dims,
                                dnn_support::toDnnType<T>::type,
                                dnn_support::dims2Tag(reduce_dims));

    auto x_mem = memory(md_x, eng, x.data<T>());

    auto out_mem = memory(md_out, eng, out_data);

    auto oper_desc = dnnl::reduction::desc(reduction_type, md_x, md_out, 0.f, 0.f);
    auto prim_desc = dnnl::reduction::primitive_desc(oper_desc, eng);

    auto reduction_prim = reduction(prim_desc);

    std::unordered_map<int, memory> reduction_args;
    reduction_args.insert({DNNL_ARG_SRC, x_mem});
    reduction_args.insert({DNNL_ARG_DST, out_mem});

    reduction_prim.execute(engine_stream, reduction_args);
    engine_stream.wait();
  }

}

template <typename T>
void MeanRawKernel(const phi::Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const std::vector<int64_t>& dims,
                  bool keep_dim,
                  bool reduce_all,
                  phi::DenseTensor* out) {
  ReduceKernel<T>(dev_ctx, "MeanRaw", x, dims, dnnl::algorithm::reduction_mean,
    keep_dim, reduce_all, out);
}

template <typename T>
void MeanKernel(const phi::Context& dev_ctx,
               const phi::DenseTensor& x,
               const std::vector<int64_t>& dims,
               bool keep_dim,
               phi::DenseTensor* out) {
  bool reduce_all = false;
  MeanRawKernel<T>(dev_ctx, x, dims, keep_dim, reduce_all, out);
}

template <typename T>
void SumRawKernel(const phi::Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const std::vector<int64_t>& dims,
                  bool keep_dim,
                  bool reduce_all,
                  phi::DataType out_dtype,
                  phi::DenseTensor* out) {
  ReduceKernel<T>(dev_ctx, "SumRaw", x, dims, dnnl::algorithm::reduction_sum,
    keep_dim, reduce_all, out);
}

template <typename T>
void SumKernel(const phi::Context& dev_ctx,
               const phi::DenseTensor& x,
               const std::vector<int64_t>& dims,
               phi::DataType out_dtype,
               bool keep_dim,
               phi::DenseTensor* out) {
  bool reduce_all = false;
  SumRawKernel<T>(dev_ctx, x, dims, keep_dim, reduce_all, out_dtype, out);
}

template <typename T>
void MaxRawKernel(const phi::Context& dev_ctx,
               const phi::DenseTensor& x,
               const std::vector<int64_t>& dims,
               bool keep_dim,
               bool reduce_all,
               phi::DenseTensor* out) {
  ReduceKernel<T>(dev_ctx, "MaxRaw", x, dims, dnnl::algorithm::reduction_max,
    keep_dim, reduce_all, out);
}

template <typename T>
void MaxKernel(const phi::Context& dev_ctx,
               const phi::DenseTensor& x,
               const std::vector<int64_t>& dims,
               bool keep_dim,
               phi::DenseTensor* out) {
  bool reduce_all = false;
  MaxRawKernel<T>(dev_ctx, x, dims, keep_dim, reduce_all, out);
}

template <typename T>
void MinRawKernel(const phi::Context& dev_ctx,
               const phi::DenseTensor& x,
               const std::vector<int64_t>& dims,
               bool keep_dim,
               bool reduce_all,
               phi::DenseTensor* out) {
  ReduceKernel<T>(dev_ctx, "MinRaw", x, dims, dnnl::algorithm::reduction_min,
    keep_dim, reduce_all, out);
}

template <typename T>
void MinKernel(const phi::Context& dev_ctx,
               const phi::DenseTensor& x,
               const std::vector<int64_t>& dims,
               bool keep_dim,
               phi::DenseTensor* out) {
  bool reduce_all = false;
  MinRawKernel<T>(dev_ctx, x, dims, keep_dim, reduce_all, out);
}

}  // namespace custom_kernel

PD_BUILD_PHI_KERNEL(
  mean_raw, intel_gpu, ALL_LAYOUT, custom_kernel::MeanRawKernel,float) {}
PD_BUILD_PHI_KERNEL(
  mean, intel_gpu, ALL_LAYOUT, custom_kernel::MeanKernel, float) {}

PD_BUILD_PHI_KERNEL(
  sum_raw, intel_gpu, ALL_LAYOUT, custom_kernel::SumRawKernel,float) {}
PD_BUILD_PHI_KERNEL(
  sum, intel_gpu, ALL_LAYOUT, custom_kernel::SumKernel, float) {}

PD_BUILD_PHI_KERNEL(
  min_raw, intel_gpu, ALL_LAYOUT, custom_kernel::MinRawKernel, float) {}
PD_BUILD_PHI_KERNEL(
  min, intel_gpu, ALL_LAYOUT, custom_kernel::MinKernel, float) {}

PD_BUILD_PHI_KERNEL(
  max_raw, intel_gpu, ALL_LAYOUT, custom_kernel::MaxRawKernel,float) {}
PD_BUILD_PHI_KERNEL(
  max, intel_gpu, ALL_LAYOUT, custom_kernel::MaxKernel, float) {}
