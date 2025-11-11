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

template <typename T>
void MultiplyRawKernelGPU(const phi::Context& dev_ctx,
                          const phi::DenseTensor& x,
                          const phi::DenseTensor& y,
                          int axis,
                          phi::DenseTensor* out) {
  show_kernel(
      "ElementWise-SYCL-MUL type=" << dnn_support::type2String<T>::name());
  void* stream = const_cast<void*>(dev_ctx.stream());
  auto* q = static_cast<sycl::queue*>(stream);

  T* out_data = dev_ctx.Alloc<T>(out);

  auto NOUT = out->numel();

  auto input_x = x.data<T>();
  auto input_y = y.data<T>();

  q->submit([&](sycl::handler& h) {
    h.parallel_for(NOUT, [input_x, input_y, out_data](sycl::id<1> i) {
      out_data[i] = input_x[i] * input_y[i];
    });
  });

  q->wait();
}

template <typename T>
void MultiplyKernelGPU(const phi::Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& y,
                       phi::DenseTensor* out) {
  int axis = -1;
  MultiplyRawKernelGPU<T>(dev_ctx, x, y, axis, out);
}

template <typename T>
void MultiplyOneDNNRawKernel(const phi::Context& dev_ctx,
                             const phi::DenseTensor& x,
                             const phi::DenseTensor& y,
                             int axis,
                             phi::DenseTensor* out) {
  show_kernel(
      "ElementWise-ONEDNN type=" << dnn_support::type2String<T>::name());
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

  auto oper_desc =
      dnnl::binary::desc(dnnl::algorithm::binary_mul, md_x, md_y, md_out);
  auto prim_desc = dnnl::binary::primitive_desc(oper_desc, eng);
  auto prim = dnnl::binary(prim_desc);

  std::unordered_map<int, dnnl::memory> binary_args;
  binary_args.insert({DNNL_ARG_SRC_0, x_mem});
  binary_args.insert({DNNL_ARG_SRC_1, y_mem});
  binary_args.insert({DNNL_ARG_DST, out_mem});

  prim.execute(engine_stream, binary_args);
  engine_stream.wait();
}

template <typename T>
void MultiplyOneDNNKernel(const phi::Context& dev_ctx,
                          const phi::DenseTensor& x,
                          const phi::DenseTensor& y,
                          phi::DenseTensor* out) {
  int axis = -1;
  MultiplyOneDNNRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T>
void MultiplyMainRaw(const phi::Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& y,
                     int axis,
                     phi::DenseTensor* out) {
  if constexpr (std::is_same<T, float>::value || std::is_same<T, int32_t>::value
                //|| std::is_same<T,double>::value
  ) {
    MultiplyOneDNNRawKernel<T>(dev_ctx, x, y, axis, out);
  } else {
    MultiplyRawKernelGPU<T>(dev_ctx, x, y, axis, out);
  }
}
template <typename T>
void MultiplyMain(const phi::Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& y,
                  phi::DenseTensor* out) {
  int axis = -1;
  MultiplyMainRaw<T>(dev_ctx, x, y, axis, out);
}

}  // namespace custom_kernel

PD_BUILD_PHI_KERNEL(multiply_raw,
                    intel_gpu,
                    ALL_LAYOUT,
                    custom_kernel::MultiplyMainRaw,
                    int32_t,
                    int64_t,
                    float,
                    double) {}

PD_BUILD_PHI_KERNEL(multiply,
                    intel_gpu,
                    ALL_LAYOUT,
                    custom_kernel::MultiplyMain,
                    int32_t,
                    int64_t,
                    float,
                    double) {}
