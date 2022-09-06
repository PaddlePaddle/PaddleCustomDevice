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

#include <CL/sycl.hpp>

#include "oneapi/dnnl/dnnl_sycl.hpp"
#include "paddle/phi/capi/all.h"
#include "phi_funcs.h"
#include "dnn_support.hpp"

namespace custom_kernel {

template <typename T>
void MultiplyRawKernel(const phi::Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& y,
                       int axis,
                       phi::DenseTensor* out) {


  auto x_dims = x.dims();
  auto y_dims = y.dims();
  auto dst_dims = phi::BroadcastDims(axis, x_dims, y_dims);
  phi::DenseTensor tmp_x, tmp_y;
  phi::BroadcastTo<T>(dev_ctx, x, dst_dims, axis, &tmp_x);
  phi::BroadcastTo<T>(dev_ctx, y, dst_dims, axis, &tmp_y);

  auto x_data = tmp_x.data<T>();
  auto y_data = tmp_y.data<T>();
  auto out_data = dev_ctx.template Alloc<T>(out);
  auto numel = out->numel();
  for (auto i = 0; i < numel; ++i) {
    out_data[i] = x_data[i] * y_data[i];
  }
}

template <typename T>
void MultiplyKernel(const phi::Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    phi::DenseTensor* out) {
  int axis = -1;
  MultiplyRawKernel<T>(dev_ctx, x, y, axis, out);
}



template <typename T>
void MultiplyRawKernelGPU(const phi::Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& y,
                       int axis,
                       phi::DenseTensor* out) {

  show_kernel("ElementWise-SYCL-MUL");


  void *stream = const_cast< void* >(dev_ctx.stream());


  // T* out_data = dev_ctx.Alloc<T>(out,out->numel() * sizeof(T));

  T* out_data = dev_ctx.HostAlloc<T>(out);

  auto NOUT = out->numel();

  auto input_x = x.data<T>();
  auto input_y = y.data<T>();

  auto* q = static_cast<sycl::queue*>(stream);
  T* gpu_mem = sycl::malloc_device<T>(out->numel(), *q);

  q->submit([&](sycl::handler& h){

   h.parallel_for(NOUT,[input_x, input_y, gpu_mem ](sycl::id<1> i){

            gpu_mem[i] =  input_x[i]*input_y[i];
            // input_x[i]++;
          //  out_data[i]=3;
   });

  });

  q->wait();
  q->submit([&](sycl::handler &h) {
  // copy hostArray to deviceArray
      h.memcpy(out_data, gpu_mem, sizeof(T) * out->numel() );
   });
  q->wait();

  sycl::free(gpu_mem, *q);

  // std::cout << "stream = " << stream << std::endl;
  // std::cout << "Out_data="<< out_data << std::endl;

/*
  auto x_dims = x.dims();
  auto y_dims = y.dims();



  auto dst_dims = phi::BroadcastDims(axis, x_dims, y_dims);
  phi::DenseTensor tmp_x, tmp_y;
  phi::BroadcastTo<T>(dev_ctx, x, dst_dims, axis, &tmp_x);
  phi::BroadcastTo<T>(dev_ctx, y, dst_dims, axis, &tmp_y);

  auto x_data = tmp_x.data<T>();
  auto y_data = tmp_y.data<T>();
  auto out_data = dev_ctx.template Alloc<T>(out);
  auto numel = out->numel();
  for (auto i = 0; i < numel; ++i) {
    out_data[i] = x_data[i] * y_data[i];
  }


*/

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
  show_kernel("ElementWise-ONEDNN");
  //void* stream = const_cast<void*>(dev_ctx.stream());
  auto* q = static_cast<sycl::queue*>(const_cast<void*>(dev_ctx.stream()));

  if (!q) {

  }

  using namespace dnnl;
  using tag = memory::format_tag;
  using dt = memory::data_type;
  show_debug("X.dims() = " << x.dims() << " Y.dims() =" << y.dims() << " OutDims()=" << out->dims() );
  auto eng = dnnl::sycl_interop::make_engine(q->get_device(), q->get_context());
  auto engine_stream = dnnl::sycl_interop::make_stream(eng, *q);

      dnnl::memory::dims common_dims = {out->numel()};
  auto common_md = memory::desc(common_dims, dnn_support::toDnnType<T>::type, tag::a);

  auto x_mem = memory(common_md, eng, x.data<T>());
  auto y_mem = memory(common_md, eng, y.data<T>());

  auto out_data = dev_ctx.template Alloc<T>(out);


  auto out_mem = memory(common_md, eng, out_data);

  auto oper_desc =
      binary::desc(algorithm::binary_mul, common_md, common_md, common_md);
  auto prim_desc = binary::primitive_desc(oper_desc, eng);
  auto prim = binary(prim_desc);

  std::unordered_map<int, memory> binary_args;
  binary_args.insert({DNNL_ARG_SRC_0, x_mem});
  binary_args.insert({DNNL_ARG_SRC_1, y_mem});
  binary_args.insert({DNNL_ARG_DST, out_mem});

  prim.execute(eng, binary_args);
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



  // template <typename T>
  // void MultiplyKernelGPU(const phi::Context& dev_ctx,
  //                        const phi::DenseTensor& x,
  //                        const phi::DenseTensor& y,
  //                        phi::DenseTensor* out) {
  //   int axis = -1;
  //   if constexpr (std::is_same<T, float>::value ||
  //                 std::is_same<T, int>::value) {
  //     MultiplyOneDNNKernel<T>(dev_ctx, x, y, axis, out);
  //   } else {
  //     MultiplyRawKernelGPU<T>(dev_ctx, x, y, axis, out);
  //   }
  // }

template <typename T>
void MultiplyMainRaw(const phi::Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& y,
                     int axis,
                     phi::DenseTensor* out) {
  // if constexpr (std::is_same<T, float>::value ||
  //               std::is_same<T, int>::value) {
  //   MultiplyOneDNNKernel<T>(dev_ctx, x, y, axis, out);
  // } else {
  //   MultiplyRawKernelGPU<T>(dev_ctx, x, y, axis, out);
  // }

  if constexpr (std::is_same<T, float>::value
               || std::is_same<T, int32_t>::value
             //  || std::is_same<T,double>::value
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
                                 // custom_kernel::MultiplyRawKernelGPU,
                                 custom_kernel::MultiplyMainRaw,
                                 int32_t,
                                 int64_t,
                                 float,
                                 double) {}

             PD_BUILD_PHI_KERNEL(multiply,
                                 intel_gpu,
                                 ALL_LAYOUT,
                                 //  custom_kernel::MultiplyKernelGPU,
                                 custom_kernel::MultiplyMain,
                                 int32_t,
                                 int64_t,
                                 float,
                                 double) {}

             // PD_BUILD_PHI_KERNEL(multiply_raw,
             //                     custom_cpu,
             //                     ALL_LAYOUT,
             //                     custom_kernel::MultiplyRawKernel,
             //                     int32_t,
             //                     int64_t,
             //                     float,
             //                     double) {}

             // PD_BUILD_PHI_KERNEL(multiply,
             //                     custom_cpu,
             //                     ALL_LAYOUT,
             //                     custom_kernel::MultiplyKernel,
             //                     int32_t,
             //                     int64_t,
             //                     float,
             //                     double) {}
