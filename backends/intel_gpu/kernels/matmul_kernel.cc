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
#include "kernels/kernels.h"
#include "kernels/phi_funcs.h"
#include "paddle/phi/capi/all.h"

namespace custom_kernel {

namespace gpu {

template <class T>
std::vector<int64_t> convert_transpose(const phi::Context& dev_ctx,
                                       const phi::DenseTensor& in,
                                       phi::DenseTensor* out) {
  std::vector<int64_t> out_dims = in.dims();

  std::vector<int> trans_axes(out_dims.size());
  std::iota(trans_axes.begin(), trans_axes.end(), 0);

  if (out_dims.size() > 1) {
    std::swap(*(trans_axes.rbegin()), *(trans_axes.rbegin() + 1));
    std::swap(*(out_dims.rbegin()), *(out_dims.rbegin() + 1));
  }

  out->Resize(out_dims);

  custom_kernel::TransposeKernelGPU<T>(dev_ctx, in, trans_axes, out);

  return out_dims;
}

template <typename T>
void MatmulKernel(const phi::Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& y,
                  bool transpose_x,
                  bool transpose_y,
                  phi::DenseTensor* out) {
  show_kernel("matmul-dnn type=" << dnn_support::type2String<T>::name());

  auto x_dims = x.dims();
  auto y_dims = y.dims();
  auto x_data = x.data<T>();
  auto y_data = y.data<T>();
  auto x_ndim = x_dims.size();
  auto y_ndim = y_dims.size();

  show_debug("all_inputs => type="
             << dnn_support::type2String<T>::name() << " x_dims=" << x.dims()
             << " transX=" << transpose_x << " y_dims=" << y.dims()
             << " transY=" << transpose_y << " x_ndim=" << x_ndim
             << " y_ndim=" << y_ndim << " out_dims=" << out->dims());

  auto* q = static_cast<sycl::queue*>(const_cast<void*>(dev_ctx.stream()));

  if (x_ndim == 1 && y_ndim == 1) {
    PD_CHECK(x_dims[0] == y_dims[0], "Vector size must be equal");
    phi::DenseTensor dotprod;
    dotprod.Resize({x_dims[0]});
    T* dotprod_mem = dev_ctx.template Alloc<T>(&dotprod);

    q->submit([&](sycl::handler& h) {
      h.parallel_for(x_dims[0], [y_data, x_data, dotprod_mem](sycl::id<1> i) {
        dotprod_mem[i] = x_data[i] * y_data[i];
      });
    });

    q->wait();
    auto out_data = dev_ctx.template Alloc<T>(out);
    auto numel = x_dims[0];
    q->single_task([numel, dotprod_mem, out_data]() {
      for (auto i = 0; i < numel; ++i) {
        *out_data += dotprod_mem[i];
      }
    });

    q->wait();

    return;
  }

  if (x_ndim > 1 && x_ndim == y_ndim) {
    auto itx = x_dims.rbegin();
    auto ity = y_dims.rbegin() + 1;

    if (transpose_x) {
      ++itx;
    }

    if (transpose_y) {
      --ity;
    }

    PD_CHECK(*itx == *ity, "M must be equal to N.");

    auto eq = std::equal(
        x_dims.begin(), x_dims.begin() + (x_ndim - 2), y_dims.begin());

    PD_CHECK(eq, "dims are not correct to use matmul");
  }

  // using namespace dnnl;
  using tag = dnnl::memory::format_tag;
  using dt = dnnl::memory::data_type;

  auto eng = dnnl::sycl_interop::make_engine(q->get_device(), q->get_context());
  auto engine_stream = dnnl::sycl_interop::make_stream(eng, *q);

  dnnl::memory::dims dims_x = x.dims();
  dnnl::memory::dims dims_y = y.dims();
  dnnl::memory::dims dims_out = out->dims();

  std::array<phi::DenseTensor, 2> transposed_mem;

  if (transpose_x) {
    show_debug("Transpose X");
    dims_x = convert_transpose<T>(dev_ctx, x, &transposed_mem[0]);
    show_debug("Transpose X got shape=" << dims_x);
  }

  if (transpose_y) {
    show_debug("Transpose Y");
    dims_y = convert_transpose<T>(dev_ctx, y, &transposed_mem[1]);
    show_debug("Transpose Y got shape=" << dims_x);
  }

  auto md_y = dnnl::memory::desc(
      dims_y, dnn_support::toDnnType<T>::type, dnn_support::dims2Tag(dims_y));

  auto md_x = dnnl::memory::desc(
      dims_x, dnn_support::toDnnType<T>::type, dnn_support::dims2Tag(dims_x));

  auto md_out = dnnl::memory::desc(dims_out,
                                   dnn_support::toDnnType<T>::type,
                                   dnn_support::dims2Tag(dims_out));

  auto memory_ptr_x = (transpose_x) ? transposed_mem[0].data<T>() : x.data<T>();
  auto memory_ptr_y = (transpose_y) ? transposed_mem[1].data<T>() : y.data<T>();

  auto x_mem = dnnl::memory(md_x, eng, memory_ptr_x);
  auto y_mem = dnnl::memory(md_y, eng, memory_ptr_y);

  auto out_data = dev_ctx.template Alloc<T>(out);

  auto out_mem = dnnl::memory(md_out, eng, out_data);

  auto mat_desc = dnnl::matmul::desc(md_x, md_y, md_out);
  auto prim_desc = dnnl::matmul::primitive_desc(mat_desc, eng);

  auto prim = dnnl::matmul(prim_desc);

  std::unordered_map<int, dnnl::memory> matmul_args;
  matmul_args.insert({DNNL_ARG_SRC_0, x_mem});
  matmul_args.insert({DNNL_ARG_WEIGHTS, y_mem});
  matmul_args.insert({DNNL_ARG_DST, out_mem});

  prim.execute(engine_stream, matmul_args);
  engine_stream.wait();
}

template <typename T>
void MatmulGradKernel(const phi::Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& y,
                      const phi::DenseTensor& out_grad,
                      bool transpose_x,
                      bool transpose_y,
                      phi::DenseTensor* dx,
                      phi::DenseTensor* dy) {
  show_kernel("matmul-dnn-grad type=" << dnn_support::type2String<T>::name()
                                      << " dx=" << (dx) << " dy=" << (dy));

  if (dx) {
    // dx = dout * y'
    auto dx_data = dev_ctx.template Alloc<T>(dx);

    MatmulKernel<T>(dev_ctx, out_grad, y, false, true, dx);
  }

  if (dy) {
    // dy = x' * dout
    auto dy_data = dev_ctx.template Alloc<T>(dy);

    MatmulKernel<T>(dev_ctx, x, out_grad, true, false, dy);
  }
}  // gpu::MatmulGradKernel
}  // namespace gpu
}  // namespace custom_kernel

PD_BUILD_PHI_KERNEL(
    matmul, intel_gpu, ALL_LAYOUT, custom_kernel::gpu::MatmulKernel, float) {}

PD_BUILD_PHI_KERNEL(matmul_grad,
                    intel_gpu,
                    ALL_LAYOUT,
                    custom_kernel::gpu::MatmulGradKernel,
                    float) {}
