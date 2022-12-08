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
#include "paddle/phi/capi/all.h"
#include "phi_funcs.h"
namespace custom_kernel {

template <typename T>
T ValueClip(const T& x) {
  const T kThreshold = static_cast<T>(-64.);
  return x < kThreshold ? kThreshold : x;
}

template <typename T>
void Softmax(int axis_dim, const T* in, T* out, size_t M, size_t N) {
  auto remain = N / axis_dim;

  for (size_t i = 0; i < M; ++i) {
    for (size_t k = 0; k < remain; ++k) {
      T max_val = in[i * N + k];
      for (size_t j = 0; j < axis_dim; ++j) {
        max_val = std::max(max_val, in[i * N + j * remain + k]);
      }

      auto exps = new T[axis_dim];
      for (size_t j = 0; j < axis_dim; ++j) {
        exps[j] = std::exp(ValueClip(in[i * N + j * remain + k] - max_val));
      }

      T sum = 0;
      for (size_t j = 0; j < axis_dim; ++j) {
        sum += exps[j];
      }

      for (size_t j = 0; j < axis_dim; ++j) {
        out[i * N + j * remain + k] = exps[j] / sum;
      }
      delete[] exps;
    }
  }
}

template <typename T>
void SoftmaxGrad(
    const T* out, const T* out_grad, int axis_dim, int M, int N, T* x_grad) {
  int num_remain = N / axis_dim;
  T* dot = new T[M * num_remain];
  for (auto i = 0; i < M; ++i) {
    for (auto k = 0; k < num_remain; ++k) {
      dot[i * num_remain + k] = 0;
      for (auto j = 0; j < axis_dim; ++j) {
        dot[i * num_remain + k] += out[i * N + j * num_remain + k] *
                                   out_grad[i * N + j * num_remain + k];
      }
    }
  }
  for (auto i = 0; i < M; ++i) {
    for (auto j = 0; j < axis_dim; ++j) {
      for (auto k = 0; k < num_remain; ++k) {
        x_grad[i * N + j * num_remain + k] =
            (out_grad[i * N + j * num_remain + k] - dot[i * num_remain + k]) *
            out[i * N + j * num_remain + k];
      }
    }
  }
  delete[] dot;
}


std::shared_ptr<dnnl::softmax_forward::primitive_desc> softmax_pd = nullptr;

template <typename T>
void SoftmaxGradKernel(const phi::Context& dev_ctx,
                       const phi::DenseTensor& out,
                       const phi::DenseTensor& out_grad,
                       int axis,
                       phi::DenseTensor* x_grad) {
  show_kernel("SoftmaxGradKernel()");
  const int rank = x_grad->dims().size();
  const int calc_axis = phi::funcs::CanonicalAxis(axis, rank);
  int axis_dim = x_grad->dims()[calc_axis];

  dev_ctx.template Alloc<T>(x_grad);
  if (x_grad->numel() == 0) {
    return;
  }


  using namespace dnnl;
  auto* q = static_cast<sycl::queue*>(const_cast<void*>(dev_ctx.stream()));

  auto eng = dnnl::sycl_interop::make_engine(q->get_device(), q->get_context());
  auto engine_stream = dnnl::sycl_interop::make_stream(eng, *q);

  dnnl::memory::dims out_dims = out.dims();

  std::vector<int> logical_axis(out_dims.size(), 0);
  for (auto i = 0; i < logical_axis.size(); ++i) {
    logical_axis[i] = i;
  }

  auto strides =
      dnn_support::computeStrides(out_dims, logical_axis);

  auto md_out =
        memory::desc(out_dims, dnn_support::toDnnType<T>::type, strides);

  auto md_out_grad =
      memory::desc(out_dims, dnn_support::toDnnType<T>::type, strides);

  auto md_x_grad =
      memory::desc(out_dims, dnn_support::toDnnType<T>::type,strides);

  auto dst_memory_p = memory(md_out, eng, out.data<T>());
  auto diff_dst_memory_p = memory(md_out_grad, eng, out_grad.data<T>());
  auto diff_src_memory_p = memory(md_x_grad, eng, x_grad->data<T>());

  auto bwd_desc = softmax_backward::desc(md_out_grad, md_out, calc_axis);
  auto bwd_pd_ = softmax_backward::primitive_desc(bwd_desc, eng, *softmax_pd);

  auto softmax_bwd_p = softmax_backward(bwd_pd_);

  std::unordered_map<int, memory> softmax_args;
  softmax_args.insert({DNNL_ARG_DST,  dst_memory_p});
  softmax_args.insert({DNNL_ARG_DIFF_DST, diff_dst_memory_p});
  softmax_args.insert({DNNL_ARG_DIFF_SRC, diff_src_memory_p});

  softmax_bwd_p.execute(engine_stream, softmax_args);
  engine_stream.wait();

}


template <typename T>
void SoftmaxKernel(const phi::Context& ctx,
                   const phi::DenseTensor& x,
                   int axis,
                   phi::DenseTensor* out) {


  if constexpr (std::is_same<T,float>::value) {

  const int rank = x.dims().size();
  const int calc_axis = phi::funcs::CanonicalAxis(axis, rank);
  int axis_dim = x.dims()[calc_axis];
  show_kernel("SoftmaxKernelOneDNN() rank=" << rank << " calc_axis=" << calc_axis
                                      << " axis_dim=" << axis_dim << " type="<< dnn_support::type2String<T>::name());


  const int n = phi::funcs::SizeToAxis(calc_axis, x.dims());
  const int d = phi::funcs::SizeFromAxis(calc_axis, x.dims());

  auto x_data = x.data<T>();
  auto out_data = ctx.template Alloc<T>(out);

  dnnl::memory::dims dims_src = x.dims();
  dnnl::memory::dims dims_dst = out->dims();

  using namespace dnnl;
  using tag = memory::format_tag;
  using dt = memory::data_type;
  auto* q = static_cast<sycl::queue*>(const_cast<void*>(ctx.stream()));

  if (!q) {
  }

  auto eng = dnnl::sycl_interop::make_engine(q->get_device(), q->get_context());
  auto engine_stream = dnnl::sycl_interop::make_stream(eng, *q);

  std::vector<int> logical_axis(dims_src.size(), 0);
  for (auto i = 0; i < logical_axis.size(); ++i) {
    logical_axis[i] = i;
  }

  auto strides =
      dnn_support::computeStrides(dims_src, logical_axis);

   auto md_src =
          memory::desc(dims_src,
                       dnn_support::toDnnType<T>::type,
                       strides);

  auto md_dst =
      memory::desc(dims_src,
                   dnn_support::toDnnType<T>::type,
                   strides);

  show_debug("ComputeStrides = " << strides);

   auto mem_src = memory(md_src, eng, x_data);
  auto mem_dst = memory(md_dst, eng, out_data);

  auto softmax_d =
      softmax_forward::desc(prop_kind::forward_training, md_src, calc_axis);

   softmax_pd = std::make_shared<softmax_forward::primitive_desc>(softmax_d, eng);

   auto softmax_prim = softmax_forward(*softmax_pd);
   std::unordered_map<int, memory> softmax_args;
   softmax_args.insert({DNNL_ARG_SRC, mem_src});
   softmax_args.insert({DNNL_ARG_DST,  mem_dst});

   // // Primitive execution.
    softmax_prim.execute(engine_stream, softmax_args);
   // Wait for the computation to finalize.
    engine_stream.wait();


  } else {

   std::stringstream ss;
   ss << "SoftMax doesn't support type="
      << dnn_support::type2String<T>::name();

   show_error(ss.str());
   throw std::runtime_error(ss.str());

  }




}

}  // namespace custom_kernel

// PD_BUILD_PHI_KERNEL(softmax,
//                     intel_gpu,
//                     ALL_LAYOUT,
//                     custom_kernel::SoftmaxKernel,
//                     float,
//                     double) {}

PD_BUILD_PHI_KERNEL(softmax,
                    intel_gpu,
                    ALL_LAYOUT,
                    custom_kernel::SoftmaxKernel,
                    float,
                    double
                    ) {}



PD_BUILD_PHI_KERNEL(softmax_grad,
                    intel_gpu,
                    ALL_LAYOUT,
                    custom_kernel::SoftmaxGradKernel,
                    float
                    // , double
                    ) {}
