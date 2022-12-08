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

#include "paddle/phi/capi/all.h"
#include "phi_funcs.h"
#include "perf_lib_layer_params.h"
#include "hpu_operator.h"

namespace custom_kernel {

class SoftmaxOperator : public HpuOperator {
  public:
  SoftmaxOperator() : HpuOperator("softmax_fwd_f32") {}
  void AddNode(const std::vector<DIMS>& ins,
              const std::vector<DIMS>& outs, int axis_dim) {
    assert(ins.size() == 1 && "input size should be 1");
    assert(outs.size() == 1 && "output size should be 1");

    synTensor inputs[1]  = {createTensor(ins[0].size(), syn_type_float, ins[0], true, "input")};
    synTensor outputs[1] = {createTensor(outs[0].size(), syn_type_float, outs[0], true, "output")};
    ns_Softmax::Params params{ins[0].size() - 1 - axis_dim};
    synStatus status = synNodeCreate(graphHandle_, inputs, outputs, ins.size(), outs.size(), &params, sizeof(params), "softmax_fwd_f32", "softmax_op", nullptr, nullptr);
    CHKSTATUS("synNodeCreate reshape failed!");
  }
};

// template <typename T>
// T ValueClip(const T& x) {
//   const T kThreshold = static_cast<T>(-64.);
//   return x < kThreshold ? kThreshold : x;
// }

// template <typename T>
// void Softmax(int axis_dim, const T* in, T* out, size_t M, size_t N) {
//   auto remain = N / axis_dim;

//   for (size_t i = 0; i < M; ++i) {
//     for (size_t k = 0; k < remain; ++k) {
//       T max_val = in[i * N + k];
//       for (size_t j = 0; j < axis_dim; ++j) {
//         max_val = std::max(max_val, in[i * N + j * remain + k]);
//       }

//       auto exps = new T[axis_dim];
//       for (size_t j = 0; j < axis_dim; ++j) {
//         exps[j] = std::exp(ValueClip(in[i * N + j * remain + k] - max_val));
//       }

//       T sum = 0;
//       for (size_t j = 0; j < axis_dim; ++j) {
//         sum += exps[j];
//       }

//       for (size_t j = 0; j < axis_dim; ++j) {
//         out[i * N + j * remain + k] = exps[j] / sum;
//       }
//       delete[] exps;
//     }
//   }
// }

template <typename T>
void SoftmaxKernel(const phi::Context& dev_ctx,
                   const phi::DenseTensor& x,
                   int axis,
                   phi::DenseTensor* out) {
  const int rank = x.dims().size();
  const int calc_axis = phi::funcs::CanonicalAxis(axis, rank);
  int axis_dim = x.dims()[calc_axis];
  // allocate memory on device.
  dev_ctx.template Alloc<T>(out);
  if (out->numel() == 0) {
    return;
  }

  SoftmaxOperator op;
  op.AddNode({x.dims()}, {out->dims()}, calc_axis);
  std::map<std::string, uint64_t> tensors;
  tensors["input"] = reinterpret_cast<uint64_t> (x.data<T>());
  tensors["output"] = reinterpret_cast<uint64_t> (out->data<T>());
  op.CompileAndExecute(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);

  // const int n = phi::funcs::SizeToAxis(calc_axis, x.dims());
  // const int d = phi::funcs::SizeFromAxis(calc_axis, x.dims());
  // Softmax(axis_dim, x.data<T>(), out->data<T>(), n, d);
}

// template <typename T>
// void SoftmaxGrad(
//     const T* out, const T* out_grad, int axis_dim, int M, int N, T* x_grad) {
//   int num_remain = N / axis_dim;
//   T* dot = new T[M * num_remain];
//   for (auto i = 0; i < M; ++i) {
//     for (auto k = 0; k < num_remain; ++k) {
//       dot[i * num_remain + k] = 0;
//       for (auto j = 0; j < axis_dim; ++j) {
//         dot[i * num_remain + k] += out[i * N + j * num_remain + k] *
//                                    out_grad[i * N + j * num_remain + k];
//       }
//     }
//   }
//   for (auto i = 0; i < M; ++i) {
//     for (auto j = 0; j < axis_dim; ++j) {
//       for (auto k = 0; k < num_remain; ++k) {
//         x_grad[i * N + j * num_remain + k] =
//             (out_grad[i * N + j * num_remain + k] - dot[i * num_remain + k]) *
//             out[i * N + j * num_remain + k];
//       }
//     }
//   }
//   delete[] dot;
// }

class SoftmaxGradOperator : public HpuOperator {
  public:
  SoftmaxGradOperator() : HpuOperator("softmax_bwd_f32") {}
  void AddNode(const std::vector<DIMS>& ins,
              const std::vector<DIMS>& outs, int axis_dim) {
    assert(ins.size() == 2 && "input size should be 2");
    assert(outs.size() == 1 && "output size should be 1");

    synTensor inputs[2]  = {createTensor(ins[0].size(), syn_type_float, ins[0], true, "input_1"),
                            createTensor(ins[1].size(), syn_type_float, ins[1], true, "input_2")};
    synTensor outputs[1] = {createTensor(outs[0].size(), syn_type_float, outs[0], true, "output")};
    ns_Softmax::Params params{ins[0].size() - 1 - axis_dim};
    synStatus status = synNodeCreate(graphHandle_, inputs, outputs, ins.size(), outs.size(), &params, sizeof(params), "softmax_bwd_f32", "softmax_bwd_f32_op", nullptr, nullptr);
    CHKSTATUS("synNodeCreate reshape failed!");
  }
};

template <typename T>
void SoftmaxGradKernel(const phi::Context& dev_ctx,
                       const phi::DenseTensor& out,
                       const phi::DenseTensor& out_grad,
                       int axis,
                       phi::DenseTensor* x_grad) {
  const int rank = x_grad->dims().size();
  const int calc_axis = phi::funcs::CanonicalAxis(axis, rank);
  int axis_dim = x_grad->dims()[calc_axis];

  // allocate memory on device.
  dev_ctx.template Alloc<T>(x_grad);
  if (x_grad->numel() == 0) {
    return;
  }

  SoftmaxGradOperator op;
  op.AddNode({out.dims(), out_grad.dims()}, {x_grad->dims()}, calc_axis);
  std::map<std::string, uint64_t> tensors;
  tensors["input_1"] = reinterpret_cast<uint64_t> (out.data<T>());
  tensors["input_2"] = reinterpret_cast<uint64_t> (out_grad.data<T>());
  tensors["output"] = reinterpret_cast<uint64_t> (x_grad->data<T>());
  op.CompileAndExecute(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);

  // const int n = phi::funcs::SizeToAxis(calc_axis, x_grad->dims());
  // const int d = phi::funcs::SizeFromAxis(calc_axis, x_grad->dims());
  // SoftmaxGrad(
  //     out.data<T>(), out_grad.data<T>(), axis_dim, n, d, x_grad->data<T>());
}

}  // namespace custom_kernel

PD_BUILD_PHI_KERNEL(softmax,
                    habana_hpu,
                    ALL_LAYOUT,
                    custom_kernel::SoftmaxKernel,
                    float,
                    double) {}

PD_BUILD_PHI_KERNEL(softmax_grad,
                    habana_hpu,
                    ALL_LAYOUT,
                    custom_kernel::SoftmaxGradKernel,
                    float,
                    double) {}
