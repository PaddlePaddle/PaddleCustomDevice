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

#include "hpu_operator.h"
#include "paddle/phi/capi/all.h"
#include "perf_lib_layer_params.h"
#include "phi_funcs.h"

namespace custom_kernel {

class SoftmaxOperator : public HpuOperator {
 public:
  SoftmaxOperator() : HpuOperator("softmax_fwd_f32") {}
  void AddNode(const std::vector<DIMS>& ins,
               const std::vector<DIMS>& outs,
               int axis_dim) {
    assert(ins.size() == 1 && "input size should be 1");
    assert(outs.size() == 1 && "output size should be 1");

    synTensor inputs[ins.size()] = {
        createTensor(ins[0].size(), syn_type_float, ins[0], true, "input")};
    synTensor outputs[outs.size()] = {
        createTensor(outs[0].size(), syn_type_float, outs[0], true, "output")};
    ns_Softmax::Params params{ins[0].size() - 1 - axis_dim};
    synStatus status = synNodeCreate(graphHandle_,
                                     inputs,
                                     outputs,
                                     ins.size(),
                                     outs.size(),
                                     &params,
                                     sizeof(params),
                                     "softmax_fwd_f32",
                                     "softmax_op",
                                     nullptr,
                                     nullptr);
    CHKSTATUS("synNodeCreate reshape failed!");
  }
};

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

  // compile
  SoftmaxOperator op;
  op.AddNode({x.dims()}, {out->dims()}, calc_axis);
  op.Compile();

  // runtime
  std::map<std::string, uint64_t> tensors;
  tensors["input"] = reinterpret_cast<uint64_t>(x.data<T>());
  tensors["output"] = reinterpret_cast<uint64_t>(out->data<T>());
  op.Execute(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

}  // namespace custom_kernel

PD_BUILD_PHI_KERNEL(softmax,
                    intel_hpu,
                    ALL_LAYOUT,
                    custom_kernel::SoftmaxKernel,
                    float,
                    phi::dtype::float16,
                    phi::dtype::bfloat16) {}
