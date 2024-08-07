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
#include "perf_lib_layer_params.h"
#include "phi_funcs.h"
#include "hpu_operator.h"

namespace custom_kernel {

class AddOperator : public HpuOperator {
 public:
  AddOperator(std::string type, synDataType dtype)
      : HpuOperator("add_fwd"), guid_(type), datatype_(dtype) {}
  void AddNode(const std::vector<DIMS>& ins,
               const std::vector<DIMS>& outs) {
    assert(ins.size() == 2 && "input size should be 2");
    assert(outs.size() == 1 && "output size should be 1");

    synTensor inputs[ins.size()] = {
        createTensor(ins[0].size(), datatype_, ins[0], true, "x"),
        createTensor(ins[1].size(), datatype_, ins[1], true, "y")};
    synTensor outputs[outs.size()] = {
        createTensor(outs[0].size(), datatype_, outs[0], true, "output")};

    synStatus status = synNodeCreate(graphHandle_,
                                     inputs,
                                     outputs,
                                     ins.size(),
                                     outs.size(),
                                     nullptr,
                                     0,
                                     guid_.c_str(),
                                     "Add",
                                     nullptr,
                                     nullptr);
    CHKSTATUS("synNodeCreate add_fwd failed!");
  }
  std::string guid_;
  synDataType datatype_;
};


template <typename T>
void AddRawKernel(const phi::Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& y,
                  int axis,
                  phi::DenseTensor* out) {
    dev_ctx.template Alloc<T>(out);

    auto x_dims = x.dims();
    auto y_dims = y.dims();

    std::string guid = "add_fwd_f32";
    synDataType datatype = syn_type_single;
    if(std::is_same<T, phi::dtype::float16>::value)
    {
        guid = "add_fwd_f16";
        datatype = syn_type_fp16;
    }
    else if(std::is_same<T, phi::dtype::bfloat16>::value)
    {
        guid = "add_fwd_bf16";
        datatype = syn_type_bf16;
    }
    AddOperator op(guid, datatype);
    op.AddNode({x_dims, y_dims}, {out->dims()});
    op.Compile();

    std::map<std::string, uint64_t> tensors;
    tensors["x"] = reinterpret_cast<uint64_t>(x.data<T>());
    tensors["y"] = reinterpret_cast<uint64_t>(y.data<T>());
    tensors["output"] = reinterpret_cast<uint64_t>(out->data<T>());

    op.Execute(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);

    return;
}

template <typename T>
void AddKernel(const phi::Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::DenseTensor& y,
               phi::DenseTensor* out) {
  int axis = -1;
  custom_kernel::AddRawKernel<T>(dev_ctx, x, y, axis, out);
}

}  // namespace custom_kernel

PD_BUILD_PHI_KERNEL(add_raw,
                    intel_hpu,
                    ALL_LAYOUT,
                    custom_kernel::AddRawKernel,
                    float,
                    phi::dtype::float16,
                    phi::dtype::bfloat16) {}

PD_BUILD_PHI_KERNEL(add,
                    intel_hpu,
                    ALL_LAYOUT,
                    custom_kernel::AddKernel,
                    float,
                    phi::dtype::float16,
                    phi::dtype::bfloat16) {}