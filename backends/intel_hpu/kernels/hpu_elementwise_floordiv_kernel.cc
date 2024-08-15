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
#include "funcs.h"
#include "glog/logging.h"
#include "hpu_operator.h"
#include "perf_lib_layer_params.h"
#include "synapse_api.h"
#include "synapse_common_types.h"
#include "utils/utills.h"

namespace custom_kernel {

class BinaryOperator : public HpuOperator {
 public:
  BinaryOperator(std::string guid_prefix, std::string node_name)
      : HpuOperator(guid_prefix), pName_(node_name) {}

  void AddNode(const std::vector<DIMS>& ins,
   const std::vector<DIMS>& outs,
   synDataType datatype) {
    assert(ins.size() == 2 && "input size should be 2");
    assert(outs.size() == 1 && "output size should be 1");

    synTensor inputs[ins.size()] = {
        createTensor(ins[0].size(), datatype, ins[0], true, "x"),
        createTensor(ins[1].size(), datatype, ins[1], true, "y")};
    synTensor outputs[outs.size()] = {
        createTensor(outs[0].size(), datatype, outs[0], true, "output")};
    synStatus status = synNodeCreate(graphHandle_,
                                     inputs,
                                     outputs,
                                     ins.size(),
                                     outs.size(),
                                     nullptr,
                                     0,
                                     guid_.c_str(),
                                     pName_.c_str(),
                                     nullptr,
                                     nullptr);
    CHKSTATUS("synNodeCreate floor_divide_fwd failed!");
  }
  std::string pName_;
};

template <typename T, typename Context>
void FloorDivKernel(const Context& dev_ctx,
     const phi::DenseTensor& x,
     const phi::DenseTensor& y,
     int axis,
     phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  
  std::vector<int64_t> x_dim = phi::vectorize<int64_t>(x.dims());
  std::vector<int64_t> y_dim = phi::vectorize<int64_t>(y.dims());
  std::vector<int64_t> outputs_dim = phi::vectorize<int64_t>(out->dims());
  
  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, nullptr_t>("floor_divide_fwd", {x_dim, y_dim}, nullptr);
  auto recipe = op_info.GetRecipe();
  
  if (recipe == nullptr) {
    BinaryOperator op(op_info.guid_, "floordiv_op");
    op.AddNode({x_dim, y_dim}, {outputs_dim}, op_info.datatype_);
    op.Compile();
    op_info.setOp(op);    
    recipe = op_info.GetRecipe();   
  }   
  
  std::map<std::string, uint64_t> tensors;  
  tensors["x"] = reinterpret_cast<uint64_t>(x.data<T>());
  tensors["y"] = reinterpret_cast<uint64_t>(y.data<T>());
  tensors["output"] = reinterpret_cast<uint64_t>(out->data<T>());
  
  RecipeRunner runner(recipe); 
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
  
  return;
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(floor_divide,
  intel_hpu,
  ALL_LAYOUT,
  custom_kernel::FloorDivKernel,
  float) {}

