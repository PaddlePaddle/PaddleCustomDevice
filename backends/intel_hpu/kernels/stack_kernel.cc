// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "kernels/funcs.h"
#include "kernels/hpu_funcs.h"
#include "kernels/hpu_operator.h"
#include "utils/utils.h"

namespace custom_kernel {

class Stack : public HpuFusedOperator {
 public:
  explicit Stack(synDataType dtype)
      : HpuFusedOperator("stack"), dtype_(dtype) {}

  void AddNode(ConvertTensors& ct, unsigned params) {
    auto inputs = ct.GetTensors();
    auto outputs = ct.GetTensors(false);

    std::vector<synTensor> syn_inputs;
    for (size_t i = 0; i < inputs.size(); i++) {
      syn_inputs.push_back(createTensorFromCT(&ct, i));
    }

    auto concat_dims = outputs[0].dims;

    // Merge concat_dims[params] and concat_dims[params+1]
    auto reduce_dim = concat_dims.size() - 2 - params;
    concat_dims[reduce_dim] *= concat_dims[reduce_dim + 1];
    concat_dims.erase(concat_dims.begin() + reduce_dim + 1);

    std::vector<synTensor> outputs_concat;
    auto concated = createTensorNoPresist("concat", dtype_, concat_dims);
    outputs_concat.push_back(concated);

    synConcatenateParams concatParams;
    concatParams.axis = params;
    AddNodeConcat(syn_inputs, outputs_concat, concatParams, guid_ + "concat");

    std::vector<synTensor> syn_outputs;
    auto stacked = createTensorFromCT(&ct, 0, false);
    syn_outputs.push_back(stacked);

    AddNodeReshape(outputs_concat, syn_outputs, guid_ + "reshape");
  }

 protected:
  synDataType dtype_;
};

template <typename T, typename Context>
void StackKernel(const Context& dev_ctx,
                 const std::vector<const phi::DenseTensor*>& x,
                 int axis,
                 phi::DenseTensor* y) {
  dev_ctx.template Alloc<T>(y);

  ConvertTensors ct;
  for (size_t i = 0; i < x.size(); i++) {
    ct.Add(x[i]);
  }
  ct.Add(y, false);

  axis = CanonicalAxis(static_cast<int64_t>(axis),
                       static_cast<int64_t>(x[0]->dims().size()));
  axis = static_cast<int64_t>(x[0]->dims().size()) - 1 - axis;
  unsigned params = static_cast<unsigned>(axis);

  std::vector<DIMS> inputs_dims = ct.GetDims();
  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, unsigned>("StackKernel", inputs_dims, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    Stack op(op_info.datatype_);
    op.AddNode(ct, params);
    op.Compile();
    op_info.setOp(op);
    recipe = op_info.GetRecipe();
  }

  RecipeRunner runner(recipe);
  auto tensors = ct.GetDeviceAddr();
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(stack,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::StackKernel,
                          float,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::float8_e4m3fn) {}
