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
#include "hpu_operator.h"
#include "utils/utills.h"

namespace custom_kernel {

class Concat : public HpuOperator {
 public:
  Concat(synDataType dtype) : HpuOperator("concat"), dtype_(dtype) {}

  void AddNode(const std::vector<DIMS>& ins,
               const std::vector<DIMS>& outs,
               unsigned params) {
    std::vector<synTensor> inputs;
    for (size_t i = 0; i < ins.size(); i++) {
      std::string name("x");
      name = name + std::to_string(i);
      inputs.push_back(
          createTensor(ins[0].size(), dtype_, ins[i], true, name.c_str()));
    }

    synTensor outputs[outs.size()] = {
        createTensor(outs[0].size(), dtype_, outs[0], true, "output")};

    synStatus status = synNodeCreate(graphHandle_,
                                     inputs.data(),
                                     outputs,
                                     ins.size(),
                                     outs.size(),
                                     &params,
                                     sizeof(params),
                                     guid_.c_str(),
                                     "CONCAT",
                                     nullptr,
                                     nullptr);
    LOG_IF(ERROR, status != synSuccess)
        << "[RUNTIME] synNodeCreate() failed = " << status;
  }

 protected:
  synDataType dtype_;
};

template <typename T, typename Context>
void ConcatKernel(const Context& dev_ctx,
                  const std::vector<const phi::DenseTensor*>& ins,
                  const phi::Scalar& axis_scalar,
                  phi::DenseTensor* out) {
  int axis = axis_scalar.to<int>();
  axis = CanonicalAxis(static_cast<int64_t>(axis),
                       static_cast<int64_t>(ins[0]->dims().size()));
  axis = static_cast<int64_t>(ins[0]->dims().size()) - 1 - axis;

  dev_ctx.template Alloc<T>(out);
  if (out->numel() == 0) {
    return;
  }

  std::vector<int64_t> outputs_dim = phi::vectorize<int64_t>(out->dims());
  std::vector<DIMS> inputs_dims = {};
  for (size_t i = 0; i < ins.size(); i++) {
    inputs_dims.push_back(phi::vectorize<int64_t>(ins[i]->dims()));
  }

  unsigned params = static_cast<unsigned>(axis);
  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, unsigned>("ConcatKernel", inputs_dims, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    Concat op(op_info.datatype_);
    op.AddNode(inputs_dims, {outputs_dim}, params);
    op.Compile();
    op_info.setOp(op);
    recipe = op_info.GetRecipe();
  }

  std::map<std::string, uint64_t> tensors;
  for (size_t i = 0; i < ins.size(); i++) {
    std::string name("x");
    name = name + std::to_string(i);
    tensors[name] = reinterpret_cast<uint64_t>(ins[i]->data<T>());
  }
  tensors["output"] = reinterpret_cast<uint64_t>(out->data<T>());
  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(concat,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::ConcatKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
