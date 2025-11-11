// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#include "habanalabs/perf_lib_layer_params.h"
#include "kernels/funcs.h"
#include "kernels/hpu_operator.h"
#include "utils/utils.h"

namespace custom_kernel {

class Amax : public HpuOperator {
 public:
  Amax() : HpuOperator("reduce_max_multi_dim_fwd_f32") {}

  void AddNode(ConvertTensors& ct, ns_Reduction::ParamsV2& params) {
    auto inputs = ct.GetTensors();
    auto outputs = ct.GetTensors(false);

    std::string guid_amax = "reduce_max_multi_dim_fwd_f32";
    std::vector<synTensor> syn_inputs;
    synStatus status = synFail;

    syn_inputs.push_back(createTensor(inputs[0].dims.size(),
                                      inputs[0].type,
                                      inputs[0].dims,
                                      true,
                                      inputs[0].name));

    std::vector<synTensor> syn_outputs;
    syn_outputs.push_back(createTensor(outputs[0].dims.size(),
                                       outputs[0].type,
                                       outputs[0].dims,
                                       true,
                                       outputs[0].name));

    status = synNodeCreate(graphHandle_,
                           syn_inputs.data(),
                           syn_outputs.data(),
                           syn_inputs.size(),
                           syn_outputs.size(),
                           &params,
                           sizeof(params),
                           guid_amax.c_str(),
                           "reduce_max_multi_dim_fwd_bf16",
                           nullptr,
                           nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);
  }
};

template <typename T, typename Context>
void AmaxKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::IntArray& dims,
                bool keep_dim,
                phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  if (x.numel() == 0) {
    return;
  }
  ConvertTensors ct;
  ct.Add(x);
  ct.Add(out, false);
  auto rank = static_cast<int32_t>(x.dims().size());
  std::vector<DIMS> inputs_dims = ct.GetDims();
  ns_Reduction::ParamsV2 params = {};
  OpCacheOperator op_info;
  params.keepDim = keep_dim;
  if (dims.size() != 0) {
    for (size_t i = 0; i < dims.size(); i++) {
      auto dim = CanonicalAxis(static_cast<int64_t>(dims[i]),
                               static_cast<int64_t>(rank));
      params.reductionDimensionMask |= 1 << (rank - dim - 1);
    }
  } else {
    params.reductionDimensionMask = (1 << rank) - 1;
  }

  op_info.prepareOpInfo<T, ns_Reduction::ParamsV2>(
      "AmaxKernel", inputs_dims, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    Amax op;
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

PD_REGISTER_PLUGIN_KERNEL(
    amax, intel_hpu, ALL_LAYOUT, custom_kernel::AmaxKernel, float) {}
