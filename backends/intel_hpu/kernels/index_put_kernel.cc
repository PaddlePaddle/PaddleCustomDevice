// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "habanalabs/perf_lib_layer_params.h"
#include "kernels/funcs.h"
#include "kernels/hpu_operator.h"
#include "utils/utils.h"

namespace custom_kernel {

class IndexPut : public HpuOperator {
 public:
  IndexPut() : HpuOperator("scatter_nd_onnx_fwd_", false) {}

  void AddNode(ConvertTensors& ct, bool is_inplace) {
    auto inputs = ct.GetTensors();
    auto outputs = ct.GetTensors(false);

    std::vector<synTensor> syn_scatter_inputs;
    synSectionHandle section_shared = nullptr;
    if (is_inplace) section_shared = createSection();
    syn_scatter_inputs.push_back(createTensor(inputs[0].dims.size(),
                                              inputs[0].type,
                                              inputs[0].dims,
                                              true,
                                              inputs[0].name,
                                              section_shared));
    guid_ = guid_ + SynDataTypeToStr(inputs[0].type);

    DIMS tmp = inputs[1].dims;
    std::vector<synTensor> syn_concat_inputs;
    for (size_t i = 1; i < inputs.size() - 1; i++) {
      // unsqueeze on dim0 directly [x, 1]
      inputs[i].dims.emplace_back(1);
      syn_concat_inputs.push_back(createTensor(inputs[i].dims.size(),
                                               inputs[i].type,
                                               inputs[i].dims,
                                               true,
                                               inputs[i].name));
    }
    // concat indices
    std::string guid_concat = "concat";
    std::string name_concat = guid_ + "concat";
    synConcatenateParams concatParams;
    concatParams.axis = 0;
    std::vector<synTensor> syn_concat_outputs;
    DIMS indices_concat_dims = inputs[1].dims;
    indices_concat_dims.back() = syn_concat_inputs.size();
    syn_concat_outputs.push_back(createTensor(indices_concat_dims.size(),
                                              inputs[1].type,
                                              indices_concat_dims,
                                              false,
                                              "indices_concat"));
    synStatus status = synNodeCreate(graphHandle_,
                                     syn_concat_inputs.data(),
                                     syn_concat_outputs.data(),
                                     syn_concat_inputs.size(),
                                     syn_concat_outputs.size(),
                                     &concatParams,
                                     sizeof(concatParams),
                                     guid_concat.c_str(),
                                     name_concat.c_str(),
                                     nullptr,
                                     nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] IndexPutKernel synNodeCreate (concat) "
             "failed = ",
             status);

    // add indices concat tensor
    syn_scatter_inputs.push_back(syn_concat_outputs[0]);
    // add update tensor
    syn_scatter_inputs.push_back(createTensor(inputs.back().dims.size(),
                                              inputs.back().type,
                                              inputs.back().dims,
                                              true,
                                              inputs.back().name));

    std::vector<synTensor> syn_scatter_outputs;
    syn_scatter_outputs.push_back(createTensor(outputs[0].dims.size(),
                                               outputs[0].type,
                                               outputs[0].dims,
                                               true,
                                               outputs[0].name,
                                               section_shared));

    status = synNodeCreate(graphHandle_,
                           syn_scatter_inputs.data(),
                           syn_scatter_outputs.data(),
                           syn_scatter_inputs.size(),
                           syn_scatter_outputs.size(),
                           nullptr,
                           0,
                           guid_.c_str(),
                           "index_put",
                           nullptr,
                           nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);
  }
};

template <typename T, typename Context>
void IndexPutKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const std::vector<const phi::DenseTensor*>& indices,
                    const phi::DenseTensor& value,
                    bool accumulate,
                    phi::DenseTensor* out) {
  PD_CHECK(accumulate == false,
           "IndexPutKernel doesn't support accumulate=true");

  dev_ctx.template Alloc<T>(out);

  ConvertTensors ct;
  ct.Add(x);
  for (const auto& index : indices) {
    ct.Add(index);
  }
  ct.Add(value);
  ct.Add(out, false);

  bool is_inplace = (out->data() == x.data());

  OpCacheOperator op_info;
  std::vector<DIMS> inputs_dims = ct.GetDims();
  op_info.prepareOpInfo<T, nullptr_t>(
      is_inplace ? "IndexPutKernel" : "_IndexPutKernel", inputs_dims, nullptr);
  auto recipe = op_info.GetRecipe();
  if (recipe == nullptr) {
    IndexPut op;

    op.AddNode(ct, is_inplace);
    op.Compile();
    op_info.setOp(op);

    recipe = op_info.GetRecipe();
  }

  std::map<std::string, uint64_t> tensors = ct.GetDeviceAddr();
  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(index_put,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::IndexPutKernel,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          float,
                          int32_t,
                          int64_t,
                          bool) {}
