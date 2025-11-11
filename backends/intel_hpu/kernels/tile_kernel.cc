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

#include "habanalabs/perf_lib_layer_params.h"
#include "kernels/funcs.h"
#include "kernels/hpu_operator.h"
#include "paddle/extension.h"
#include "utils/utils.h"

namespace custom_kernel {

class Tile : public HpuOperator {
 public:
  Tile() : HpuOperator("repeat_pt_fwd_", false) {}

  void AddNode(ConvertTensors& ct, ns_RepeatPt::Params& params) {
    auto inputs = ct.GetTensors();
    auto outputs = ct.GetTensors(false);

    auto x_tensor = createTensor(inputs[0].dims.size(),
                                 inputs[0].type,
                                 inputs[0].dims,
                                 true,
                                 inputs[0].name);

    auto out_tensor = createTensor(outputs[0].dims.size(),
                                   outputs[0].type,
                                   outputs[0].dims,
                                   true,
                                   outputs[0].name);

    synTensor ins[] = {x_tensor};
    synTensor outs[] = {out_tensor};
    std::string guid = guid_ + SynDataTypeToStr(inputs[0].type);
    synStatus status = synNodeCreate(graphHandle_,
                                     ins,
                                     outs,
                                     1,
                                     1,
                                     &params,
                                     sizeof(params),
                                     guid.c_str(),
                                     guid.c_str(),
                                     nullptr,
                                     nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate tile failed = ", status);
  }
};

template <typename T, typename Context>
void TileKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::IntArray& repeat_times,
                phi::DenseTensor* out) {
  VLOG(6) << "call HPU TileKernel";

  dev_ctx.template Alloc<T>(out);

  ns_RepeatPt::Params params;
  int max_repeat_times = sizeof(params.repeat) / sizeof(params.repeat[0]);
  int repeat_times_size = repeat_times.size();
  PADDLE_ENFORCE_GE(
      max_repeat_times,
      repeat_times_size,
      phi::errors::ResourceExhausted("unsupported repeat_times size."));

  for (int i = 0; i < repeat_times_size; i++) {
    params.repeat[i] = repeat_times[i];
  }
  params.size = repeat_times_size;

  ConvertTensors ct;
  ct.Add(x);
  ct.Add(out, false);
  std::vector<DIMS> inputs_dims = ct.GetDims();

  std::string guid_prefix = "TileKernel";
  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, ns_RepeatPt::Params>(
      guid_prefix, inputs_dims, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    Tile op;

    op.AddNode(ct, params);
    op.Compile();
    op_info.setOp(op);

    recipe = op_info.GetRecipe();
  }

  std::map<std::string, uint64_t> tensors = ct.GetDeviceAddr();
  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(tile,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::TileKernel,
                          bool,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
