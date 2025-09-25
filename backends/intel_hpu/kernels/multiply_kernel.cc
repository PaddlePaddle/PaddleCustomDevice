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
#include "kernels/hpu_funcs.h"
#include "kernels/hpu_operator.h"
#include "paddle/extension.h"
#include "utils/utils.h"

namespace custom_kernel {

class Multiply : public HpuFusedOperator {
 public:
  Multiply() : HpuFusedOperator("MultiplyKernel", false) {}

  template <typename T>
  void AddNode(ConvertTensors* ct) {
    auto inputs = ct->GetTensors();
    auto outputs = ct->GetTensors(false);

    std::vector<synTensor> ins;
    std::vector<synTensor> outs;

    auto x_tensor = createTensorFromCT(ct, 0);
    auto y_tensor = createTensorFromCT(ct, 1);
    auto out_tensor = createTensorFromCT(ct, 0, false);

    if (inputs[0].type == syn_type_int64) {
      std::string guid;
      std::string name;
      std::vector<DIMS> inputs_dims = ct->GetDims();
      std::vector<DIMS> outputs_dims = ct->GetDims(false);

      auto x_i32 =
          createTensorNoPresist("x_i32", syn_type_int32, inputs_dims[0]);
      auto y_i32 =
          createTensorNoPresist("y_i32", syn_type_int32, inputs_dims[1]);
      auto out_i32 =
          createTensorNoPresist("out_i32", syn_type_int32, outputs_dims[0]);

      ins.push_back(x_tensor);
      outs.push_back(x_i32);
      guid = "cast_i64_to_i32";
      name = "cast_x_i64_to_i32";
      AddNodeCast(ins, outs, guid, name);

      ins.clear();
      ins.push_back(y_tensor);
      outs.clear();
      outs.push_back(y_i32);
      name = "cast_y_i64_to_i32";
      AddNodeCast(ins, outs, guid, name);

      ins.clear();
      ins.push_back(x_i32);
      ins.push_back(y_i32);
      outs.clear();
      outs.push_back(out_i32);
      AddNodeMultiply<int32_t>(ins, outs, guid_);

      ins.clear();
      ins.push_back(out_i32);
      outs.clear();
      outs.push_back(out_tensor);
      guid = "cast_i32_to_i64";
      name = "cast_out_i32_to_i64";
      AddNodeCast(ins, outs, guid, name);
    } else {
      ins.push_back(x_tensor);
      ins.push_back(y_tensor);
      outs.push_back(out_tensor);
      AddNodeMultiply<T>(ins, outs, guid_);
    }
  }
};

template <typename T, typename Context>
void MultiplyRawKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& y,
                       int axis,
                       phi::DenseTensor* out) {
  VLOG(6) << "call HPU MultiplyRawKernel";

  dev_ctx.template Alloc<T>(out);

  ConvertTensors ct;
  ct.Add(x);
  ct.Add(y);
  ct.Add(out, false);
  std::vector<DIMS> inputs_dims = ct.GetDims();

  std::string guid_prefix = "MultiplyKernel";
  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, ns_RepeatPt::Params>(
      guid_prefix, inputs_dims, nullptr);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    Multiply op;

    op.AddNode<T>(&ct);
    op.Compile();
    op_info.setOp(op);

    recipe = op_info.GetRecipe();
  }

  std::map<std::string, uint64_t> tensors = ct.GetDeviceAddr();
  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

template <typename T, typename Context>
void MultiplyKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    phi::DenseTensor* out) {
  MultiplyRawKernel<T, Context>(dev_ctx, x, y, -1, out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(multiply_raw,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::MultiplyRawKernel,
                          int64_t,
                          int,
                          int16_t,
                          int8_t,
                          uint8_t,
                          bool,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::float8_e4m3fn,
                          float) {}

PD_REGISTER_PLUGIN_KERNEL(multiply,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::MultiplyKernel,
                          int64_t,
                          int,
                          int16_t,
                          int8_t,
                          uint8_t,
                          bool,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::float8_e4m3fn,
                          float) {}
