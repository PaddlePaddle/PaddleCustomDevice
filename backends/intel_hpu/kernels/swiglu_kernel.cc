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

#include "habanalabs/perf_lib_layer_params.h"
#include "habanalabs/synapse_api.h"
#include "habanalabs/synapse_common_types.h"
#include "kernels/funcs.h"
#include "kernels/hpu_funcs.h"
#include "kernels/hpu_operator.h"
#include "utils/utils.h"

namespace custom_kernel {

class SwiGlu : public HpuFusedOperator {
 public:
  SwiGlu() : HpuFusedOperator("swiglu_fwd_", false) {}

  template <typename T>
  void AddNode(ConvertTensors* ct) {
    const synDataType cast_type = syn_type_single;
    auto inputs = ct->GetTensors();
    auto outputs = ct->GetTensors(false);
    bool need_cast = (inputs[0].type == syn_type_fp16);
    bool need_split = (inputs.size() == 1);

    std::vector<DIMS> inputs_dims = ct->GetDims();
    std::vector<DIMS> outputs_dims = ct->GetDims(false);
    synTensor x = createTensorFromCT(ct, 0);

    synDataType inner_type = (need_cast ? cast_type : inputs[0].type);
    // cast if needed
    synTensor cast_x = nullptr;
    synTensor cast_y = nullptr;
    if (need_cast) {
      cast_x = createTensorNoPresist("cast_x", cast_type, inputs_dims[0]);
      std::string guid = "cast_" + SynDataTypeToStr(inputs[0].type) + "_to_" +
                         SynDataTypeToStr(cast_type);
      std::string node_name = guid_ + guid + "_x";
      std::vector<synTensor> cast_in1 = {x};
      std::vector<synTensor> cast_out1 = {cast_x};
      AddNodeCast(cast_in1, cast_out1, guid, node_name);

      if (!need_split) {
        synTensor y = createTensorFromCT(ct, 1);
        cast_y = createTensorNoPresist("cast_y", cast_type, inputs_dims[0]);
        guid = "cast_" + SynDataTypeToStr(inputs[1].type) + "_to_" +
               SynDataTypeToStr(cast_type);
        node_name = guid_ + guid + "_y";
        std::vector<synTensor> cast_in2 = {y};
        std::vector<synTensor> cast_out2 = {cast_y};
        AddNodeCast(cast_in2, cast_out2, guid, node_name);
      }
    } else {
      cast_x = x;
      if (!need_split) {
        cast_y = createTensorFromCT(ct, 1);
      }
    }

    // split if needed
    synTensor split_x = nullptr;
    synTensor split_y = nullptr;
    if (need_split) {
      // no y in input, need to split
      synSplitParams params;
      params.axis = 0;
      split_x = createTensorNoPresist("split_x", inner_type, outputs_dims[0]);
      split_y = createTensorNoPresist("split_y", inner_type, outputs_dims[0]);
      std::vector<synTensor> split_in = {cast_x};
      std::vector<synTensor> split_out = {split_x, split_y};
      std::string node_name = guid_ + "split";
      AddNodeSplit(split_in, split_out, params, node_name);
    } else {
      split_x = cast_x;
      split_y = cast_y;
    }

    // silu
    synTensor silu_x =
        createTensorNoPresist("silu_x", inner_type, outputs_dims[0]);
    std::vector<synTensor> silu_in = {split_x};
    std::vector<synTensor> silu_out = {silu_x};
    std::string silu_node_name = guid_ + "silu";
    if (need_cast) {
      AddNodeSilu<float>(silu_in, silu_out, silu_node_name);
    } else {
      AddNodeSilu<T>(silu_in, silu_out, silu_node_name);
    }

    // mul
    synTensor mult_xy = nullptr;
    if (need_cast) {
      mult_xy = createTensorNoPresist("mult_xy", inner_type, outputs_dims[0]);
    } else {
      mult_xy = createTensorFromCT(ct, 0, false);
    }
    std::vector<synTensor> mult_in = {silu_x, split_y};
    std::vector<synTensor> mult_out = {mult_xy};
    std::string mult_node_name = guid_ + "mult";
    if (need_cast) {
      AddNodeMultiply<float>(mult_in, mult_out, mult_node_name);
    } else {
      AddNodeMultiply<T>(mult_in, mult_out, mult_node_name);
    }

    // cast back
    if (need_cast) {
      synTensor cast_res = createTensorFromCT(ct, 0, false);

      std::string guid = "cast_" + SynDataTypeToStr(inner_type) + "_to_" +
                         SynDataTypeToStr(outputs[0].type);
      std::string node_name = guid_ + guid + "_res";
      std::vector<synTensor> cast_in = {mult_xy};
      std::vector<synTensor> cast_out = {cast_res};
      AddNodeCast(cast_in, cast_out, guid, node_name);
    }
  }
};

template <typename T, typename Context>
void SwiGluKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const paddle::optional<phi::DenseTensor>& y,
                  phi::DenseTensor* out) {
  // allocate memory on device.
  dev_ctx.template Alloc<T>(out);
  if (out->numel() == 0) {
    return;
  }

  custom_kernel::ConvertTensors ct;
  ct.Add(x);

  std::string kernel_name = "SwiGluKernelX";
  if (y) {
    const auto& y_tensor = y.get();
    kernel_name += "Y";
    ct.Add(y_tensor);
  }

  ct.Add(out, false);

  OpCacheOperator op_info;
  std::vector<int64_t> x_dims = phi::vectorize<int64_t>(x.dims());
  op_info.prepareOpInfo<T, nullptr_t>(kernel_name, {x_dims}, nullptr);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    SwiGlu op;

    op.AddNode<T>(&ct);
    op.Compile();
    op_info.setOp(op);

    recipe = op_info.GetRecipe();
  }

  std::map<std::string, uint64_t> tensors = ct.GetDeviceAddr();
  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(swiglu,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::SwiGluKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
