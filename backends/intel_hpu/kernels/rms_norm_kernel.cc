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
#include "habanalabs/synapse_api.h"
#include "habanalabs/synapse_common_types.h"
#include "kernels/funcs.h"
#include "kernels/hpu_operator.h"
#include "utils/utils.h"

namespace custom_kernel {

class RMS : public HpuOperator {
 public:
  explicit RMS(std::string guid_prefix, synDataType dtype)
      : HpuOperator(guid_prefix), dtype_(dtype) {}
  void AddNode(const std::vector<DIMS>& ins,
               const std::vector<DIMS>& outs,
               ns_LayerNormKernel::Params params) {
    std::vector<synTensor> inputs;
    std::vector<synTensor> outputs;

    if (ins.size() > 2) {
      std::vector<synTensor> add_inputs;
      std::vector<synTensor> add_outputs;

      auto x = createTensor(ins[0].size(), dtype_, ins[0], true, "x");
      add_inputs.push_back(x);

      auto residual =
          createTensor(ins[0].size(), dtype_, ins[0], true, "residual");
      add_inputs.push_back(residual);

      auto residual_out =
          createTensor(ins[0].size(), dtype_, ins[0], true, "residual_out");
      add_outputs.push_back(residual_out);

      std::string add_guid_ = "add_fwd_" + SynDataTypeToStr(dtype_);
      synStatus status = synNodeCreate(graphHandle_,
                                       add_inputs.data(),
                                       add_outputs.data(),
                                       add_inputs.size(),
                                       add_outputs.size(),
                                       nullptr,
                                       0,
                                       add_guid_.c_str(),
                                       "Add",
                                       nullptr,
                                       nullptr);
      PD_CHECK(status == synSuccess,
               "[RUNTIME] synNodeCreate () failed = %d",
               status);

      inputs.push_back(residual_out);
    } else {
      auto x = createTensor(ins[0].size(), dtype_, ins[0], true, "x");
      inputs.push_back(x);
    }

    // syn_type_single
    auto w = createTensor(ins[1].size(), dtype_, ins[1], true, "w");
    inputs.push_back(w);

    auto out = createTensor(outs[0].size(), dtype_, outs[0], true, "out");
    outputs.push_back(out);
    auto mean_square =
        createTensor(outs[1].size(), dtype_, outs[1], false, "inv_var");
    outputs.push_back(mean_square);

    synStatus status = synNodeCreate(graphHandle_,
                                     inputs.data(),
                                     outputs.data(),
                                     inputs.size(),
                                     outputs.size(),
                                     &params,
                                     sizeof(params),
                                     guid_.c_str(),
                                     "RMS",
                                     nullptr,
                                     nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);
  }

 protected:
  synDataType dtype_;
};

template <typename T, typename Context>
void RmsNormKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const paddle::optional<phi::DenseTensor>& bias,
                   const paddle::optional<phi::DenseTensor>& residual,
                   const phi::DenseTensor& norm_weight,
                   const paddle::optional<phi::DenseTensor>& norm_bias,
                   const float epsilon,
                   const int begin_norm_axis,
                   const float quant_scale,
                   const int quant_round_type,
                   const float quant_max_bound,
                   const float quant_min_bound,
                   phi::DenseTensor* out,
                   phi::DenseTensor* residual_out,
                   phi::DenseTensor* inv_var) {
  dev_ctx.template Alloc<T>(out);
  dev_ctx.template Alloc<T>(inv_var);

  std::vector<int64_t> x_dims = phi::vectorize<int64_t>(x.dims());
  std::vector<int64_t> w_dims = phi::vectorize<int64_t>(norm_weight.dims());
  std::vector<int64_t> outputs_dim = phi::vectorize<int64_t>(out->dims());
  std::vector<int64_t> inv_var_dim = phi::vectorize<int64_t>(inv_var->dims());

  std::vector<int64_t> residual_dims;
  if (residual) {
    dev_ctx.template Alloc<T>(residual_out);
    residual_dims = phi::vectorize<int64_t>(residual->dims());
  }

  std::vector<int64_t> x_reshape_dims;
  for (int i = 0; i < begin_norm_axis; ++i) {
    x_reshape_dims.push_back(x_dims[i]);
  }

  int32_t cols = 1;
  for (int i = begin_norm_axis; i < x.dims().size(); ++i) {
    cols *= x_dims[i];
  }
  x_reshape_dims.push_back(cols);

  if (x_reshape_dims.size() < 3) {
    for (int i = x_reshape_dims.size(); i < 3; i++)
      x_reshape_dims.insert(x_reshape_dims.begin(), 1);
  }

  ns_LayerNormKernel::Params params;
  memset(reinterpret_cast<void*>(&params),
         0x00,
         sizeof(ns_LayerNormKernel::Params));
  params.epsValid = true;
  params.eps = epsilon;

  OpCacheOperator op_info;
  if (residual) {
    op_info.prepareOpInfo<T, ns_LayerNormKernel::Params>(
        "rms_norm_ex_fwd", {x_reshape_dims, w_dims, residual_dims}, &params);
  } else {
    op_info.prepareOpInfo<T, ns_LayerNormKernel::Params>(
        "rms_norm_ex_fwd", {x_reshape_dims, w_dims}, &params);
  }
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    RMS op(op_info.guid_, op_info.datatype_);
    if (residual) {
      op.AddNode({x_reshape_dims, w_dims, residual_dims},
                 {outputs_dim, inv_var_dim},
                 params);
    } else {
      op.AddNode({x_reshape_dims, w_dims}, {outputs_dim, inv_var_dim}, params);
    }
    op.Compile();
    op_info.setOp(op);

    recipe = op_info.GetRecipe();
  }

  std::map<std::string, uint64_t> tensors;
  tensors["x"] = reinterpret_cast<uint64_t>(x.data<T>());
  tensors["w"] = reinterpret_cast<uint64_t>(norm_weight.data<T>());
  if (residual) {
    tensors["residual"] = reinterpret_cast<uint64_t>(residual->data<T>());
    tensors["residual_out"] =
        reinterpret_cast<uint64_t>(residual_out->data<T>());
  }
  tensors["out"] = reinterpret_cast<uint64_t>(out->data<T>());
  tensors["inv_var"] = reinterpret_cast<uint64_t>(inv_var->data<T>());

  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

template <typename T, typename Context>
void FusedRmsNormQuantKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& x,
    const paddle::optional<phi::DenseTensor>& bias,
    const paddle::optional<phi::DenseTensor>& residual,
    const phi::DenseTensor& norm_weight,
    const paddle::optional<phi::DenseTensor>& norm_bias,
    const float epsilon,
    const int begin_norm_axis,
    const float quant_scale,
    const int quant_round_type,
    const float quant_max_bound,
    const float quant_min_bound,
    phi::DenseTensor* out,
    phi::DenseTensor* residual_out,
    phi::DenseTensor* inv_var) {
  custom_kernel::RmsNormKernel<T, Context>(dev_ctx,
                                           x,
                                           bias,
                                           residual,
                                           norm_weight,
                                           norm_bias,
                                           epsilon,
                                           begin_norm_axis,
                                           quant_scale,
                                           quant_round_type,
                                           quant_max_bound,
                                           quant_min_bound,
                                           out,
                                           residual_out,
                                           inv_var);
}
}  // namespace custom_kernel

// Add the original kernel name rms_norm for compatibility with Paddle 3.2.2
PD_REGISTER_PLUGIN_KERNEL(rms_norm,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::RmsNormKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(fused_rms_norm_quant,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::FusedRmsNormQuantKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
