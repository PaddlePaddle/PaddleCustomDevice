// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#include "kernels/hpu_operator.h"
#include "paddle/extension.h"
#include "utils/utils.h"

#define SDPA_SET_INPUT_AND_FLAGS(ptr, flag_name)  \
  if (ptr) {                                      \
    flags |= SdpaFlags_t::SDPA_FLAGS_##flag_name; \
    ct.Add(ptr);                                  \
  }

namespace custom_kernel {

struct SDPAParams {
  bool has_atten_mask;
  ns_Sdpa::ParamsV3 params;
};

class FusedFp8Sdpa : public HpuOperator {
 public:
  FusedFp8Sdpa() : HpuOperator("sdpa_recomp_fwd_hf8") {}
  void AddNode(ConvertTensors& ct, SDPAParams& params) {
    auto inputs = ct.GetTensors();
    auto outputs = ct.GetTensors(false);

    std::vector<synTensor> sync_inputs;
    synStatus status = synFail;
    for (size_t i = 0; i < 3; i++) {
      sync_inputs.push_back(createTensor(inputs[i].dims.size(),
                                         inputs[i].type,
                                         inputs[i].dims,
                                         true,
                                         inputs[i].name));
    }

    // atten mask
    if (!params.has_atten_mask) {
      sync_inputs.push_back(nullptr);
    }

    // seed
    sync_inputs.push_back(nullptr);

    for (size_t i = 3; i < inputs.size(); i++) {
      sync_inputs.push_back(createTensor(inputs[i].dims.size(),
                                         inputs[i].type,
                                         inputs[i].dims,
                                         true,
                                         inputs[i].name));
    }

    std::vector<synTensor> sync_outputs;
    for (size_t i = 0; i < outputs.size(); i++) {
      sync_outputs.push_back(createTensor(outputs[i].dims.size(),
                                          outputs[i].type,
                                          outputs[i].dims,
                                          true,
                                          outputs[i].name));
    }

    status = synNodeCreate(graphHandle_,
                           sync_inputs.data(),
                           sync_outputs.data(),
                           sync_inputs.size(),
                           sync_outputs.size(),
                           &params.params,
                           sizeof(params.params),
                           guid_.c_str(),
                           guid_.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);
  }
};

template <typename T, typename Context>
void fused_fp8_sdpa(const Context& dev_ctx,
                    const phi::DenseTensor& q,
                    const phi::DenseTensor& k,
                    const phi::DenseTensor& v,
                    const paddle::optional<phi::DenseTensor>& attn_mask,
                    const paddle::optional<phi::DenseTensor>& d_scale_q,
                    const paddle::optional<phi::DenseTensor>& d_scale_k,
                    const paddle::optional<phi::DenseTensor>& d_scale_v,
                    const paddle::optional<phi::DenseTensor>& q_scale_s,
                    const paddle::optional<phi::DenseTensor>& q_scale_o,
                    const paddle::optional<phi::DenseTensor>& d_scale_s,
                    float scale,
                    bool causal,
                    phi::DenseTensor* out) {
  // allocate memory on device.
  dev_ctx.template Alloc<T>(out);
  if (out->numel() == 0) {
    return;
  }

  ConvertTensors ct;
  ct.Add(q);
  ct.Add(k);
  ct.Add(v);

  unsigned int flags = 0;

  SDPA_SET_INPUT_AND_FLAGS(d_scale_q.get_ptr(), D_SCALE_Q)
  SDPA_SET_INPUT_AND_FLAGS(d_scale_k.get_ptr(), D_SCALE_K)
  SDPA_SET_INPUT_AND_FLAGS(d_scale_v.get_ptr(), D_SCALE_V)
  SDPA_SET_INPUT_AND_FLAGS(q_scale_s.get_ptr(), Q_SCALE_S)
  SDPA_SET_INPUT_AND_FLAGS(q_scale_o.get_ptr(), Q_SCALE_O)
  SDPA_SET_INPUT_AND_FLAGS(d_scale_s.get_ptr(), D_SCALE_S)

  SDPAParams params{};

  if (attn_mask.get_ptr()) {
    ct.Add(attn_mask.get_ptr());
    params.has_atten_mask = true;
  }

  params.params.scale = scale;
  params.params.is_causal = causal;
  params.params.dropout.ratio = 0;
  params.params.is_inference = true;
  params.params.softmax_mode = SDPA_DEFAULT_SOFTMAX;
  params.params.flags = flags;

  ct.Add(*out, false);
  std::vector<DIMS> inputs_dims = ct.GetDims();

  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, SDPAParams>(
      "FusedFp8SdpaKernel", inputs_dims, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    FusedFp8Sdpa op;
    op.AddNode(ct, params);
    op.Compile();
    op_info.setOp(op);
    recipe = op_info.GetRecipe();
  }

  auto tensors = ct.GetDeviceAddr();
  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

}  // namespace custom_kernel

std::vector<paddle::Tensor> FusedFp8SdpaForward(
    const paddle::Tensor& q,
    const paddle::Tensor& k,
    const paddle::Tensor& v,
    const paddle::optional<paddle::Tensor>& attn_mask,
    const paddle::optional<paddle::Tensor>& d_scale_q,
    const paddle::optional<paddle::Tensor>& d_scale_k,
    const paddle::optional<paddle::Tensor>& d_scale_v,
    const paddle::optional<paddle::Tensor>& q_scale_s,
    const paddle::optional<paddle::Tensor>& q_scale_o,
    const paddle::optional<paddle::Tensor>& d_scale_s,
    bool causal,
    float scale) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(q.place()));

  auto q_tensor = static_cast<const phi::DenseTensor*>(q.impl().get());
  auto k_tensor = static_cast<const phi::DenseTensor*>(k.impl().get());
  auto v_tensor = static_cast<const phi::DenseTensor*>(v.impl().get());

  // attn_mask
  phi::DenseTensor* attn_mask_tensor = nullptr;
  if (attn_mask) {
    auto attn_mask_ptr = *(attn_mask.get_ptr());
    attn_mask_tensor =
        static_cast<phi::DenseTensor*>(attn_mask_ptr.impl().get());
  }

  // s_scale_q
  phi::DenseTensor* d_scale_q_tensor = nullptr;
  if (d_scale_q) {
    auto d_scale_q_ptr = *(d_scale_q.get_ptr());
    d_scale_q_tensor =
        static_cast<phi::DenseTensor*>(d_scale_q_ptr.impl().get());
  }

  // d_scale_k
  phi::DenseTensor* d_scale_k_tensor = nullptr;
  if (d_scale_k) {
    auto d_scale_k_ptr = *(d_scale_k.get_ptr());
    d_scale_k_tensor =
        static_cast<phi::DenseTensor*>(d_scale_k_ptr.impl().get());
  }

  // d_scale_v
  phi::DenseTensor* d_scale_v_tensor = nullptr;
  if (d_scale_v) {
    auto d_scale_v_ptr = *(d_scale_v.get_ptr());
    d_scale_v_tensor =
        static_cast<phi::DenseTensor*>(d_scale_v_ptr.impl().get());
  }

  // q_scale_s
  phi::DenseTensor* q_scale_s_tensor = nullptr;
  if (q_scale_s) {
    auto q_scale_s_ptr = *(q_scale_s.get_ptr());
    q_scale_s_tensor =
        static_cast<phi::DenseTensor*>(q_scale_s_ptr.impl().get());
  }

  // q_scale_o
  phi::DenseTensor* q_scale_o_tensor = nullptr;
  if (q_scale_o) {
    auto q_scale_o_ptr = *(q_scale_o.get_ptr());
    q_scale_o_tensor =
        static_cast<phi::DenseTensor*>(q_scale_o_ptr.impl().get());
  }

  // d_scale_s
  phi::DenseTensor* d_scale_s_tensor = nullptr;
  if (d_scale_s) {
    auto d_scale_s_ptr = *(d_scale_s.get_ptr());
    d_scale_s_tensor =
        static_cast<phi::DenseTensor*>(d_scale_s_ptr.impl().get());
  }

  auto out_tensor = std::make_shared<phi::DenseTensor>();
  out_tensor->Resize(q_tensor->dims());

  custom_kernel::fused_fp8_sdpa<phi::dtype::bfloat16>(
      *dev_ctx,
      *q_tensor,
      *k_tensor,
      *v_tensor,
      attn_mask ? *attn_mask_tensor : paddle::optional<phi::DenseTensor>(),
      d_scale_q ? *d_scale_q_tensor : paddle::optional<phi::DenseTensor>(),
      d_scale_k ? *d_scale_k_tensor : paddle::optional<phi::DenseTensor>(),
      d_scale_v ? *d_scale_v_tensor : paddle::optional<phi::DenseTensor>(),
      q_scale_s ? *q_scale_s_tensor : paddle::optional<phi::DenseTensor>(),
      q_scale_o ? *q_scale_o_tensor : paddle::optional<phi::DenseTensor>(),
      d_scale_s ? *d_scale_s_tensor : paddle::optional<phi::DenseTensor>(),
      scale,
      causal,
      out_tensor.get());

  paddle::Tensor out(out_tensor);

  return {out};
}

std::vector<std::vector<int64_t>> FusedFp8SdpaForwardShape(
    const std::vector<int64_t>& query_states_shape,
    const std::vector<int64_t>& key_states_shape,
    const std::vector<int64_t>& value_states_shape) {
  int64_t bsz = query_states_shape[0];
  int64_t num_heads = query_states_shape[1];
  int64_t seq_len = query_states_shape[2];
  int head_dim = query_states_shape[3];
  return {{bsz, num_heads, seq_len, head_dim}};
}

std::vector<paddle::DataType> FusedFp8SdpaForwardDtype(
    const paddle::DataType& query_states_dtype,
    const paddle::DataType& key_states_dtype,
    const paddle::DataType& value_states_dtype) {
  return {paddle::DataType::BFLOAT16};
}

PD_BUILD_OP(fused_fp8_sdpa)
    .Inputs({
        "q",
        "k",
        "v",
        paddle::Optional("attn_mask"),
        paddle::Optional("d_scale_q"),
        paddle::Optional("d_scale_k"),
        paddle::Optional("d_scale_v"),
        paddle::Optional("q_scale_s"),
        paddle::Optional("q_scale_o"),
        paddle::Optional("d_scale_s"),
    })
    .Attrs({"causal: bool", "scaling_factor: float"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(FusedFp8SdpaForward))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedFp8SdpaForwardShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedFp8SdpaForwardDtype));
