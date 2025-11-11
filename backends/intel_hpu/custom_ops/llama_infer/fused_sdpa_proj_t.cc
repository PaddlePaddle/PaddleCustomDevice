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

#define SDPA_SET_INPUT(ptr) \
  if (ptr) {                \
    ct.Add(ptr);            \
  }

#define SDPA_SET_INPUT_AND_FLAGS(ptr, flag_name)  \
  if (ptr) {                                      \
    flags |= SdpaFlags_t::SDPA_FLAGS_##flag_name; \
    ct.Add(ptr);                                  \
  }

namespace custom_kernel {

struct FusedSdpaProjParams {
  ns_Sdpa::ParamsV3 sdpa_params;
  bool is_GQA = false;
  bool fp8_sdpa = false;
  bool fp8_gemm = false;
};

class FusedSdpaProjBTMH : public HpuFusedOperator {
 public:
  explicit FusedSdpaProjBTMH(std::string guid_prefix, synDataType dtype)
      : HpuFusedOperator(guid_prefix, false), dtype_(dtype) {}
  template <typename T>
  void AddNode(ConvertTensors& ct, FusedSdpaProjParams params) {
    auto inputs = ct.GetTensors();
    auto outputs = ct.GetTensors(false);

    std::vector<synTensor> kv_inputs;
    kv_inputs.push_back(createTensorFromCT(&ct, 1));
    auto k_v_dims = inputs[1].dims;
    k_v_dims[0] = 1;

    synSliceParamsV2 sliceParams;
    for (uint64_t i = 0; i < k_v_dims.size(); i++) {
      sliceParams.axes[i] = i;
      sliceParams.steps[i] = 1;
      sliceParams.starts[i] = 0;
      sliceParams.ends[i] = k_v_dims[k_v_dims.size() - 1 - i];
    }

    std::vector<synTensor> k_slice;
    auto k_split = createTensorNoPresist("k_split", inputs[1].type, k_v_dims);
    k_slice.push_back(k_split);

    AddNodeSlice(kv_inputs, k_slice, sliceParams, guid_ + "slice_key");

    std::vector<synTensor> v_slice;
    auto v_split = createTensorNoPresist("v_split", inputs[1].type, k_v_dims);
    v_slice.push_back(v_split);
    sliceParams.starts[k_v_dims.size() - 1] = 1;
    sliceParams.ends[k_v_dims.size() - 1] = 2;

    AddNodeSlice(
        kv_inputs, v_slice, sliceParams, inputs[1].type + "slice_value");

    k_v_dims.erase(k_v_dims.begin());
    std::vector<synTensor> key_squeezed;
    auto key_states =
        createTensorNoPresist("key_states", inputs[1].type, k_v_dims);
    key_squeezed.push_back(key_states);

    synSqueezeParams squeezeParams;
    squeezeParams.axis = 4;

    AddNodeSqueeze(k_slice, key_squeezed, squeezeParams, guid_ + "squeeze_key");

    std::vector<synTensor> value_squeezed;
    auto value_states =
        createTensorNoPresist("value_states", inputs[1].type, k_v_dims);
    value_squeezed.push_back(value_states);

    AddNodeSqueeze(
        v_slice, value_squeezed, squeezeParams, guid_ + "squeeze_value");

    std::vector<int64_t> q_dims = std::vector<int64_t>(inputs[0].dims);
    std::vector<int64_t> qt_dims(q_dims.cbegin(), q_dims.cend());
    int rank = q_dims.size();
    qt_dims[rank - 3] = q_dims[rank - 2];
    qt_dims[rank - 2] = q_dims[rank - 3];

    std::vector<int> axis = {0, 2, 1, 3};
    synTransposeParams trans_params;
    for (size_t i = 0; i < axis.size(); i++) {
      trans_params.permutation[i] =
          static_cast<TransposePermutationDim>(axis[i]);
    }
    trans_params.tensorDim = rank;

    std::vector<synTensor> q_inputs;
    q_inputs.push_back(createTensorFromCT(&ct, 0));

    std::vector<synTensor> q_transpose;
    auto q_t = createTensorNoPresist("q_t", inputs[0].type, qt_dims);
    q_transpose.push_back(q_t);

    AddNodeTranspose(
        q_inputs, q_transpose, trans_params, guid_ + "transpose_q");

    std::vector<int64_t> kvt_dims(k_v_dims.cbegin(), k_v_dims.cend());
    kvt_dims[rank - 3] = k_v_dims[rank - 2];
    kvt_dims[rank - 2] = k_v_dims[rank - 3];

    std::vector<synTensor> k_transpose;
    auto k_t = createTensorNoPresist("k_t", inputs[1].type, kvt_dims);
    k_transpose.push_back(k_t);

    AddNodeTranspose(
        key_squeezed, k_transpose, trans_params, guid_ + "transpose_k");

    std::vector<synTensor> v_transpose;
    auto v_t = createTensorNoPresist("v_t", inputs[1].type, kvt_dims);
    v_transpose.push_back(v_t);

    AddNodeTranspose(
        value_squeezed, v_transpose, trans_params, guid_ + "transpose_v");

    std::vector<synTensor> attn_outputs;
    if (params.is_GQA) {
      int q_heads = qt_dims[1];
      int kv_heads = kvt_dims[1];
      int q_heads_per_group = q_heads / kv_heads;

      std::vector<int64_t> q_reshape;
      q_reshape.push_back(qt_dims[0]);
      q_reshape.push_back(kv_heads);
      q_reshape.push_back(q_heads_per_group);
      q_reshape.push_back(qt_dims[2]);
      q_reshape.push_back(qt_dims[3]);

      std::vector<synTensor> q_out_reshape;
      auto q_r = createTensorNoPresist("q_r", inputs[0].type, q_reshape);
      q_out_reshape.push_back(q_r);
      AddNodeReshape(q_transpose, q_out_reshape, guid_ + "reshape_q");

      kvt_dims.insert(kvt_dims.begin() + 2, 1);

      std::vector<synTensor> k_out_reshape;
      auto k_r = createTensorNoPresist("k_r", inputs[1].type, kvt_dims);
      k_out_reshape.push_back(k_r);
      AddNodeReshape(k_transpose, k_out_reshape, guid_ + "reshape_k");

      std::vector<synTensor> v_out_reshape;
      auto v_r = createTensorNoPresist("v_r", inputs[1].type, kvt_dims);
      v_out_reshape.push_back(v_r);
      AddNodeReshape(v_transpose, v_out_reshape, guid_ + "reshape_v");

      std::vector<synTensor> attn_inputs;
      attn_inputs.push_back(q_r);
      attn_inputs.push_back(k_r);
      attn_inputs.push_back(v_r);
      if (!params.sdpa_params.is_causal) {
        attn_inputs.push_back(createTensor(inputs[3].dims.size(),
                                           inputs[3].type,
                                           inputs[3].dims,
                                           true,
                                           inputs[3].name));
      }
      if (params.fp8_sdpa) {
        attn_inputs.push_back(nullptr);  // Mask
        attn_inputs.push_back(nullptr);  // Seed
        for (size_t i = 3; i < inputs.size() - 2; i++) {
          attn_inputs.push_back(createTensor(inputs[i].dims.size(),
                                             inputs[i].type,
                                             inputs[i].dims,
                                             true,
                                             inputs[i].name));
        }
      }
      std::vector<synTensor> attn_outputs_r;
      auto attn = createTensorNoPresist("attn", dtype_, q_reshape);
      attn_outputs_r.push_back(attn);

      if (params.fp8_sdpa) {
        AddNodeSdpaRecomp<phi::dtype::float8_e4m3fn>(attn_inputs,
                                                     attn_outputs_r,
                                                     params.sdpa_params,
                                                     guid_ + "sdpa_recomp");
      } else {
        AddNodeSdpaRecomp<T>(attn_inputs,
                             attn_outputs_r,
                             params.sdpa_params,
                             guid_ + "sdpa_recomp");
      }

      auto attn_o = createTensorNoPresist("attn_o", dtype_, qt_dims);
      attn_outputs.push_back(attn_o);
      AddNodeReshape(attn_outputs_r, attn_outputs, guid_ + "reshape_sdpa");
    } else {
      std::vector<synTensor> attn_inputs;
      attn_inputs.push_back(q_t);
      attn_inputs.push_back(k_t);
      attn_inputs.push_back(v_t);
      if (params.fp8_sdpa) {
        attn_inputs.push_back(nullptr);  // Mask
        attn_inputs.push_back(nullptr);  // Seed
        for (size_t i = 3; i < inputs.size() - 2; i++) {
          attn_inputs.push_back(createTensor(inputs[i].dims.size(),
                                             inputs[i].type,
                                             inputs[i].dims,
                                             true,
                                             inputs[i].name));
        }
      }
      // params.is_causal = true; ==> input[3] is not used
      // input[3] is in use ==> params.is_causal = false;
      auto attn = createTensorNoPresist("attn", dtype_, qt_dims);
      attn_outputs.push_back(attn);

      if (params.fp8_sdpa) {
        AddNodeSdpaRecomp<phi::dtype::float8_e4m3fn>(attn_inputs,
                                                     attn_outputs,
                                                     params.sdpa_params,
                                                     guid_ + "sdpa_recomp");
      } else {
        AddNodeSdpaRecomp<T>(attn_inputs,
                             attn_outputs,
                             params.sdpa_params,
                             guid_ + "sdpa_recomp");
      }
    }

    std::vector<synTensor> attn_out_transpose;
    auto attn_t = createTensorNoPresist("attn_t", dtype_, q_dims);
    attn_out_transpose.push_back(attn_t);

    AddNodeTranspose(attn_outputs,
                     attn_out_transpose,
                     trans_params,
                     guid_ + "transpose_out");

    std::vector<int64_t> attn_reshape;
    attn_reshape.push_back(q_dims[0] * q_dims[1]);
    attn_reshape.push_back(q_dims[2] * q_dims[3]);

    std::vector<synTensor> attn_out_reshape;
    auto attn_r = createTensorNoPresist("attn_r", dtype_, attn_reshape);
    attn_out_reshape.push_back(attn_r);

    AddNodeReshape(attn_out_transpose, attn_out_reshape, guid_ + "reshape_out");

    std::vector<synTensor> mul_inputs;
    if (params.fp8_sdpa) {
      ns_CastKernel::Params cast_to_fp8_params;
      cast_to_fp8_params.round_mode = CAST_ROUND_HALF_NE;
      std::vector<synTensor> attn_out_cast;
      auto attn_c =
          createTensorNoPresist("attn_c", inputs[2].type, attn_reshape);
      attn_out_cast.push_back(attn_c);
      AddNodeConvertToFP8<T>(attn_out_reshape,
                             attn_out_cast,
                             cast_to_fp8_params,
                             guid_ + "cast_out");
      mul_inputs.push_back(attn_c);
    } else {
      mul_inputs.push_back(attn_r);
    }

    mul_inputs.push_back(createTensor(inputs[2].dims.size(),
                                      inputs[2].type,
                                      inputs[2].dims,
                                      true,
                                      inputs[2].name));
    std::vector<synTensor> mul_outputs;
    mul_outputs.push_back(createTensor(outputs[0].dims.size(),
                                       outputs[0].type,
                                       outputs[0].dims,
                                       true,
                                       outputs[0].name));
    synGEMMParams gemm_params;
    if (params.fp8_gemm) {
      gemm_params.transpose_a = false;
      gemm_params.transpose_b = true;
      auto in_scale = createTensorFromCT(&ct, inputs.size() - 2);
      auto out_scale = createTensorFromCT(&ct, inputs.size() - 1);
      mul_inputs.push_back(in_scale);
      mul_inputs.push_back(out_scale);
      AddNodeFusedFP8Gemm<T>(
          mul_inputs, mul_outputs, gemm_params, guid_ + "gemm_fp8");
    } else {
      gemm_params.transpose_a = false;
      gemm_params.transpose_b = false;
      AddNodeBatchGemm(
          mul_inputs, mul_outputs, gemm_params, guid_ + "batchgemm");
    }
  }

 protected:
  synDataType dtype_;
};

template <typename T, typename Context>
void FusedSdpaProjBTMHKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& query_states,
    const phi::DenseTensor& key_value_states,
    const phi::DenseTensor& attn_mask,
    const phi::DenseTensor& linear_weights,
    phi::DenseTensor* out_linear,
    const phi::Scalar& scaling_factor,
    const phi::Scalar& causal,
    const phi::Scalar& mode,
    const paddle::optional<phi::DenseTensor>& d_scale_q,
    const paddle::optional<phi::DenseTensor>& d_scale_k,
    const paddle::optional<phi::DenseTensor>& d_scale_v,
    const paddle::optional<phi::DenseTensor>& q_scale_s,
    const paddle::optional<phi::DenseTensor>& q_scale_o,
    const paddle::optional<phi::DenseTensor>& d_scale_s,
    const paddle::optional<phi::DenseTensor>& linear_in_scale,
    const paddle::optional<phi::DenseTensor>& weight_scale) {
  ConvertTensors ct;
  ct.Add(query_states);
  ct.Add(key_value_states);
  std::vector<DIMS> in_out_dims = ct.GetDims();

  ct.Add(linear_weights);
  if (causal.to<bool>() == false) {
    ct.Add(attn_mask);
  }

  unsigned int flags = 0;
  SDPA_SET_INPUT_AND_FLAGS(d_scale_q.get_ptr(), D_SCALE_Q)
  SDPA_SET_INPUT_AND_FLAGS(d_scale_k.get_ptr(), D_SCALE_K)
  SDPA_SET_INPUT_AND_FLAGS(d_scale_v.get_ptr(), D_SCALE_V)
  SDPA_SET_INPUT_AND_FLAGS(q_scale_s.get_ptr(), Q_SCALE_S)
  SDPA_SET_INPUT_AND_FLAGS(q_scale_o.get_ptr(), Q_SCALE_O)
  SDPA_SET_INPUT_AND_FLAGS(d_scale_s.get_ptr(), D_SCALE_S)
  SDPA_SET_INPUT(linear_in_scale.get_ptr())
  SDPA_SET_INPUT(weight_scale.get_ptr())

  ct.Add(out_linear, false);
  std::vector<DIMS> out_dims = ct.GetDims(false);
  in_out_dims.insert(in_out_dims.end(), out_dims.begin(), out_dims.end());

  std::vector<int64_t> query_states_dims =
      phi::vectorize<int64_t>(query_states.dims());
  std::vector<int64_t> key_value_states_dims =
      phi::vectorize<int64_t>(key_value_states.dims());
  int num_head = query_states_dims[2];
  int num_kv_head = key_value_states_dims[3];

  std::string guid_prefix = "fused_sdpa_proj_causal_";
  if (linear_weights.dtype() == phi::DataType::FLOAT8_E4M3FN) {
    guid_prefix = "fused_fp8_sdpa_proj_causal_";
  }
  if (num_head == num_kv_head) {
    guid_prefix += "MHA_";
  } else {
    guid_prefix += "GQA_";
  }
  guid_prefix += "fwd_";

  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, nullptr_t>(guid_prefix, in_out_dims, nullptr);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    FusedSdpaProjParams params;
    memset(reinterpret_cast<void*>(&params), 0x00, sizeof(FusedSdpaProjParams));
    params.sdpa_params.scale = scaling_factor.to<float>();
    params.sdpa_params.is_causal = causal.to<bool>();
    params.sdpa_params.dropout.ratio = 0.0;
    params.sdpa_params.dropout.disableMaskOut = false;
    params.sdpa_params.is_inference = true;
    params.sdpa_params.softmax_mode =
        static_cast<SdpaSoftmaxMode_t>(mode.to<int>());
    params.sdpa_params.flags = flags;
    params.fp8_sdpa = (query_states.dtype() == phi::DataType::FLOAT8_E4M3FN);
    params.fp8_gemm = (linear_weights.dtype() == phi::DataType::FLOAT8_E4M3FN);
    if (num_head != num_kv_head) {
      params.is_GQA = true;
    }

    FusedSdpaProjBTMH op(guid_prefix, op_info.datatype_);
    op.AddNode<T>(ct, params);
    op.Compile();
    op_info.setOp(op);

    recipe = op_info.GetRecipe();
  }

  std::map<std::string, uint64_t> tensors = ct.GetDeviceAddr();
  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

}  // namespace custom_kernel

std::vector<paddle::Tensor> FusedBaseSdpaProjBTMH(
    const paddle::Tensor& query_states,
    const paddle::Tensor& key_value_states,
    const paddle::optional<paddle::Tensor>& attn_mask,
    const paddle::optional<paddle::Tensor>& valid_seq_len,
    const paddle::Tensor& linear_weights,
    float scaling_factor,
    bool causal,
    int mode,
    const paddle::optional<paddle::Tensor>& d_scale_q,
    const paddle::optional<paddle::Tensor>& d_scale_k,
    const paddle::optional<paddle::Tensor>& d_scale_v,
    const paddle::optional<paddle::Tensor>& q_scale_s,
    const paddle::optional<paddle::Tensor>& q_scale_o,
    const paddle::optional<paddle::Tensor>& d_scale_s,
    const paddle::optional<paddle::Tensor>& linear_in_scale,
    const paddle::optional<paddle::Tensor>& weight_scale) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(
          query_states.place()));
  auto query_states_tensor =
      static_cast<const phi::DenseTensor*>(query_states.impl().get());
  auto key_value_states_tensor =
      static_cast<const phi::DenseTensor*>(key_value_states.impl().get());
  phi::DenseTensor* attn_mask_tensor = nullptr;
  if (attn_mask) {
    auto attn_mask_ptr = *(attn_mask.get_ptr());
    attn_mask_tensor =
        static_cast<phi::DenseTensor*>(attn_mask_ptr.impl().get());
  }
  auto linear_weights_tensor =
      static_cast<const phi::DenseTensor*>(linear_weights.impl().get());

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

  // linear_in_scale
  phi::DenseTensor* linear_in_scale_tensor = nullptr;
  if (linear_in_scale) {
    auto linear_in_scale_ptr = *(linear_in_scale.get_ptr());
    linear_in_scale_tensor =
        static_cast<phi::DenseTensor*>(linear_in_scale_ptr.impl().get());
  }

  // weight_scale
  phi::DenseTensor* weight_scale_tensor = nullptr;
  if (weight_scale) {
    auto weight_scale_ptr = *(weight_scale.get_ptr());
    weight_scale_tensor =
        static_cast<phi::DenseTensor*>(weight_scale_ptr.impl().get());
  }

  // allocate memory on device.
  int64_t bsz = query_states.dims()[0];
  int64_t seq_len = query_states.dims()[1];
  int hidden_size = linear_weights.dims()[1];

  std::shared_ptr<phi::DenseTensor> out_linear =
      std::make_shared<phi::DenseTensor>();
  out_linear->Resize(phi::make_ddim({bsz * seq_len, hidden_size}));
  if (query_states.dtype() == phi::DataType::FLOAT8_E4M3FN) {
    dev_ctx->Alloc(out_linear.get(), phi::DataType::BFLOAT16);
  } else {
    dev_ctx->Alloc(out_linear.get(), query_states_tensor->dtype());
  }

  if (!valid_seq_len) {
    if (query_states.dtype() == phi::DataType::FLOAT16) {
      custom_kernel::FusedSdpaProjBTMHKernel<phi::dtype::float16>(
          *dev_ctx,
          *query_states_tensor,
          *key_value_states_tensor,
          attn_mask_tensor ? *attn_mask_tensor : phi::DenseTensor(),
          *linear_weights_tensor,
          out_linear.get(),
          phi::Scalar(scaling_factor),
          phi::Scalar(causal),
          phi::Scalar(mode),
          paddle::optional<phi::DenseTensor>(),
          paddle::optional<phi::DenseTensor>(),
          paddle::optional<phi::DenseTensor>(),
          paddle::optional<phi::DenseTensor>(),
          paddle::optional<phi::DenseTensor>(),
          paddle::optional<phi::DenseTensor>(),
          paddle::optional<phi::DenseTensor>(),
          paddle::optional<phi::DenseTensor>());
    } else if (query_states.dtype() == phi::DataType::BFLOAT16 ||
               query_states.dtype() == phi::DataType::FLOAT8_E4M3FN) {
      custom_kernel::FusedSdpaProjBTMHKernel<phi::dtype::bfloat16>(
          *dev_ctx,
          *query_states_tensor,
          *key_value_states_tensor,
          attn_mask_tensor ? *attn_mask_tensor : phi::DenseTensor(),
          *linear_weights_tensor,
          out_linear.get(),
          phi::Scalar(scaling_factor),
          phi::Scalar(causal),
          phi::Scalar(mode),
          d_scale_q ? *d_scale_q_tensor : paddle::optional<phi::DenseTensor>(),
          d_scale_k ? *d_scale_k_tensor : paddle::optional<phi::DenseTensor>(),
          d_scale_v ? *d_scale_v_tensor : paddle::optional<phi::DenseTensor>(),
          q_scale_s ? *q_scale_s_tensor : paddle::optional<phi::DenseTensor>(),
          q_scale_o ? *q_scale_o_tensor : paddle::optional<phi::DenseTensor>(),
          d_scale_s ? *d_scale_s_tensor : paddle::optional<phi::DenseTensor>(),
          linear_in_scale ? *linear_in_scale_tensor
                          : paddle::optional<phi::DenseTensor>(),
          weight_scale ? *weight_scale_tensor
                       : paddle::optional<phi::DenseTensor>());
    } else {
      throw std::runtime_error("Unsupported data type for FusedSdpaProjKernel");
    }
  }

  return {paddle::Tensor(out_linear)};
}

std::vector<paddle::Tensor> FusedSdpaProjBTMH(
    const paddle::Tensor& query_states,
    const paddle::Tensor& key_value_states,
    const paddle::optional<paddle::Tensor>& attn_mask,
    const paddle::optional<paddle::Tensor>& valid_seq_len,
    const paddle::Tensor& linear_weights,
    float scaling_factor,
    bool causal = false,
    int mode = 0) {
  return FusedBaseSdpaProjBTMH(query_states,
                               key_value_states,
                               attn_mask,
                               valid_seq_len,
                               linear_weights,
                               scaling_factor,
                               causal,
                               mode,
                               paddle::optional<paddle::Tensor>(),
                               paddle::optional<paddle::Tensor>(),
                               paddle::optional<paddle::Tensor>(),
                               paddle::optional<paddle::Tensor>(),
                               paddle::optional<paddle::Tensor>(),
                               paddle::optional<paddle::Tensor>(),
                               paddle::optional<paddle::Tensor>(),
                               paddle::optional<paddle::Tensor>());
}

std::vector<paddle::Tensor> FusedSdpaFp8ProjBTMH(
    const paddle::Tensor& query_states,
    const paddle::Tensor& key_value_states,
    const paddle::optional<paddle::Tensor>& attn_mask,
    const paddle::optional<paddle::Tensor>& valid_seq_len,
    const paddle::Tensor& linear_weights,
    const paddle::optional<paddle::Tensor>& d_scale_q,
    const paddle::optional<paddle::Tensor>& d_scale_k,
    const paddle::optional<paddle::Tensor>& d_scale_v,
    const paddle::optional<paddle::Tensor>& q_scale_s,
    const paddle::optional<paddle::Tensor>& q_scale_o,
    const paddle::optional<paddle::Tensor>& d_scale_s,
    const paddle::optional<paddle::Tensor>& linear_in_scale,
    const paddle::optional<paddle::Tensor>& weight_scale,
    float scaling_factor,
    bool causal = false,
    int mode = 0) {
  return FusedBaseSdpaProjBTMH(query_states,
                               key_value_states,
                               attn_mask,
                               valid_seq_len,
                               linear_weights,
                               scaling_factor,
                               causal,
                               mode,
                               d_scale_q,
                               d_scale_k,
                               d_scale_v,
                               q_scale_s,
                               q_scale_o,
                               d_scale_s,
                               linear_in_scale,
                               weight_scale);
}

std::vector<std::vector<int64_t>> FusedSdpaProjBTMHShape(
    const std::vector<int64_t>& query_states_shape,
    const std::vector<int64_t>& key_value_states_shape,
    const paddle::optional<std::vector<int64_t>>& attn_mask_shape,
    const paddle::optional<std::vector<int64_t>>& valid_seq_len_shape,
    const std::vector<int64_t>& linear_weights_shape) {
  int64_t bsz = query_states_shape[0];
  int64_t seq_len = query_states_shape[1];
  int hidden_size = linear_weights_shape[1];
  return {{bsz * seq_len, hidden_size}};
}

std::vector<paddle::DataType> FusedSdpaProjBTMHDtype(
    const paddle::DataType& query_states_dtype,
    const paddle::DataType& key_value_states_dtype,
    const paddle::optional<paddle::DataType>& attn_mask_dtype,
    const paddle::optional<paddle::DataType>& valid_seq_len_dtype,
    const paddle::DataType& linear_weights_dtype) {
  return {query_states_dtype};
}

PD_BUILD_OP(fused_sdpa_proj_t)
    .Inputs({"query_states",
             "key_value_states",
             paddle::Optional("attn_mask"),
             paddle::Optional("valid_seq_len"),
             "linear_weights"})
    .Outputs({"out_linear"})
    .Attrs({"scaling_factor: float", "causal:bool", "softmax_mode:int"})
    .SetKernelFn(PD_KERNEL(FusedSdpaProjBTMH))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedSdpaProjBTMHShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedSdpaProjBTMHDtype));

PD_BUILD_OP(fused_fp8_sdpa_proj_t)
    .Inputs({"query_states",
             "key_value_states",
             paddle::Optional("attn_mask"),
             paddle::Optional("valid_seq_len"),
             "linear_weights",
             paddle::Optional("d_scale_q"),
             paddle::Optional("d_scale_k"),
             paddle::Optional("d_scale_v"),
             paddle::Optional("q_scale_s"),
             paddle::Optional("q_scale_o"),
             paddle::Optional("d_scale_s"),
             paddle::Optional("linear_in_scale"),
             paddle::Optional("weight_scale")})
    .Outputs({"out_linear"})
    .Attrs({"scaling_factor:float", "causal:bool", "softmax_mode:int"})
    .SetKernelFn(PD_KERNEL(FusedSdpaFp8ProjBTMH))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedSdpaProjBTMHShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedSdpaProjBTMHDtype));
