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

struct FusedQkvRopeParams {
  ns_LayerNormKernel::Params rmsnorm_params;
  int head_dim;
  int num_head;
  int kv_num_head;

  bool use_neox_style = true;
  bool transpose = true;
  bool with_qkv_biases = false;
  bool fp8_proj = false;
  bool fp8_out = false;
  bool use_qk_rmsnorm = false;
};

class FusedQkvRope : public HpuFusedOperator {
 public:
  explicit FusedQkvRope(std::string guid_prefix, synDataType dtype)
      : HpuFusedOperator(guid_prefix), dtype_(dtype) {}
  template <typename T>
  void AddNode(ConvertTensors& ct, FusedQkvRopeParams& params) {
    auto ins = ct.GetTensors();
    auto outs = ct.GetTensors(false);

    int src_index = 0;
    int qkv_weights_index = 1;
    int rotary_embs_index = 2;
    int qkv_biases_index = 3;
    int scale_input_index = (params.with_qkv_biases ? (qkv_biases_index + 1)
                                                    : (rotary_embs_index + 1));
    int q_gamma_index =
        params.fp8_proj
            ? params.fp8_out ? (scale_input_index + 5) : (scale_input_index + 2)
            : scale_input_index;

    auto src = createTensorFromCT(&ct, src_index);
    auto qkv_weights = createTensorFromCT(&ct, qkv_weights_index);

    std::vector<synTensor> linear_inputs;
    linear_inputs.push_back(src);
    linear_inputs.push_back(qkv_weights);
    synTensor qkv_biases;
    if (params.with_qkv_biases) {
      qkv_biases = createTensorFromCT(&ct, qkv_biases_index);
    }

    auto tmp_dims = ins[src_index].dims;
    auto wt_dims = ins[qkv_weights_index].dims;
    tmp_dims[2] = params.transpose ? wt_dims[0] : wt_dims[1];
    auto qkv_out = createTensorNoPresist("qkv_out", dtype_, tmp_dims);
    std::vector<synTensor> linear_outputs;
    linear_outputs.push_back(qkv_out);

    std::vector<synTensor> reshape_inputs;

    if ((!params.fp8_proj) &&
        (params.transpose)) {  // bfloat16 + transpose=true
      if (params.with_qkv_biases) {
        linear_inputs.push_back(qkv_biases);
      }
      AddNodeLinear<T>(linear_inputs, linear_outputs, guid_ + "linear");
      reshape_inputs.push_back(qkv_out);
    } else {
      synGEMMParams gemm_params;
      gemm_params.transpose_a = false;
      gemm_params.transpose_b = params.transpose;

      if (params.fp8_proj) {
        auto scale_input = createTensorFromCT(&ct, scale_input_index);
        auto scale_weight = createTensorFromCT(&ct, scale_input_index + 1);
        linear_inputs.push_back(scale_input);
        linear_inputs.push_back(scale_weight);
        AddNodeFusedFP8Gemm<T>(
            linear_inputs, linear_outputs, gemm_params, guid_ + "fp8_gemm");
      } else {  // bfloat16 + transpose=false
        AddNodeBatchGemm(
            linear_inputs, linear_outputs, gemm_params, guid_ + "batchgemm");
      }

      if (params.with_qkv_biases) {
        auto qkv_out_with_bias =
            createTensorNoPresist("qkv_out_with_bias", dtype_, tmp_dims);
        std::vector<synTensor> qkv_add_inputs;
        qkv_add_inputs.push_back(qkv_out);
        qkv_add_inputs.push_back(qkv_biases);
        std::vector<synTensor> qkv_add_outputs = {qkv_out_with_bias};
        AddNodeAdd<T>(qkv_add_inputs, qkv_add_outputs, guid_ + "add_bias");
        reshape_inputs.push_back(qkv_out_with_bias);
      } else {
        reshape_inputs.push_back(qkv_out);
      }
    }

    auto reshape_dims = ins[src_index].dims;
    reshape_dims[2] = params.num_head + 2 * params.kv_num_head;
    reshape_dims.push_back(params.head_dim);

    std::vector<synTensor> reshape_outputs;
    auto reshape_out =
        createTensorNoPresist("reshape_out", dtype_, reshape_dims);
    reshape_outputs.push_back(reshape_out);
    AddNodeReshape(reshape_inputs, reshape_outputs, guid_ + "reshape_qkv");

    auto kv_dims = outs[1].dims;
    kv_dims.erase(kv_dims.begin());
    auto q_split = createTensorNoPresist("q_split", dtype_, outs[0].dims);
    auto k_split = createTensorNoPresist("k_split", dtype_, kv_dims);
    auto v_split = createTensorNoPresist("v_split", dtype_, kv_dims);
    std::vector<synTensor> split_outpus;
    split_outpus.push_back(q_split);
    split_outpus.push_back(k_split);
    split_outpus.push_back(v_split);

    synSplitParams splitParams;
    splitParams.axis = 1;
    AddNodeSplit(reshape_outputs, split_outpus, splitParams, guid_ + "split");

    std::vector<synTensor> rotary_embs_inputs;
    auto rotary_embs_c = createTensorFromCT(&ct, rotary_embs_index);
    rotary_embs_inputs.push_back(rotary_embs_c);

    auto rotary_embs_dims = ins[rotary_embs_index].dims;
    rotary_embs_dims[0] = 1;

    std::vector<synTensor> cos_inputs;
    auto cos_in = createTensorNoPresist("cos_in", dtype_, rotary_embs_dims);
    cos_inputs.push_back(cos_in);

    synSliceParamsV2 sliceParams;
    for (uint64_t i = 0; i < rotary_embs_dims.size(); i++) {
      sliceParams.axes[i] = i;
      sliceParams.steps[i] = 1;
      sliceParams.starts[i] = 0;
      sliceParams.ends[i] = rotary_embs_dims[rotary_embs_dims.size() - 1 - i];
    }
    AddNodeSlice(
        rotary_embs_inputs, cos_inputs, sliceParams, guid_ + "slice_cos");

    std::vector<synTensor> sin_inputs;
    auto sin_in = createTensorNoPresist("sin_in", dtype_, rotary_embs_dims);
    sin_inputs.push_back(sin_in);
    sliceParams.starts[rotary_embs_dims.size() - 1] = 1;
    sliceParams.ends[rotary_embs_dims.size() - 1] = 2;
    AddNodeSlice(
        rotary_embs_inputs, sin_inputs, sliceParams, guid_ + "slice_sin");

    rotary_embs_dims.erase(rotary_embs_dims.begin());
    auto sin_sq =
        createTensorNoPresist("sin_squeezed", dtype_, rotary_embs_dims);
    std::vector<synTensor> sin_squeezed;
    sin_squeezed.push_back(sin_sq);

    synSqueezeParams squeezeParams;
    squeezeParams.axis = 4;
    AddNodeSqueeze(
        sin_inputs, sin_squeezed, squeezeParams, guid_ + "squeeze_sin");

    auto cos_sq =
        createTensorNoPresist("cos_squeezed", dtype_, rotary_embs_dims);
    std::vector<synTensor> cos_squeezed;
    cos_squeezed.push_back(cos_sq);
    AddNodeSqueeze(
        cos_inputs, cos_squeezed, squeezeParams, guid_ + "squeeze_cos");

    std::vector<synTensor> inputs_q;
    std::vector<synTensor> outputs_q;
    std::vector<synTensor> inputs_k;
    std::vector<synTensor> outputs_k;

    if (params.use_qk_rmsnorm) {
      auto q_rmsnorm = createTensorNoPresist("q_rmsnorm", dtype_, outs[0].dims);
      auto k_rmsnorm = createTensorNoPresist("k_rmsnorm", dtype_, kv_dims);
      synTensor q_gamma = createTensorFromCT(&ct, q_gamma_index);
      synTensor k_gamma = createTensorFromCT(&ct, q_gamma_index + 1);

      auto tmp_q_dims = outs[0].dims;
      tmp_q_dims[3] = 1;
      auto q_rmsnorm_var =
          createTensorNoPresist("q_rmsnorm_var", dtype_, tmp_q_dims);

      auto tmp_k_dims = kv_dims;
      tmp_k_dims[3] = 1;
      auto k_rmsnorm_var =
          createTensorNoPresist("k_rmsnorm_var", dtype_, tmp_k_dims);

      std::vector<synTensor> rmsnorm_inputs_q;
      rmsnorm_inputs_q.push_back(q_split);
      rmsnorm_inputs_q.push_back(q_gamma);
      std::vector<synTensor> rmsnorm_outputs_q;
      rmsnorm_outputs_q.push_back(q_rmsnorm);
      rmsnorm_outputs_q.push_back(q_rmsnorm_var);

      AddNodeRmsNorm<T>(rmsnorm_inputs_q,
                        rmsnorm_outputs_q,
                        params.rmsnorm_params,
                        guid_ + "q_rmsnorm");

      std::vector<synTensor> rmsnorm_inputs_k;
      rmsnorm_inputs_k.push_back(k_split);
      rmsnorm_inputs_k.push_back(k_gamma);
      std::vector<synTensor> rmsnorm_outputs_k;
      rmsnorm_outputs_k.push_back(k_rmsnorm);
      rmsnorm_outputs_k.push_back(k_rmsnorm_var);

      AddNodeRmsNorm<T>(rmsnorm_inputs_k,
                        rmsnorm_outputs_k,
                        params.rmsnorm_params,
                        guid_ + "k_rmsnorm");

      inputs_q.push_back(q_rmsnorm);
      inputs_k.push_back(k_rmsnorm);
    } else {
      inputs_q.push_back(q_split);
      inputs_k.push_back(k_split);
    }

    inputs_q.push_back(sin_sq);
    inputs_q.push_back(cos_sq);

    synTensor q_states = nullptr;
    if (params.fp8_out) {
      q_states = createTensorNoPresist("q_states", dtype_, outs[0].dims);
    } else {
      q_states = createTensorFromCT(&ct, 0, false);
    }
    outputs_q.push_back(q_states);

    ns_RoPESt2::ParamsV2 ropeParams;
    ropeParams.offset = 0;
    ropeParams.mode = params.use_neox_style
                          ? ROTARY_POS_EMBEDDING_MODE_BLOCKWISE
                          : ROTARY_POS_EMBEDDING_MODE_PAIRWISE;
    AddNodeRope<T>(inputs_q, outputs_q, ropeParams, guid_ + "rope_q");

    inputs_k.push_back(sin_sq);
    inputs_k.push_back(cos_sq);

    auto k_rope = createTensorNoPresist("k_rope", dtype_, kv_dims);
    outputs_k.push_back(k_rope);
    AddNodeRope<T>(inputs_k, outputs_k, ropeParams, guid_ + "rope_k");

    std::vector<synTensor> inputs_concat;
    if (params.fp8_out) {
      ns_CastKernel::Params cast_to_fp8_params;
      cast_to_fp8_params.round_mode = CAST_ROUND_HALF_NE;
      auto scale_q = createTensorFromCT(&ct, scale_input_index + 2);
      auto scale_k = createTensorFromCT(&ct, scale_input_index + 3);
      auto scale_v = createTensorFromCT(&ct, scale_input_index + 4);

      auto q_state_fp8 = createTensorFromCT(&ct, 0, false);
      std::vector<synTensor> cast_q_ins = {q_states, scale_q};
      std::vector<synTensor> cast_q_outs = {q_state_fp8};
      AddNodeConvertToFP8<T>(
          cast_q_ins, cast_q_outs, cast_to_fp8_params, guid_ + "cast_q");

      auto k_state_fp8 = createTensorNoPresist(
          "k_state_fp8", ins[qkv_weights_index].type, kv_dims);
      std::vector<synTensor> cast_k_ins = {k_rope, scale_k};
      std::vector<synTensor> cast_k_outs = {k_state_fp8};
      AddNodeConvertToFP8<T>(
          cast_k_ins, cast_k_outs, cast_to_fp8_params, guid_ + "cast_k");

      auto v_state_fp8 = createTensorNoPresist(
          "v_state_fp8", ins[qkv_weights_index].type, kv_dims);
      std::vector<synTensor> cast_v_ins = {v_split, scale_v};
      std::vector<synTensor> cast_v_outs = {v_state_fp8};
      AddNodeConvertToFP8<T>(
          cast_v_ins, cast_v_outs, cast_to_fp8_params, guid_ + "cast_v");
      inputs_concat.push_back(k_state_fp8);
      inputs_concat.push_back(v_state_fp8);
    } else {
      inputs_concat.push_back(k_rope);
      inputs_concat.push_back(v_split);
    }

    kv_dims[0] *= 2;
    auto kv_concat = createTensorNoPresist("kv_concat", outs[1].type, kv_dims);
    std::vector<synTensor> outputs_concat;
    outputs_concat.push_back(kv_concat);

    synConcatenateParams concatParams;
    concatParams.axis = 3;
    AddNodeConcat(
        inputs_concat, outputs_concat, concatParams, guid_ + "concat");

    std::vector<synTensor> outputs_stack;

    auto kv_state = createTensorFromCT(&ct, 1, false);
    outputs_stack.push_back(kv_state);
    AddNodeReshape(outputs_concat, outputs_stack, guid_ + "reshaped_kv");
  }

 protected:
  synDataType dtype_;
};

template <typename T, typename Context>
void FusedQkvRopeKernel(const Context& dev_ctx,
                        const phi::DenseTensor& src,
                        const phi::DenseTensor& qkv_weights,
                        const paddle::optional<phi::DenseTensor>& qkv_biases,
                        const phi::DenseTensor& rotary_embs,
                        const paddle::optional<phi::DenseTensor>& scale_input,
                        const paddle::optional<phi::DenseTensor>& scale_weight,
                        const paddle::optional<phi::DenseTensor>& scale_q,
                        const paddle::optional<phi::DenseTensor>& scale_k,
                        const paddle::optional<phi::DenseTensor>& scale_v,
                        phi::DenseTensor* query_states,
                        phi::DenseTensor* key_value_states,
                        const paddle::optional<phi::DenseTensor>& q_norm_weight,
                        const paddle::optional<phi::DenseTensor>& k_norm_weight,
                        const phi::Scalar& head_dim,
                        const phi::Scalar& num_head,
                        const phi::Scalar& total_batch,
                        const phi::Scalar& transpose,
                        const phi::Scalar& use_neox_style,
                        const phi::Scalar& epsilon) {
  int total_batch_ = total_batch.to<int>();
  std::vector<int64_t> src_dims = phi::vectorize<int64_t>(src.dims());
  int bsz_seqlen = src_dims[0];
  int seq_len = bsz_seqlen / total_batch_;
  src_dims[0] = total_batch_;
  src_dims.insert(src_dims.begin() + 1, seq_len);
  phi::DenseTensor src_resize(src);
  src_resize.Resize(phi::make_ddim(src_dims));

  std::vector<int64_t> qkv_weights_dims =
      phi::vectorize<int64_t>(qkv_weights.dims());

  int head_dim_ = head_dim.to<int>();
  int num_head_ = num_head.to<int>();
  bool transpose_ = transpose.to<bool>();
  bool use_neox_style_ = use_neox_style.to<bool>();
  const int64_t fused_hidden_size =
      transpose_ ? qkv_weights_dims[0] : qkv_weights_dims[1];
  const int kv_num_head =
      (fused_hidden_size - num_head_ * head_dim_) / head_dim_ / 2;

  ConvertTensors ct;
  ct.Add(src_resize);
  ct.Add(qkv_weights);
  ct.Add(rotary_embs);
  ct.Add(query_states, false);
  ct.Add(key_value_states, false);

  std::string guid_prefix = "fused_qkv_rope";
  if (qkv_biases) {
    ct.Add(qkv_biases.get());
    guid_prefix += "_bias";
  }

  if (scale_input && scale_weight) {
    guid_prefix += "_fp8";
    ct.Add(scale_input.get());
    ct.Add(scale_weight.get());
    if (scale_q && scale_k && scale_v) {
      ct.Add(scale_q.get());
      ct.Add(scale_k.get());
      ct.Add(scale_v.get());
      guid_prefix += "_hf8";
    } else if (scale_q || scale_k || scale_v) {
      throw std::runtime_error(
          "Need all scale_q, scale_k and scale_v for FusedFp8QkvRopeKernel");
    } else {
      guid_prefix += "_bf16";
    }
  } else if (scale_input || scale_weight) {
    throw std::runtime_error(
        "Need both scale_input and scale_weight for FusedFp8QkvRopeKernel");
  }

  if (q_norm_weight && k_norm_weight) {
    guid_prefix += "_qk_norm";
    ct.Add(q_norm_weight.get());
    ct.Add(k_norm_weight.get());
  } else if (q_norm_weight || k_norm_weight) {
    throw std::runtime_error(
        "Need both q_norm_weight and k_norm_weight for FusedQkvRopeKernel");
  }

  guid_prefix += "_fwd_";

  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, nullptr_t>(
      guid_prefix, {src_dims, qkv_weights_dims}, nullptr);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    FusedQkvRopeParams params;
    memset(reinterpret_cast<void*>(&params), 0x00, sizeof(FusedQkvRopeParams));
    params.rmsnorm_params.epsValid = true;
    params.rmsnorm_params.eps = epsilon.to<float>();
    params.head_dim = head_dim_;
    params.num_head = num_head_;
    params.kv_num_head = kv_num_head;
    params.transpose = transpose_;
    params.use_neox_style = use_neox_style_;

    if (qkv_biases) {
      params.with_qkv_biases = true;
    }
    if (scale_input) {
      params.fp8_proj = true;
    }
    if (scale_q) {
      params.fp8_out = true;
    }
    if (q_norm_weight && k_norm_weight) {
      params.use_qk_rmsnorm = true;
    }

    FusedQkvRope op(guid_prefix, op_info.datatype_);
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

template <typename Context>
void CallFusedQkvRopeKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& src,
    const phi::DenseTensor& qkv_weights,
    const paddle::optional<phi::DenseTensor>& qkv_biases,
    const phi::DenseTensor& rotary_embs,
    const paddle::optional<phi::DenseTensor>& scale_input,
    const paddle::optional<phi::DenseTensor>& scale_weight,
    const paddle::optional<phi::DenseTensor>& scale_q,
    const paddle::optional<phi::DenseTensor>& scale_k,
    const paddle::optional<phi::DenseTensor>& scale_v,
    phi::DenseTensor* query_states,
    phi::DenseTensor* key_value_states,
    const paddle::optional<phi::DenseTensor>& q_norm,
    const paddle::optional<phi::DenseTensor>& k_norm,
    const phi::Scalar& head_dim,
    const phi::Scalar& num_head,
    const phi::Scalar& total_batch,
    const phi::Scalar& transpose,
    const phi::Scalar& use_neox_style,
    const phi::Scalar& epsilon) {
  if (src.dtype() == phi::DataType::FLOAT16) {
    custom_kernel::FusedQkvRopeKernel<phi::dtype::float16>(dev_ctx,
                                                           src,
                                                           qkv_weights,
                                                           qkv_biases,
                                                           rotary_embs,
                                                           scale_input,
                                                           scale_weight,
                                                           scale_q,
                                                           scale_k,
                                                           scale_v,
                                                           query_states,
                                                           key_value_states,
                                                           q_norm,
                                                           k_norm,
                                                           head_dim,
                                                           num_head,
                                                           total_batch,
                                                           transpose,
                                                           use_neox_style,
                                                           epsilon);
  } else if (src.dtype() == phi::DataType::BFLOAT16) {
    custom_kernel::FusedQkvRopeKernel<phi::dtype::bfloat16>(dev_ctx,
                                                            src,
                                                            qkv_weights,
                                                            qkv_biases,
                                                            rotary_embs,
                                                            scale_input,
                                                            scale_weight,
                                                            scale_q,
                                                            scale_k,
                                                            scale_v,
                                                            query_states,
                                                            key_value_states,
                                                            q_norm,
                                                            k_norm,
                                                            head_dim,
                                                            num_head,
                                                            total_batch,
                                                            transpose,
                                                            use_neox_style,
                                                            epsilon);
  } else {
    throw std::runtime_error("Unsupported data type for FusedQkvRopeKernel");
  }
}

std::vector<paddle::Tensor> FusedQkvRopeImpl(
    const paddle::Tensor& src,
    const paddle::Tensor& qkv_weights,
    const paddle::optional<paddle::Tensor>& qkv_biases,
    const paddle::Tensor& rotary_embs,
    const paddle::optional<paddle::Tensor>& q_norm_weights,
    const paddle::optional<paddle::Tensor>& k_norm_weights,
    int head_dim,
    int num_head,
    int total_batch,
    bool transpose,
    bool use_neox_style,
    float epsilon) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(src.place()));
  auto src_tensor = static_cast<const phi::DenseTensor*>(src.impl().get());
  auto qkv_weights_tensor =
      static_cast<const phi::DenseTensor*>(qkv_weights.impl().get());
  auto rotary_embs_tensor =
      static_cast<const phi::DenseTensor*>(rotary_embs.impl().get());

  auto qkv_biases_tensor = paddle::optional<phi::DenseTensor>();
  if (qkv_biases) {
    auto qkv_biases_dt =
        static_cast<phi::DenseTensor*>(qkv_biases->impl().get());
    qkv_biases_tensor = paddle::optional<phi::DenseTensor>(*qkv_biases_dt);
  }

  auto q_norm_weights_tensor = paddle::optional<phi::DenseTensor>();
  if (q_norm_weights) {
    auto q_norm_weights_dt =
        static_cast<phi::DenseTensor*>(q_norm_weights->impl().get());
    q_norm_weights_tensor =
        paddle::optional<phi::DenseTensor>(*q_norm_weights_dt);
  }

  auto k_norm_weights_tensor = paddle::optional<phi::DenseTensor>();
  if (k_norm_weights) {
    auto k_norm_weights_dt =
        static_cast<phi::DenseTensor*>(k_norm_weights->impl().get());
    k_norm_weights_tensor =
        paddle::optional<phi::DenseTensor>(*k_norm_weights_dt);
  }

  // allocate memory on device.
  int64_t bsz = src.dims()[0];
  int64_t seq_len = bsz / total_batch;
  int64_t fused_hidden_size =
      transpose ? qkv_weights.dims()[0] : qkv_weights.dims()[1];
  int kv_num_head = (fused_hidden_size - num_head * head_dim) / head_dim / 2;

  std::shared_ptr<phi::DenseTensor> query_states =
      std::make_shared<phi::DenseTensor>();
  query_states->Resize(
      phi::make_ddim({total_batch, seq_len, num_head, head_dim}));
  dev_ctx->Alloc(query_states.get(), src_tensor->dtype());

  std::shared_ptr<phi::DenseTensor> key_value_states =
      std::make_shared<phi::DenseTensor>();
  key_value_states->Resize(
      phi::make_ddim({2, total_batch, seq_len, kv_num_head, head_dim}));
  dev_ctx->Alloc(key_value_states.get(), src_tensor->dtype());

  CallFusedQkvRopeKernel(*dev_ctx,
                         *src_tensor,
                         *qkv_weights_tensor,
                         qkv_biases_tensor,
                         *rotary_embs_tensor,
                         paddle::optional<phi::DenseTensor>(),
                         paddle::optional<phi::DenseTensor>(),
                         paddle::optional<phi::DenseTensor>(),
                         paddle::optional<phi::DenseTensor>(),
                         paddle::optional<phi::DenseTensor>(),
                         query_states.get(),
                         key_value_states.get(),
                         q_norm_weights_tensor,
                         k_norm_weights_tensor,
                         phi::Scalar(head_dim),
                         phi::Scalar(num_head),
                         phi::Scalar(total_batch),
                         phi::Scalar(transpose),
                         phi::Scalar(use_neox_style),
                         phi::Scalar(epsilon));
  return {paddle::Tensor(query_states), paddle::Tensor(key_value_states)};
}

std::vector<std::vector<int64_t>> FusedQkvRopeShape(
    const std::vector<int64_t>& src_shape,
    const std::vector<int64_t>& qkv_weights_shape,
    const paddle::optional<std::vector<int64_t>>& qkv_biases_shape,
    const std::vector<int64_t>& rotary_embs_shape,
    const paddle::optional<std::vector<int64_t>>& q_norm_weights_shape,
    const paddle::optional<std::vector<int64_t>>& k_norm_weights_shape,
    int head_dim,
    int num_head,
    int total_batch,
    bool transpose,
    bool use_neox_style) {
  int64_t bsz = src_shape[0];
  int64_t seq_len = bsz / total_batch;
  int64_t fused_hidden_size =
      transpose ? qkv_weights_shape[0] : qkv_weights_shape[1];
  int kv_num_head = (fused_hidden_size - num_head * head_dim) / head_dim / 2;
  return {{total_batch, seq_len, num_head, head_dim},
          {2, total_batch, seq_len, kv_num_head, head_dim}};
}

std::vector<paddle::DataType> FusedQkvRopeDtype(
    const paddle::DataType& src_dtype,
    const paddle::DataType& qkv_weights_dtype,
    const paddle::optional<paddle::DataType>& qkv_biases_dtype,
    const paddle::DataType& rotary_embs_dtype,
    const paddle::optional<paddle::DataType>& q_norm_weights_dtype,
    const paddle::optional<paddle::DataType>& k_norm_weights_dtype) {
  return {src_dtype, src_dtype};
}

PD_BUILD_OP(fused_qkv_rope_bf16)
    .Inputs({"src",
             "qkv_weights",
             paddle::Optional("qkv_biases"),
             "rotary_embs",
             paddle::Optional("q_norm_weights"),
             paddle::Optional("k_norm_weights")})
    .Outputs({"query_states", "key_value_states"})
    .Attrs({"head_dim: int",
            "num_head: int",
            "total_batch: int",
            "transpose: bool",
            "use_neox_style: bool",
            "epsilon: float"})
    .SetKernelFn(PD_KERNEL(FusedQkvRopeImpl))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedQkvRopeShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedQkvRopeDtype));

std::vector<paddle::Tensor> FusedFp8QkvRopeImpl(
    const paddle::Tensor& src,
    const paddle::Tensor& qkv_weights,
    const paddle::optional<paddle::Tensor>& qkv_biases,
    const paddle::Tensor& rotary_embs,
    const paddle::optional<paddle::Tensor>& scale_input,
    const paddle::optional<paddle::Tensor>& scale_weight,
    const paddle::optional<paddle::Tensor>& scale_q,
    const paddle::optional<paddle::Tensor>& scale_k,
    const paddle::optional<paddle::Tensor>& scale_v,
    const paddle::optional<paddle::Tensor>& q_norm_weights,
    const paddle::optional<paddle::Tensor>& k_norm_weights,
    int head_dim,
    int num_head,
    int total_batch,
    bool transpose,
    bool use_neox_style,
    float epsilon) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(src.place()));
  auto src_tensor = static_cast<const phi::DenseTensor*>(src.impl().get());
  auto qkv_weights_tensor =
      static_cast<const phi::DenseTensor*>(qkv_weights.impl().get());
  auto rotary_embs_tensor =
      static_cast<const phi::DenseTensor*>(rotary_embs.impl().get());

  auto qkv_biases_tensor = paddle::optional<phi::DenseTensor>();
  if (qkv_biases) {
    auto qkv_biases_dt =
        static_cast<phi::DenseTensor*>(qkv_biases->impl().get());
    qkv_biases_tensor = paddle::optional<phi::DenseTensor>(*qkv_biases_dt);
  }

  auto scale_input_tensor = paddle::optional<phi::DenseTensor>();
  auto scale_weight_tensor = paddle::optional<phi::DenseTensor>();
  if (scale_input) {
    auto scale_input_dt =
        static_cast<phi::DenseTensor*>(scale_input->impl().get());
    scale_input_tensor = paddle::optional<phi::DenseTensor>(*scale_input_dt);
  }
  if (scale_weight) {
    auto scale_weight_dt =
        static_cast<phi::DenseTensor*>(scale_weight->impl().get());
    scale_weight_tensor = paddle::optional<phi::DenseTensor>(*scale_weight_dt);
  }

  auto scale_q_tensor = paddle::optional<phi::DenseTensor>();
  auto scale_k_tensor = paddle::optional<phi::DenseTensor>();
  auto scale_v_tensor = paddle::optional<phi::DenseTensor>();
  if (scale_q) {
    auto scale_q_dt = static_cast<phi::DenseTensor*>(scale_q->impl().get());
    scale_q_tensor = paddle::optional<phi::DenseTensor>(*scale_q_dt);
  }
  if (scale_k) {
    auto scale_k_dt = static_cast<phi::DenseTensor*>(scale_k->impl().get());
    scale_k_tensor = paddle::optional<phi::DenseTensor>(*scale_k_dt);
  }
  if (scale_v) {
    auto scale_v_dt = static_cast<phi::DenseTensor*>(scale_v->impl().get());
    scale_v_tensor = paddle::optional<phi::DenseTensor>(*scale_v_dt);
  }

  auto q_norm_weights_tensor = paddle::optional<phi::DenseTensor>();
  if (q_norm_weights) {
    auto q_norm_weights_dt =
        static_cast<phi::DenseTensor*>(q_norm_weights->impl().get());
    q_norm_weights_tensor =
        paddle::optional<phi::DenseTensor>(*q_norm_weights_dt);
  }

  auto k_norm_weights_tensor = paddle::optional<phi::DenseTensor>();
  if (k_norm_weights) {
    auto k_norm_weights_dt =
        static_cast<phi::DenseTensor*>(k_norm_weights->impl().get());
    k_norm_weights_tensor =
        paddle::optional<phi::DenseTensor>(*k_norm_weights_dt);
  }

  // allocate memory on device.
  int64_t bsz = src.dims()[0];
  int64_t seq_len = bsz / total_batch;
  int64_t fused_hidden_size =
      transpose ? qkv_weights.dims()[0] : qkv_weights.dims()[1];
  int kv_num_head = (fused_hidden_size - num_head * head_dim) / head_dim / 2;

  std::shared_ptr<phi::DenseTensor> query_states =
      std::make_shared<phi::DenseTensor>();
  query_states->Resize(
      phi::make_ddim({total_batch, seq_len, num_head, head_dim}));

  std::shared_ptr<phi::DenseTensor> key_value_states =
      std::make_shared<phi::DenseTensor>();
  key_value_states->Resize(
      phi::make_ddim({2, total_batch, seq_len, kv_num_head, head_dim}));

  if (scale_q) {
    dev_ctx->Alloc(query_states.get(), qkv_weights_tensor->dtype());
    dev_ctx->Alloc(key_value_states.get(), qkv_weights_tensor->dtype());
  } else {
    dev_ctx->Alloc(query_states.get(), src_tensor->dtype());
    dev_ctx->Alloc(key_value_states.get(), src_tensor->dtype());
  }

  CallFusedQkvRopeKernel(*dev_ctx,
                         *src_tensor,
                         *qkv_weights_tensor,
                         qkv_biases_tensor,
                         *rotary_embs_tensor,
                         scale_input_tensor,
                         scale_weight_tensor,
                         scale_q_tensor,
                         scale_k_tensor,
                         scale_v_tensor,
                         query_states.get(),
                         key_value_states.get(),
                         q_norm_weights_tensor,
                         k_norm_weights_tensor,
                         phi::Scalar(head_dim),
                         phi::Scalar(num_head),
                         phi::Scalar(total_batch),
                         phi::Scalar(transpose),
                         phi::Scalar(use_neox_style),
                         phi::Scalar(epsilon));
  return {paddle::Tensor(query_states), paddle::Tensor(key_value_states)};
}

std::vector<std::vector<int64_t>> FusedFp8QkvRopeShape(
    const std::vector<int64_t>& src_shape,
    const std::vector<int64_t>& qkv_weights_shape,
    const paddle::optional<std::vector<int64_t>>& qkv_biases_shape,
    const std::vector<int64_t>& rotary_embs_shape,
    const std::vector<int64_t>& scale_input_shape,
    const std::vector<int64_t>& scale_weight_shape,
    const paddle::optional<std::vector<int64_t>>& q_norm_weights_shape,
    const paddle::optional<std::vector<int64_t>>& k_norm_weights_shape,
    int head_dim,
    int num_head,
    int total_batch,
    bool transpose,
    bool use_neox_style) {
  int64_t bsz = src_shape[0];
  int64_t seq_len = bsz / total_batch;
  int64_t fused_hidden_size =
      transpose ? qkv_weights_shape[0] : qkv_weights_shape[1];
  int kv_num_head = (fused_hidden_size - num_head * head_dim) / head_dim / 2;
  return {{total_batch, seq_len, num_head, head_dim},
          {2, total_batch, seq_len, kv_num_head, head_dim}};
}

std::vector<paddle::DataType> FusedFp8QkvRopeDtype(
    const paddle::DataType& src_dtype,
    const paddle::DataType& qkv_weights_dtype,
    const paddle::optional<paddle::DataType>& qkv_biases_dtype,
    const paddle::DataType& rotary_embs_dtype,
    const paddle::DataType& scale_input_dtype,
    const paddle::DataType& scale_weight_dtype,
    const paddle::optional<paddle::DataType>& q_norm_weights_dtype,
    const paddle::optional<paddle::DataType>& k_norm_weights_dtype) {
  return {src_dtype, src_dtype};
}

PD_BUILD_OP(fused_qkv_rope)
    .Inputs({"src",
             "qkv_weights",
             paddle::Optional("qkv_biases"),
             "rotary_embs",
             "scale_input",
             "scale_weight",
             "scale_q",
             "scale_k",
             "scale_v",
             paddle::Optional("q_rmsnorm_weights"),
             paddle::Optional("k_rmsnorm_weights")})
    .Outputs({"query_states", "key_value_states"})
    .Attrs({"head_dim: int",
            "num_head: int",
            "total_batch: int",
            "transpose: bool",
            "use_neox_style: bool",
            "epsilon: float"})
    .SetKernelFn(PD_KERNEL(FusedFp8QkvRopeImpl))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedFp8QkvRopeShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedFp8QkvRopeDtype));
