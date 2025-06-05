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

struct FusedRmsQkvRopeParams {
  ns_LayerNormKernel::Params rmsnorm_params;

  int head_dim;
  int num_head;
  int kv_num_head;

  bool with_qkv_biases = false;
};

class FusedRmsQkvRopeT : public HpuFusedOperator {
 public:
  explicit FusedRmsQkvRopeT(std::string guid_prefix, synDataType dtype)
      : HpuFusedOperator(guid_prefix), dtype_(dtype) {}
  template <typename T>
  void AddNode(ConvertTensors& ct, FusedRmsQkvRopeParams& params) {
    auto ins = ct.GetTensors();
    auto outs = ct.GetTensors(false);

    synSectionHandle section = createSection();
    auto src = createTensorFromCT(&ct, 0);
    auto residual = createTensorFromCT(&ct, 4, true, section);
    auto residual_out = createTensorFromCT(&ct, 2, false, section);

    std::vector<synTensor> add_residual_in;
    add_residual_in.push_back(src);
    add_residual_in.push_back(residual);

    std::vector<synTensor> add_residual_out;
    add_residual_out.push_back(residual_out);

    AddNodeAdd<T>(add_residual_in, add_residual_out, guid_ + "add_residual");

    auto ln_scales = createTensorFromCT(&ct, 1);

    std::vector<synTensor> rmsnorm_inputs;
    rmsnorm_inputs.push_back(residual_out);
    rmsnorm_inputs.push_back(ln_scales);

    auto tmp_dims = ins[0].dims;
    tmp_dims[2] = 1;
    auto norm_out = createTensorNoPresist("norm_out", dtype_, ins[0].dims);
    auto norm_var = createTensorNoPresist("norm_var", dtype_, tmp_dims);

    std::vector<synTensor> rmsnorm_outputs;
    rmsnorm_outputs.push_back(norm_out);
    rmsnorm_outputs.push_back(norm_var);

    AddNodeRmsNorm<T>(rmsnorm_inputs,
                      rmsnorm_outputs,
                      params.rmsnorm_params,
                      guid_ + "rmsnorm");

    auto qkv_weights = createTensorFromCT(&ct, 2);
    std::vector<synTensor> linear_inputs;
    linear_inputs.push_back(norm_out);
    linear_inputs.push_back(qkv_weights);
    if (params.with_qkv_biases) {
      auto qkv_biases = createTensorFromCT(&ct, 5);
      linear_inputs.push_back(qkv_biases);
    }

    auto wt_dims = ins[2].dims;
    tmp_dims[2] = wt_dims[0];
    auto qkv_out = createTensorNoPresist("qkv_out", dtype_, tmp_dims);
    std::vector<synTensor> linear_outputs;
    linear_outputs.push_back(qkv_out);
    AddNodeLinear<T>(linear_inputs, linear_outputs, guid_ + "linear");

    auto reshape_dims = ins[0].dims;
    reshape_dims[2] = params.num_head + 2 * params.kv_num_head;
    reshape_dims.push_back(params.head_dim);

    std::vector<synTensor> reshape_outputs;
    auto reshape_out =
        createTensorNoPresist("reshape_out", dtype_, reshape_dims);
    reshape_outputs.push_back(reshape_out);
    AddNodeReshape(linear_outputs, reshape_outputs, guid_ + "reshape_qkv");

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
    auto rotary_embs_c = createTensorFromCT(&ct, 3);
    rotary_embs_inputs.push_back(rotary_embs_c);

    auto rotary_embs_dims = ins[3].dims;
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
    inputs_q.push_back(q_split);
    inputs_q.push_back(sin_sq);
    inputs_q.push_back(cos_sq);

    auto q_states = createTensorFromCT(&ct, 0, false);
    outputs_q.push_back(q_states);

    ns_RoPESt2::ParamsV2 ropeParams;
    ropeParams.offset = 0;
    ropeParams.mode = ROTARY_POS_EMBEDDING_MODE_BLOCKWISE;
    AddNodeRope<T>(inputs_q, outputs_q, ropeParams, guid_ + "rope_q");

    std::vector<synTensor> inputs_k;
    std::vector<synTensor> outputs_k;
    inputs_k.push_back(k_split);
    inputs_k.push_back(sin_sq);
    inputs_k.push_back(cos_sq);

    auto k_rope = createTensorNoPresist("k_rope", dtype_, kv_dims);
    outputs_k.push_back(k_rope);
    AddNodeRope<T>(inputs_k, outputs_k, ropeParams, guid_ + "rope_k");

    std::vector<synTensor> inputs_concat;
    std::vector<synTensor> outputs_concat;
    inputs_concat.push_back(k_rope);
    inputs_concat.push_back(v_split);

    kv_dims[0] *= 2;
    auto kv_concat = createTensorNoPresist("kv_concat", dtype_, kv_dims);
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
void FusedRmsQkvRopeTKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& src,
    const phi::DenseTensor& residual,
    const phi::DenseTensor& ln_scales,
    const phi::DenseTensor& qkv_weights,
    const paddle::optional<phi::DenseTensor>& qkv_biases,
    const phi::DenseTensor& rotary_embs,
    phi::DenseTensor* query_states,
    phi::DenseTensor* key_value_states,
    const phi::Scalar& epsilon,
    const phi::Scalar& head_dim,
    const phi::Scalar& num_head) {
  std::vector<int64_t> src_dims = phi::vectorize<int64_t>(src.dims());
  std::vector<int64_t> qkv_weights_dims =
      phi::vectorize<int64_t>(qkv_weights.dims());

  int head_dim_ = head_dim.to<int>();
  int num_head_ = num_head.to<int>();
  const int64_t fused_hidden_size = qkv_weights_dims[0];
  const int kv_num_head =
      (fused_hidden_size - num_head_ * head_dim_) / head_dim_ / 2;

  ConvertTensors ct;
  ct.Add(src);
  ct.Add(ln_scales);
  ct.Add(qkv_weights);
  ct.Add(rotary_embs);
  ct.Add(residual);
  ct.Add(query_states, false);
  ct.Add(key_value_states, false);
  ct.Add(residual, false);

  std::string guid_prefix = "fused_rms_qkv_rope_t_fwd_";
  if (qkv_biases) {
    ct.Add(qkv_biases.get());
    guid_prefix = "fused_rms_qkv_bias_rope_t_fwd_";
  }

  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, nullptr_t>(
      guid_prefix, {src_dims, qkv_weights_dims}, nullptr);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    FusedRmsQkvRopeParams params;
    memset(
        reinterpret_cast<void*>(&params), 0x00, sizeof(FusedRmsQkvRopeParams));
    params.rmsnorm_params.epsValid = true;
    params.rmsnorm_params.eps = epsilon.to<float>();
    params.head_dim = head_dim_;
    params.num_head = num_head_;
    params.kv_num_head = kv_num_head;
    if (qkv_biases) {
      params.with_qkv_biases = true;
    }

    FusedRmsQkvRopeT op(guid_prefix, op_info.datatype_);
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
void CallFusedRmsQkvRopeTKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& src,
    const phi::DenseTensor& residual,
    const phi::DenseTensor& ln_scales,
    const phi::DenseTensor& qkv_weights,
    const paddle::optional<phi::DenseTensor>& qkv_biases,
    const phi::DenseTensor& rotary_embs,
    phi::DenseTensor* query_states,
    phi::DenseTensor* key_value_states,
    const phi::Scalar& epsilon,
    const phi::Scalar& head_dim,
    const phi::Scalar& num_head) {
  if (src.dtype() == phi::DataType::FLOAT16) {
    custom_kernel::FusedRmsQkvRopeTKernel<phi::dtype::float16>(dev_ctx,
                                                               src,
                                                               residual,
                                                               ln_scales,
                                                               qkv_weights,
                                                               qkv_biases,
                                                               rotary_embs,
                                                               query_states,
                                                               key_value_states,
                                                               epsilon,
                                                               head_dim,
                                                               num_head);
  } else if (src.dtype() == phi::DataType::BFLOAT16) {
    custom_kernel::FusedRmsQkvRopeTKernel<phi::dtype::bfloat16>(
        dev_ctx,
        src,
        residual,
        ln_scales,
        qkv_weights,
        qkv_biases,
        rotary_embs,
        query_states,
        key_value_states,
        epsilon,
        head_dim,
        num_head);
  } else {
    throw std::runtime_error(
        "Unsupported data type for FusedRmsQkvRopeTKernel");
  }
}

std::vector<paddle::Tensor> FusedRmsQkvRopeTImpl(
    const paddle::Tensor& src,
    const paddle::Tensor& ln_scales,
    const paddle::Tensor& qkv_weights,
    const paddle::optional<paddle::Tensor>& qkv_biases,
    const paddle::Tensor& rotary_embs,
    const paddle::Tensor& residual,
    float epsilon,
    int head_dim,
    int num_head) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(src.place()));
  auto src_tensor = static_cast<const phi::DenseTensor*>(src.impl().get());
  auto ln_scales_tensor =
      static_cast<const phi::DenseTensor*>(ln_scales.impl().get());
  auto qkv_weights_tensor =
      static_cast<const phi::DenseTensor*>(qkv_weights.impl().get());
  auto rotary_embs_tensor =
      static_cast<const phi::DenseTensor*>(rotary_embs.impl().get());
  auto residual_tensor =
      static_cast<const phi::DenseTensor*>(residual.impl().get());

  auto qkv_biases_tensor = paddle::optional<phi::DenseTensor>();
  if (qkv_biases) {
    auto qkv_biases_dt =
        static_cast<phi::DenseTensor*>(qkv_biases->impl().get());
    qkv_biases_tensor = paddle::optional<phi::DenseTensor>(*qkv_biases_dt);
  }

  // allocate memory on device.
  int64_t bsz = src.dims()[0];
  int64_t seq_len = src.dims()[1];
  int64_t fused_hidden_size = qkv_weights.dims()[0];
  int kv_num_head = (fused_hidden_size - num_head * head_dim) / head_dim / 2;

  std::shared_ptr<phi::DenseTensor> query_states =
      std::make_shared<phi::DenseTensor>();
  query_states->Resize(phi::make_ddim({bsz, seq_len, num_head, head_dim}));
  dev_ctx->Alloc(query_states.get(), src_tensor->dtype());

  std::shared_ptr<phi::DenseTensor> key_value_states =
      std::make_shared<phi::DenseTensor>();
  key_value_states->Resize(
      phi::make_ddim({2, bsz, seq_len, kv_num_head, head_dim}));
  dev_ctx->Alloc(key_value_states.get(), src_tensor->dtype());

  CallFusedRmsQkvRopeTKernel(*dev_ctx,
                             *src_tensor,
                             *residual_tensor,
                             *ln_scales_tensor,
                             *qkv_weights_tensor,
                             qkv_biases_tensor,
                             *rotary_embs_tensor,
                             query_states.get(),
                             key_value_states.get(),
                             phi::Scalar(epsilon),
                             phi::Scalar(head_dim),
                             phi::Scalar(num_head));
  return {paddle::Tensor(query_states), paddle::Tensor(key_value_states)};
}

std::vector<std::vector<int64_t>> FusedRmsQkvRopeTShape(
    const std::vector<int64_t>& src_shape,
    const std::vector<int64_t>& ln_scales_shape,
    const std::vector<int64_t>& qkv_weights_shape,
    const paddle::optional<std::vector<int64_t>>& qkv_biases_shape,
    const std::vector<int64_t>& rotary_embs_shape,
    const std::vector<int64_t>& residual_shape,
    float epsilon,
    int head_dim,
    int num_head) {
  int64_t bsz = src_shape[0];
  int64_t seq_len = src_shape[1];
  int64_t fused_hidden_size = qkv_weights_shape[0];
  int kv_num_head = (fused_hidden_size - num_head * head_dim) / head_dim / 2;
  return {{bsz, seq_len, num_head, head_dim},
          {2, bsz, seq_len, kv_num_head, head_dim}};
}

std::vector<paddle::DataType> FusedRmsQkvRopeTDtype(
    const paddle::DataType& src_dtype,
    const paddle::DataType& ln_scales_dtype,
    const paddle::DataType& qkv_weights_dtype,
    const paddle::optional<paddle::DataType>& qkv_biases_dtype,
    const paddle::DataType& rotary_embs_dtype,
    const paddle::DataType& residual_dtype) {
  return {src_dtype, src_dtype};
}

PD_BUILD_OP(fused_rms_qkv_rope_t)
    .Inputs({"src",
             "ln_scales",
             "qkv_weights",
             paddle::Optional("qkv_biases"),
             "rotary_embs",
             "residual"})
    .Outputs({"query_states", "key_value_states"})
    .Attrs({"epsilon: float", "head_dim: int", "num_head: int"})
    .SetKernelFn(PD_KERNEL(FusedRmsQkvRopeTImpl))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedRmsQkvRopeTShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedRmsQkvRopeTDtype));
