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

#ifdef PADDLE_WITH_ATB

#include "atb_layers/fused_lm_head_layer.h"
#include "fused_blha_layer_op_utils.h"  // NOLINT

std::vector<std::vector<int64_t>> LMHeadInferShape(
    const std::vector<int64_t>& x,
    const std::vector<int64_t>& norm_weight,
    const std::vector<int64_t>& linear_weight,
    float epsilon,
    bool trans_weight,
    int64_t rank,
    int64_t nranks,
    int64_t root,
    int ring_id) {
  std::vector<int64_t> out_dims;
  out_dims.push_back(x[0]);
  if (trans_weight) {
    out_dims.push_back(linear_weight[0] * nranks);
  } else {
    out_dims.push_back(linear_weight[1] * nranks);
  }
  return {out_dims};
}

std::vector<paddle::DataType> LMHeadInferDType(
    const paddle::DataType& x,
    const paddle::DataType& norm_weight,
    const paddle::DataType& linear_weight,
    float epsilon,
    bool trans_weight,
    int64_t rank,
    int64_t nranks,
    int64_t root,
    int ring_id) {
  PADDLE_ENFORCE_EQ(x,
                    phi::DataType::FLOAT16,
                    phi::errors::InvalidArgument("only support float16."));
  return {x};
}

std::vector<paddle::Tensor> LMHeadOp(
    const paddle::Tensor& x,  // [bs, max_seqlen, emb_dim]
    const paddle::Tensor& norm_weight,
    const paddle::Tensor& linear_weight,
    float epsilon,
    bool trans_weight,
    int64_t rank,
    int64_t nranks,
    int64_t root,
    int ring_id) {
  auto place = x.place();
  const auto& dev_ctx = *static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(place));

  auto out_dtype = x.dtype();
  auto x_shape = x.shape();
  auto w_shape = linear_weight.shape();
  std::vector<int64_t> out_shape;
  out_shape.push_back(x_shape[0]);
  if (trans_weight) {
    out_shape.push_back(w_shape[0] * nranks);
  } else {
    out_shape.push_back(w_shape[1] * nranks);
  }

  paddle::Tensor out(place);
  init_tensor(dev_ctx, out_dtype, out_shape, &out);

  atb_layers::FusedLmHeadLayerParam param;
  param.epsilon = epsilon;
  param.trans_weight = trans_weight;
  param.rank = rank;
  param.nranks = nranks;
  param.root = root;
  param.comm = nullptr;

  atb_layers::OperationRunner runner;
  runner.create(param);
  runner.bind_input(x);
  runner.bind_input(norm_weight);
  runner.bind_input(linear_weight);
  runner.bind_output(&out);
  runner.run(dev_ctx);
  return {out};
}

PD_BUILD_OP(lm_head)
    .Inputs({"x", "norm_weight", "linear_weight"})
    .Outputs({"out"})
    .Attrs({
        "epsilon: float",
        "trans_weight: bool",
        "rank: int64_t",
        "nranks: int64_t",
        "root: int64_t",
        "ring_id: int",
    })
    .SetKernelFn(PD_KERNEL(LMHeadOp))
    .SetInferShapeFn(PD_INFER_SHAPE(LMHeadInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(LMHeadInferDType));

#endif
