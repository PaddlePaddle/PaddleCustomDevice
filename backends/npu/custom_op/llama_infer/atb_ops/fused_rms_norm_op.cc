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

#include "atb_layers/fused_rms_norm.h"
#include "fused_blha_layer_op_utils.h"  // NOLINT

std::vector<std::vector<int64_t>> RmsNormShape(
    const std::vector<int64_t>& x,
    const std::vector<int64_t>& norm_weight,
    const paddle::optional<std::vector<int64_t>>& residual,
    float epsilon) {
  std::vector<int64_t> out_dims = x;

  return {out_dims};
}

static atb_layers::OperationRunner g_RmsNormRunner;

std::vector<paddle::Tensor> RmsNormOp(
    const paddle::Tensor& x,
    const paddle::Tensor& norm_weight,
    const paddle::optional<paddle::Tensor>& residual,
    float epsilon) {
  auto place = x.place();
  const auto& dev_ctx = *static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(place));

  auto out_dtype = x.dtype();
  std::vector<int64_t> out_shape = x.shape();

  paddle::Tensor out(place);
  init_tensor(dev_ctx, out_dtype, out_shape, &out);
  paddle::Tensor residual_out(place);
  init_tensor(dev_ctx, out_dtype, out_shape, &residual_out);

  if (g_RmsNormRunner.is_initialized()) {
    g_RmsNormRunner.reset_variant_pack();
  }

  atb_layers::RmsNormParam param;
  param.epsilon = epsilon;
  param.has_residual = residual.is_initialized();
  g_RmsNormRunner.create(param);

  g_RmsNormRunner.bind_input(x);
  g_RmsNormRunner.bind_input(norm_weight);
  if (residual.is_initialized()) {
    g_RmsNormRunner.bind_input(residual.get());
  }

  g_RmsNormRunner.bind_output(&out);
  if (residual.is_initialized()) {
    g_RmsNormRunner.bind_output(&residual_out);
  }
  g_RmsNormRunner.run(dev_ctx);
  return {out, residual_out};
}

PD_BUILD_OP(atb_rms_norm)  // atb_flash_attention   rms_norm
    .Inputs({"x", "norm_weight", "residual@OPTIONAL"})  // tensor
    .Outputs({"out", "residual_out"})                   // tensor
    .Attrs({
        "epsilon: float",  // int/float/bool
    })
    .SetKernelFn(PD_KERNEL(RmsNormOp))               // 适配
    .SetInferShapeFn(PD_INFER_SHAPE(RmsNormShape));  // shape校验
// .SetInferDtypeFn(PD_INFER_DTYPE(RmsNormDType));  // type校验

#endif
