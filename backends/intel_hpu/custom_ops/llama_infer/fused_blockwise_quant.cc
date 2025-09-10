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

#include "paddle/extension.h"
#include "utils/utils.h"

#define BLOCKSIZE 128

std::vector<paddle::Tensor> FusedBlockwiseQuantForward(
    const paddle::Tensor& x) {
  assert(x.dims().size() == 2);
  int m = x.dims()[0];
  int n = x.dims()[1];

  auto x_padded = paddle::concat(
      {x,
       paddle::zeros(
           {(BLOCKSIZE - m % BLOCKSIZE) % BLOCKSIZE, n}, x.dtype(), x.place())},
      0);
  x_padded = paddle::concat(
      {x_padded,
       paddle::zeros(
           {x_padded.dims()[0], (BLOCKSIZE - n % BLOCKSIZE) % BLOCKSIZE},
           x.dtype(),
           x.place())},
      1);

  auto x_view = paddle::reshape(x_padded,
                                {x_padded.dims()[0] / BLOCKSIZE,
                                 BLOCKSIZE,
                                 x_padded.dims()[1] / BLOCKSIZE,
                                 BLOCKSIZE});
  auto x_abs = paddle::abs(x_view).cast(paddle::DataType::FLOAT32);
  auto x_amax = paddle::experimental::amax(x_abs, {1, 3}, true);
  auto scale = paddle::experimental::view_shape(
      x_amax * (1.0 / 240.0), {x_view.dims()[0], x_view.dims()[2]});
  auto x_scaled =
      (paddle::divide(x_view.cast(paddle::DataType::FLOAT32), x_amax) * 240.0)
          .cast(paddle::DataType::FLOAT8_E4M3FN);
  x_scaled = paddle::experimental::slice(
      paddle::reshape(x_scaled, {x_padded.dims()[0], x_padded.dims()[1]}),
      {0, 1},
      {0, 0},
      {m, n},
      {},
      {});

  return {x_scaled.contiguous(), scale};
}

std::vector<std::vector<int64_t>> FusedBlockwiseQuantShape(
    const std::vector<int64_t>& x_shape) {
  return {x_shape};
}

std::vector<paddle::DataType> FusedBlockwiseQuantDtype(
    const paddle::DataType& x_dtype) {
  return {paddle::DataType::FLOAT8_E4M3FN};
}

PD_BUILD_OP(fused_blockwise_quant)
    .Inputs({"x"})
    .Outputs({"out", "scale"})
    .SetKernelFn(PD_KERNEL(FusedBlockwiseQuantForward))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedBlockwiseQuantShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedBlockwiseQuantDtype));
