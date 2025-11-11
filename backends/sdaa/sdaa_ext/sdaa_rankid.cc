// BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
// reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
// OF SUCH DAMAGE.

#include <sdaa_runtime.h>

#include <vector>

#include "kernels/funcs/sdaa_baseop.h"
#include "kernels/funcs/sdaa_funcs.h"
#include "paddle/extension.h"

std::vector<std::vector<int64_t>> RankIdInferShape(
    const std::vector<int64_t>& x_shape) {
  return {{-1}};
}

std::vector<paddle::DataType> RankIdInferDtype(
    const paddle::DataType& x_dtype) {
  return {paddle::DataType::INT32};
}

std::vector<paddle::Tensor> CustomRankIds(const paddle::Tensor& x) {
  std::vector<int32_t> ranks;
  int32_t value = 0;
  int dev_count = 0;
  SDAACHECK(sdaaGetDeviceCount(&dev_count));
  PADDLE_ENFORCE_GE(
      dev_count,
      0,
      phi::errors::InvalidArgument("there is no device on the platform"));
  for (int i = 0; i < dev_count; i++) {
    SDAACHECK(sdaaDeviceGetAttribute(&value, sdaaDevAttrPhyCardId, i));
    ranks.push_back(value);
  }
  auto out = paddle::zeros({static_cast<int64_t>(ranks.size())},
                           paddle::DataType::INT32,
                           phi::CPUPlace());
  auto* out_data = out.data<int32_t>();
  for (int i = 0; i < ranks.size(); ++i) {
    out_data[i] = ranks[i];
  }
  return {out};
}

PD_BUILD_OP(rank_ids)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(CustomRankIds))
    .SetInferShapeFn(PD_INFER_SHAPE(RankIdInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(RankIdInferDtype));
