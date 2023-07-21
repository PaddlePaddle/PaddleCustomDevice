// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

// Copyright©2020-2023 Shanghai Biren Technology Co., Ltd. All rights reserved.

#include "glog/logging.h"
#include "kernels/funcs/br_paddle_supa.h"
#include "paddle/phi/extension.h"

namespace supa {
template <typename T, typename Context>
void HardSigmoidKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       float slope,
                       float offset,
                       phi::DenseTensor* out) {
  br_device::OP_PARAMS(HardSigmoid) param;
  param.slope_ = slope;
  param.offset_ = offset;
  br_device::SupaOpRunner<T, Context> runner(dev_ctx, param, {&x}, {out});
  runner.Run();
}

template <typename T, typename Context>
void HardSigmoidGradKernel(const Context& dev_ctx,
                           const phi::DenseTensor& out,
                           const phi::DenseTensor& dout,
                           float slope,
                           float offset,
                           phi::DenseTensor* dx) {
  br_device::OP_PARAMS(HardSigmoidGrad) param;
  param.slope_ = slope;
  param.offset_ = offset;
  br_device::SupaOpRunner<T, Context> runner(
      dev_ctx, param, {&out, &dout}, {dx});
  runner.Run();
}
}  // namespace supa

PD_REGISTER_PLUGIN_KERNEL(
    hard_sigmoid, SUPA, ALL_LAYOUT, supa::HardSigmoidKernel, float) {}

PD_REGISTER_PLUGIN_KERNEL(
    hard_sigmoid_grad, SUPA, ALL_LAYOUT, supa::HardSigmoidGradKernel, float) {}
