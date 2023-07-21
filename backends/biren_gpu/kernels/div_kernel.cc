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

// Div forward kernel
template <typename T, typename Context>
void DivKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::DenseTensor& y,
               phi::DenseTensor* out) {
  VLOG(4) << "Enter DivKernel";
  br_device::SupaOpRunner<T, Context> runner(
      dev_ctx, br_device::OP_PARAMS(Div)(), {&x, &y}, {out});
  runner.Run();
  VLOG(4) << "Leave DivKernel";
}

template <typename T, typename Context>
void DivGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& y,
                   const phi::DenseTensor& out,
                   const phi::DenseTensor& dout,
                   int axis,
                   phi::DenseTensor* dx,
                   phi::DenseTensor* dy) {
  VLOG(4) << "Enter DivGradKernel";
  br_device::SupaOpRunner<T, Context> runner(
      dev_ctx, br_device::OP_PARAMS(DivGrad)(), {&x, &y, &dout}, {dx, dy});
  runner.Run();
  VLOG(4) << "Leave DivGradKernel";
}

}  // namespace supa

PD_REGISTER_PLUGIN_KERNEL(divide, SUPA, ALL_LAYOUT, supa::DivKernel, float) {}

PD_REGISTER_PLUGIN_KERNEL(
    divide_grad, SUPA, ALL_LAYOUT, supa::DivGradKernel, float) {}
