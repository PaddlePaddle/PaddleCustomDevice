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

// Abs forward kernel
template <typename T, typename Context>
void AbsKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  VLOG(1) << "Enter AbsKernel.";
  br_device::SupaOpRunner<T, Context> runner(
      dev_ctx, br_device::OP_PARAMS(Abs)(), {&x}, {out});
  runner.Run();
  VLOG(1) << "Leave AbsKernel.";
}

// AbsGrad kernel
template <typename T, typename Context>
void AbsGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& dout,
                   phi::DenseTensor* dx) {
  VLOG(1) << "Enter AbsGrad.";
  br_device::SupaOpRunner<T, Context> runner(
      dev_ctx, br_device::OP_PARAMS(AbsGrad)(), {&x, &dout}, {dx});
  runner.Run();
  VLOG(1) << "Leave AbsGrad.";
}

}  // namespace supa

PD_REGISTER_PLUGIN_KERNEL(abs, SUPA, ALL_LAYOUT, supa::AbsKernel, float) {}

PD_REGISTER_PLUGIN_KERNEL(
    abs_grad, SUPA, ALL_LAYOUT, supa::AbsGradKernel, float) {}
