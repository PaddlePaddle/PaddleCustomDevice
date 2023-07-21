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
void ElementwisePowKernel(const Context& dev_ctx,
                          const phi::DenseTensor& x,
                          const phi::DenseTensor& y,
                          phi::DenseTensor* out) {
  VLOG(1) << "Enter ElementwisePowKernel.";

  br_device::SupaOpRunner<T, Context> runner(
      dev_ctx, br_device::OP_PARAMS(ElementwisePow)(), {&x, &y}, {out});
  runner.Run();

  VLOG(1) << "Leave ElementwisePowKernel.";
}

}  // namespace supa

PD_REGISTER_PLUGIN_KERNEL(
    elementwise_pow, SUPA, ALL_LAYOUT, supa::ElementwisePowKernel, float) {}
