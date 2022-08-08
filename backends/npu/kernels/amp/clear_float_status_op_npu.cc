// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

namespace custom_kernel {
template <typename T, typename Context>
void ClearFloatStatus(const Context& dev_ctx,
                      const phi::DenseTensor& t_float_status,
                      phi::DenseTensor* float_status_out) {
  const auto* float_status = &t_float_status;
  dev_ctx.template Alloc<T>(float_status_out);

  // NOTE(zhiqiu): NPUClearFloatStatus modifies the input.
  PADDLE_ENFORCE_EQ(float_status_out,
                    float_status,
                    phi::errors::PreconditionNotMet(
                        "The input(FloatStatus) and Output(FloatStatusOut) "
                        "should be the same."));
  phi::DenseTensor tmp;
  tmp.Resize({8});
  dev_ctx.template Alloc<float>(&tmp);
  const auto& runner =
      NpuOpRunner("NPUClearFloatStatus", {tmp}, {*float_status_out});
  auto stream = dev_ctx.stream();
  runner.Run(stream);
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(clear_float_status,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::ClearFloatStatus,
                          float,
                          double) {}
