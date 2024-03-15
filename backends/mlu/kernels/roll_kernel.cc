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

#include "kernels/funcs/mlu_baseop.h"
#include "kernels/funcs/mlu_funcs.h"

namespace custom_kernel {

template <typename T, typename Context>
void RollKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::IntArray& shifts,
                const std::vector<int64_t>& axis,
                phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  std::vector<int> shifts_data(shifts.GetData().begin(),
                               shifts.GetData().end());
  std::vector<int> axis_int32(axis.begin(), axis.end());

  MLUCnnlTensorDesc input_desc(x);
  MLUCnnlTensorDesc output_desc(*out);
  MLUCnnl::Roll(dev_ctx,
                input_desc.get(),
                GetBasePtr(&x),
                shifts_data.data(),
                shifts_data.size(),
                axis_int32.data(),
                axis_int32.size(),
                output_desc.get(),
                GetBasePtr(out));
}

template <typename T, typename Context>
void RollGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x UNUSED,
                    const phi::DenseTensor& out_grad,
                    const phi::IntArray& shifts,
                    const std::vector<int64_t>& axis,
                    phi::DenseTensor* x_grad) {
  std::vector<int> shifts_data(shifts.GetData().begin(),
                               shifts.GetData().end());
  std::vector<int> axis_int32(axis.begin(), axis.end());
  dev_ctx.template Alloc<T>(x_grad);

  for (int i = 0; i < shifts_data.size(); ++i) {
    shifts_data[i] = 0 - shifts_data[i];
  }

  MLUCnnlTensorDesc out_grad_desc(out_grad);
  MLUCnnlTensorDesc x_grad_desc(*x_grad);
  MLUCnnl::Roll(dev_ctx,
                out_grad_desc.get(),
                GetBasePtr(&out_grad),
                shifts_data.data(),
                shifts_data.size(),
                axis_int32.data(),
                axis_int32.size(),
                x_grad_desc.get(),
                GetBasePtr(x_grad));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(roll,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::RollKernel,
                          float,
                          phi::dtype::float16,
                          int) {}

PD_REGISTER_PLUGIN_KERNEL(roll_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::RollGradKernel,
                          float,
                          phi::dtype::float16,
                          int) {}
