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

/// If a variable is a empty variable, that name will be used.
constexpr char kEmptyVarName[] = "@EMPTY@";

static inline int64_t ComputeAxis(int64_t axis, int64_t rank) {
  PADDLE_ENFORCE_EQ(
      axis >= -rank && axis < rank,
      true,
      phi::errors::InvalidArgument(
          "The axis is expected to be in range of [%d, %d), but got %d",
          -rank,
          rank,
          axis));
  if (axis < 0) {
    axis = axis + rank;
  }
  return axis > 0 ? axis : 0;
}

template <typename T, typename Context>
void ConcatKernel(const Context& dev_ctx,
                  const std::vector<const phi::DenseTensor*>& ins,
                  const phi::Scalar& axis_scalar,
                  phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  int axis = axis_scalar.to<int>();
  axis = ComputeAxis(static_cast<int64_t>(axis),
                     static_cast<int64_t>(ins[0]->dims().size()));

  std::vector<phi::DenseTensor> inputs;
  std::vector<std::string> names;
  for (size_t i = 0; i < ins.size(); ++i) {
    if (ins[i] && ins[i]->numel() > 0) {
      inputs.push_back(*ins[i]);
      names.push_back("x" + std::to_string(i));
    } else {
      continue;
    }
  }

  auto stream = dev_ctx.stream();
  NpuOpRunner runner{
      "ConcatD",
      {inputs},
      {*out},
      {{"concat_dim", axis}, {"N", static_cast<int>(inputs.size())}}};
  runner.AddInputNames(names);
  runner.Run(stream);
}

template <typename T, typename Context>
void ConcatGradKernel(const Context& dev_ctx,
                      const std::vector<const phi::DenseTensor*>& ins,
                      const phi::DenseTensor& dout,
                      const phi::Scalar& axis_scalar,
                      std::vector<phi::DenseTensor*> outs) {
  auto stream = dev_ctx.stream();

  int axis = axis_scalar.to<int>();
  axis = ComputeAxis(static_cast<int64_t>(axis),
                     static_cast<int64_t>(ins[0]->dims().size()));

  int offset = 0;
  for (size_t j = 0; j < outs.size(); ++j) {
    if (outs[j] && outs[j]->numel() != 0UL) {
      dev_ctx.template Alloc<T>(outs[j]);

      std::vector<int> offsets;
      std::vector<int> sizes;
      for (int dim = 0; dim < ins[j]->dims().size(); ++dim) {
        if (dim == axis) {
          offsets.push_back(offset);
          sizes.push_back(ins[j]->dims()[dim]);
        } else {
          offsets.push_back(0);
          sizes.push_back(ins[j]->dims()[dim]);
        }
      }
      const auto& runner = NpuOpRunner("SliceD",
                                       {dout},
                                       {*outs[j]},
                                       {{"offsets", offsets}, {"size", sizes}});
      runner.Run(stream);
    }
    if (ins[j]->numel() != 0UL) {
      offset += ins[j]->dims()[axis];
    }
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(concat,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::ConcatKernel,
                          int,
                          int64_t,
                          float,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(concat_grad,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::ConcatGradKernel,
                          int,
                          int64_t,
                          float,
                          double) {}
