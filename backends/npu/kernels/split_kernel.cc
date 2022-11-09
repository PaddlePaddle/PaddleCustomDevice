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
void SplitKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const phi::IntArray& num_or_sections,
                 const phi::Scalar& axis_scalar,
                 std::vector<phi::DenseTensor*> outs) {
  // need to infershape output
  if (num_or_sections.FromTensor() || axis_scalar.FromTensor()) {
    std::vector<phi::MetaTensor> out_metas;
    out_metas.reserve(outs.size());
    std::vector<phi::MetaTensor*> out_metas_ptr;
    for (size_t i = 0; i < outs.size(); ++i) {
      out_metas.push_back(outs[i]);
      out_metas_ptr.push_back(&out_metas.back());
    }

    phi::SplitInferMeta(x, num_or_sections, axis_scalar, out_metas_ptr);

    for (size_t i = 0; i < out_metas.size(); ++i) {
      outs[i]->Resize(out_metas[i].dims());
    }
  }

  int axis = axis_scalar.to<int>();

  std::vector<phi::DenseTensor> outputs;
  for (size_t j = 0; j < outs.size(); ++j) {
    dev_ctx.template Alloc<T>(outs[j]);
    outputs.push_back(*outs[j]);
  }

  auto sections = num_or_sections.GetData();

  if (sections.size() == 1) {
    NpuOpRunner runner;
    runner.SetType("SplitD").AddInput(x).AddOutputs(outputs).AddAttrs(
        {{"split_dim", axis},
         {"num_split", static_cast<int32_t>(sections[0])}});
    auto stream = dev_ctx.stream();
    runner.Run(stream);
  } else {
    NpuOpRunner runner;
    runner.SetType("SplitVD").AddInput(x).AddOutputs(outputs).AddAttrs(
        {{"size_splits", sections},
         {"split_dim", axis},
         {"num_split", static_cast<int32_t>(sections.size())}});
    auto stream = dev_ctx.stream();
    runner.Run(stream);
  }
}

template <typename T, typename Context>
void SplitWithNumKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        int num,
                        const phi::Scalar& axis_scalar,
                        std::vector<phi::DenseTensor*> outs) {
  int axis_value = axis_scalar.to<int>();
  auto input_axis_dim = x.dims().at(axis_value);
  std::vector<int64_t> sections_vec;
  for (int i = 0; i < num; ++i) {
    sections_vec.push_back(input_axis_dim / num);
  }
  phi::IntArray sections(sections_vec);
  custom_kernel::SplitKernel<T, Context>(
      dev_ctx, x, sections, axis_scalar, outs);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(split,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SplitKernel,
                          float,
                          double,
                          int64_t,
                          int,
                          bool,
                          uint8_t,
                          int8_t) {}

PD_REGISTER_PLUGIN_KERNEL(split_with_num,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SplitWithNumKernel,
                          float,
                          double,
                          int64_t,
                          int,
                          bool,
                          uint8_t,
                          int8_t) {}
