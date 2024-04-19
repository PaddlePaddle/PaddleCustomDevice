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
void AclopSplitKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::IntArray& num_or_sections,
                      const phi::Scalar& axis_scalar,
                      std::vector<phi::DenseTensor*> outs) {
  // need to infershape output
  auto sections = num_or_sections.GetData();
  int axis = axis_scalar.to<int>();

  if (!num_or_sections.FromTensor() && !axis_scalar.FromTensor() &&
      // when the outs.size() does not match to the sections[0],
      // the ascend op "Split" will fail. So we change this situation
      // to SplitWithNum to resize outs.
      sections.size() == 1 && outs.size() == sections[0]) {
    std::vector<phi::DenseTensor> outputs;
    for (size_t j = 0; j < outs.size(); ++j) {
      dev_ctx.template Alloc<T>(outs[j]);
      outputs.push_back(*outs[j]);
    }
    NpuOpRunner runner;
    runner.SetType("Split")
        .AddInput(dev_ctx, std::vector<int32_t>({axis}))
        .AddInput(x)
        .AddOutputs(outputs)
        .AddAttrs({{"num_split", static_cast<int32_t>(sections[0])}});
    auto stream = dev_ctx.stream();
    runner.Run(stream);
  } else {
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

    std::vector<phi::DenseTensor> outputs;
    for (size_t j = 0; j < outs.size(); ++j) {
      dev_ctx.template Alloc<T>(outs[j]);
      outputs.push_back(*outs[j]);
    }
    NpuOpRunner runner;
    runner.SetType("SplitV")
        .AddInput(x)
        .AddInput(dev_ctx, std::move(sections))
        .AddInput(dev_ctx, std::vector<int32_t>({axis}))
        .AddOutputs(outputs)
        .AddAttrs({{"num_split", static_cast<int32_t>(sections.size())}});
    auto stream = dev_ctx.stream();
    runner.Run(stream);
  }
}

template <typename T, typename Context>
void SplitKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const phi::IntArray& num_or_sections,
                 const phi::Scalar& axis_scalar,
                 std::vector<phi::DenseTensor*> outs) {
  // need to infershape output
  auto sections = num_or_sections.GetData();
  int64_t axis = axis_scalar.to<int64_t>();

  if (!num_or_sections.FromTensor() && !axis_scalar.FromTensor() &&
      // when the outs.size() does not match to the sections[0],
      // the ascend op "Split" will fail. So we change this situation
      // to SplitWithNum to resize outs.
      sections.size() == 1 && outs.size() == sections[0]) {
    DO_COMPATIBILITY(aclnnSplitTensor,
                     (custom_kernel::AclopSplitKernel<T, Context>(
                         dev_ctx, x, num_or_sections, axis_scalar, outs)));

    uint64_t splitSections = x.dims()[axis] / sections[0];
    std::vector<phi::DenseTensor*> outputs;
    for (size_t j = 0; j < outs.size(); ++j) {
      dev_ctx.template Alloc<T>(outs[j]);
      outputs.push_back(outs[j]);
    }
    EXEC_NPU_CMD(aclnnSplitTensor, dev_ctx, x, splitSections, axis, outputs);
  } else {
    DO_COMPATIBILITY(aclnnSplitWithSize,
                     (custom_kernel::AclopSplitKernel<T, Context>(
                         dev_ctx, x, num_or_sections, axis_scalar, outs)));

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

    std::vector<phi::DenseTensor*> outputs;
    for (size_t j = 0; j < outs.size(); ++j) {
      dev_ctx.template Alloc<T>(outs[j]);
      outputs.push_back(outs[j]);
    }
    static const auto aclCreateIntArray = GET_OP_API_FUNC(aclCreateIntArray);
    auto sections_acl = aclCreateIntArray(sections.data(), sections.size());
    EXEC_NPU_CMD(aclnnSplitWithSize, dev_ctx, x, sections_acl, axis, outputs);
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
                          int8_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

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
                          int8_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
