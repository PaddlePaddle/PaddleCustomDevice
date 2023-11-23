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

#include "kernels/common_ops/common_ops.h"
#include "kernels/funcs/gcu_kernel_funcs.h"
#include "kernels/funcs/gcu_op_runner.h"

namespace custom_kernel {
template <typename T, typename Context>
void SplitKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const phi::IntArray& num_or_sections,
                 const phi::Scalar& axis_scalar,
                 std::vector<phi::DenseTensor*> outs) {
  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "split_with_num", split_with_num);
    for (size_t i = 0; i < outs.size(); ++i) {
      if (outs[i]->initialized()) {
        auto gcu_memory = GetGcuMemory(*outs[i], false);
        std::vector<int64_t> dims = gcu_memory->dims;
        std::vector<int64_t> tensor_dims = phi::vectorize(outs[i]->dims());
        if (tensor_dims.empty()) {
          tensor_dims = {1};
        }
        if (dims != tensor_dims) {
          auto tmp = EmptyTensor(dev_ctx, outs[i]->meta());
          dev_ctx.template Alloc(&tmp, tmp.dtype());
          *outs[i] = tmp;
        }
      } else {
        dev_ctx.template Alloc<T>(outs[i]);
      }
    }
    split(
        dev_ctx, x, axis_scalar.to<int>(), 0, num_or_sections.GetData(), outs);
    PADDLE_GCU_KERNEL_END("split_with_num", split_with_num);
  } else {
    TensorNameMap input_names;
    input_names["X"] = {"x"};
    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};

    TensorNameMap output_names;
    TensorValueMap outputs;
    std::vector<std::string> names;
    names.reserve(outs.size());
    std::vector<phi::DenseTensor*> values;
    values.reserve(outs.size());
    for (size_t i = 0; i < outs.size(); ++i) {
      dev_ctx.template Alloc<T>(outs[i]);
      names.emplace_back(std::string("out") + std::to_string(i));

      if (UseScatterMemory()) {
        phi::DenseTensor* tmp_tensor = new phi::DenseTensor();
        tmp_tensor->Resize(outs[i]->dims());
        dev_ctx.template Alloc(tmp_tensor, outs[i]->dtype());
        values.emplace_back(tmp_tensor);
      } else {
        values.emplace_back(outs[i]);
      }
    }
    output_names["Out"] = names;
    outputs["Out"] = values;

    GcuAttributeMap attrs;
    attrs["axis"] = axis_scalar.to<int>();

    std::vector<int> sections = GetIntList(num_or_sections.GetData());
    attrs["sections"] = sections;
    attrs["num"] = 0;

    GcuRunner(
        input_names, inputs, output_names, outputs, attrs, "split", dev_ctx);

    if (UseScatterMemory()) {
      for (size_t i = 0; i < outs.size(); ++i) {
        *(outs[i]) = *(values[i]);
        delete values[i];
      }
    }
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
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::SplitKernel,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(split_with_num,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::SplitWithNumKernel,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16) {}
