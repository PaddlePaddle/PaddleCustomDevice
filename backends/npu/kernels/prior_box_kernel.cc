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
void PriorBoxKernel(const Context& dev_ctx,
                    const phi::DenseTensor& input,
                    const phi::DenseTensor& image,
                    const std::vector<float>& min_sizes,
                    const std::vector<float>& max_sizes,
                    const std::vector<float>& aspect_ratios,
                    const std::vector<float>& variances,
                    bool flip,
                    bool clip,
                    float step_w,
                    float step_h,
                    float offset,
                    bool min_max_aspect_ratios_order,
                    phi::DenseTensor* out,
                    phi::DenseTensor* var) {
  PADDLE_ENFORCE_EQ(out->dims(),
                    var->dims(),
                    phi::errors::Unimplemented(
                        "the shape of boxes and variances must be same in "
                        "the npu kernel of prior_box, but got boxes->dims() "
                        "= [%s], variances->dims() = [%s]",
                        out->dims(),
                        var->dims()));

  phi::DenseTensor out_t;
  auto out_dims = phi::vectorize(out->dims());
  out_dims.insert(out_dims.begin(), 2);
  phi::DenseTensorMeta out_t_meta = {input.dtype(), phi::make_ddim(out_dims)};
  out_t.set_meta(out_t_meta);
  dev_ctx.template Alloc<T>(&out_t);

  NPUAttributeMap attr_input = {{"min_size", min_sizes},
                                {"max_size", max_sizes},
                                {"aspect_ratio", aspect_ratios},
                                {"step_h", step_h},
                                {"step_w", step_w},
                                {"flip", flip},
                                {"clip", clip},
                                {"offset", offset},
                                {"variance", variances}};

  auto stream = dev_ctx.stream();

  const auto& runner =
      NpuOpRunner("PriorBox", {input, image}, {out_t}, attr_input);
  runner.Run(stream);

  out_t.Resize(phi::make_ddim({out_t.numel()}));

  dev_ctx.template Alloc<T>(out);
  dev_ctx.template Alloc<T>(var);

  auto out_dim = out->dims();
  auto var_dim = var->dims();

  out->Resize(phi::make_ddim({out->numel()}));
  var->Resize(phi::make_ddim({var->numel()}));

  std::vector<int64_t> offset1(1, 0);
  std::vector<int64_t> size1(1, out->numel());
  std::vector<int64_t> offset2(1, out->numel());
  std::vector<int64_t> size2(1, var->numel());

  NpuOpRunner runner1;
  runner1.SetType("Slice")
      .AddInput(out_t)
      .AddInput(dev_ctx, std::move(offset1))
      .AddInput(dev_ctx, std::move(size1))
      .AddOutput(*out)
      .Run(stream);
  NpuOpRunner runner2;
  runner2.SetType("Slice")
      .AddInput(out_t)
      .AddInput(dev_ctx, std::move(offset2))
      .AddInput(dev_ctx, std::move(size2))
      .AddOutput(*var)
      .Run(stream);

  out->Resize(out_dim);
  var->Resize(var_dim);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(prior_box,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::PriorBoxKernel,
                          float,
                          phi::dtype::float16) {}
