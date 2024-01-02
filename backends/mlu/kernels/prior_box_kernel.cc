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

#include "kernels/funcs/mlu_baseop.h"
#include "kernels/funcs/mlu_funcs.h"

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
  int im_width = image.dims()[3];
  int im_height = image.dims()[2];

  int width = input.dims()[3];
  int height = input.dims()[2];

  std::vector<float> new_aspect_ratios;
  phi::ExpandAspectRatios(aspect_ratios, flip, &new_aspect_ratios);
  phi::DenseTensor ratios;
  TensorFromVector(dev_ctx, new_aspect_ratios, dev_ctx, &ratios);
  dev_ctx.Wait();
  MLUOpTensorDesc new_aspect_ratios_desc(ratios);

  phi::DenseTensor min;
  TensorFromVector(dev_ctx, min_sizes, dev_ctx, &min);
  dev_ctx.Wait();
  MLUOpTensorDesc min_sizes_desc(min);

  phi::DenseTensor max;
  TensorFromVector(dev_ctx, max_sizes, dev_ctx, &max);
  dev_ctx.Wait();
  MLUOpTensorDesc max_sizes_desc(max);

  phi::DenseTensor var_tensor;
  TensorFromVector(dev_ctx, variances, dev_ctx, &var_tensor);
  dev_ctx.Wait();
  MLUOpTensorDesc variances_attr_desc(var_tensor);

  dev_ctx.template Alloc<T>(out);
  dev_ctx.template Alloc<T>(var);
  MLUOpTensorDesc var_desc(*var);
  MLUOpTensorDesc output_desc(*out);

  MLUOP::OpPriorBox(dev_ctx,
                    min_sizes_desc.get(),
                    GetBasePtr(&min),
                    new_aspect_ratios_desc.get(),
                    GetBasePtr(&ratios),
                    variances_attr_desc.get(),
                    GetBasePtr(&var_tensor),
                    max_sizes_desc.get(),
                    GetBasePtr(&max),
                    height,
                    width,
                    im_height,
                    im_width,
                    step_h,
                    step_w,
                    offset,
                    clip,
                    min_max_aspect_ratios_order,
                    output_desc.get(),
                    GetBasePtr(out),
                    var_desc.get(),
                    GetBasePtr(var));
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    prior_box, mlu, ALL_LAYOUT, custom_kernel::PriorBoxKernel, float) {}
