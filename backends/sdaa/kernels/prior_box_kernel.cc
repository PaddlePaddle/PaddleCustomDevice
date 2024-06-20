// BSD 3- Clause License Copyright (c) 2024, Tecorigin Co., Ltd. All rights
// reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
// OF SUCH DAMAGE4

#include <algorithm>

#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

inline void ExpandAspectRatios(const std::vector<float>& input_aspect_ratior,
                               bool flip,
                               std::vector<float>* output_aspect_ratior) {
  constexpr float epsilon = 1e-6;
  output_aspect_ratior->clear();
  output_aspect_ratior->push_back(1.0f);
  for (size_t i = 0; i < input_aspect_ratior.size(); ++i) {
    float ar = input_aspect_ratior[i];
    bool already_exist = false;
    for (size_t j = 0; j < output_aspect_ratior->size(); ++j) {
      if (fabs(ar - output_aspect_ratior->at(j)) < epsilon) {
        already_exist = true;
        break;
      }
    }
    if (!already_exist) {
      output_aspect_ratior->push_back(ar);
      if (flip) {
        output_aspect_ratior->push_back(1.0f / ar);
      }
    }
  }
}

template <typename T, typename Context>
void PriorBoxKernel(const Context& ctx,
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
  VLOG(4) << "Call SDAA PriorBoxKernel";
  std::vector<float> new_aspect_ratios;
  ExpandAspectRatios(aspect_ratios, flip, &new_aspect_ratios);

  T new_step_w = static_cast<T>(step_w);
  T new_step_h = static_cast<T>(step_h);
  T new_offset = static_cast<T>(offset);

  auto im_width = image.dims()[3];
  auto im_height = image.dims()[2];

  auto width = input.dims()[3];
  auto height = input.dims()[2];

  T step_width, step_height;
  if (new_step_w == 0 || new_step_h == 0) {
    step_width = static_cast<T>(im_width) / width;
    step_height = static_cast<T>(im_height) / height;
  } else {
    step_width = new_step_w;
    step_height = new_step_h;
  }

  ctx.template Alloc<T>(out);
  ctx.template Alloc<T>(var);

  phi::DenseTensor r;
  phi::TensorFromVector(new_aspect_ratios, ctx, &r);
  auto aspect_ratios_desc = sdaa_ops::GetTecodnnTensorDesc(
      {new_aspect_ratios.size()}, r.dtype(), TensorFormat::Undefined);

  phi::DenseTensor min;
  phi::TensorFromVector(min_sizes, ctx, &min);
  auto min_desc = sdaa_ops::GetTecodnnTensorDesc(
      {min_sizes.size()}, min.dtype(), TensorFormat::Undefined);

  phi::DenseTensor max;
  phi::TensorFromVector(max_sizes, ctx, &max);
  auto max_desc = sdaa_ops::GetTecodnnTensorDesc(
      {max_sizes.size()}, max.dtype(), TensorFormat::Undefined);

  phi::DenseTensor v;
  phi::TensorFromVector(variances, ctx, &v);
  auto variances_desc = sdaa_ops::GetTecodnnTensorDesc(
      {variances.size()}, v.dtype(), TensorFormat::Undefined);
  PADDLE_ENFORCE_EQ(
      variances.size() == 4UL,
      true,
      phi::errors::InvalidArgument("The variances size must equal to 4. "
                                   " But got variances's size = [%s]. ",
                                   variances.size()));

  auto out_desc = sdaa_ops::GetTecodnnTensorDesc(
      phi::vectorize<int>(out->dims()), out->dtype(), TensorFormat::Undefined);
  auto var_desc = sdaa_ops::GetTecodnnTensorDesc(
      phi::vectorize<int>(var->dims()), var->dtype(), TensorFormat::Undefined);

  tecodnnHandle_t tecodnn_handle = custom_kernel::GetHandleFromCTX(ctx);

  TECODNN_CHECK(tecodnnPriorBox(tecodnn_handle,
                                height,
                                width,
                                im_height,
                                im_width,
                                new_offset,
                                step_width,
                                step_height,
                                clip,
                                min_max_aspect_ratios_order,
                                min_desc,
                                min.data(),
                                max_desc,
                                max.data(),
                                aspect_ratios_desc,
                                r.data(),
                                variances_desc,
                                v.data(),
                                out_desc,
                                out->data(),
                                var_desc,
                                var->data()));

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(min_desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(max_desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(aspect_ratios_desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(variances_desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(out_desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(var_desc));
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    prior_box, sdaa, ALL_LAYOUT, custom_kernel::PriorBoxKernel, float) {}
