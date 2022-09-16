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

#include "kernels/funcs/mlu_funcs.h"

namespace custom_kernel {

template <typename T, typename Context>
void DropoutRawKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const paddle::optional<phi::DenseTensor>& seed_tensor,
                      const phi::Scalar& p,
                      bool is_test,
                      const std::string& mode,
                      int seed,
                      bool fix_seed,
                      phi::DenseTensor* out,
                      phi::DenseTensor* mask) {
  dev_ctx.template Alloc<T>(out);

  auto dropout_prob = p.to<float>();
  const bool is_upscale = (mode == "upscale_in_train");

  MLUCnnlTensorDesc x_desc(x);
  MLUCnnlTensorDesc out_desc(*out);
  if (!is_test) {
    // exec dropout op for training only.
    int seed_data = 0;
    if (seed_tensor) {
      std::vector<int> seed_vec;
      TensorToVector(dev_ctx, seed_tensor.get(), dev_ctx, &seed_vec);
      seed_data = seed_vec[0];
    } else {
      seed_data = fix_seed ? seed : 0;
    }

    dev_ctx.template Alloc<uint8_t>(mask);
    MLUCnnlTensorDesc mask_desc(*mask);
    // Special case when dropout_prob is 1.0
    if (dropout_prob == 1.0f) {
      auto value_t = static_cast<T>(0.0f);
      MLUCnnl::Fill(dev_ctx,
                    CNNL_POINTER_MODE_HOST,
                    &value_t,
                    out_desc.get(),
                    GetBasePtr(out));
      MLUCnnl::Fill(dev_ctx,
                    CNNL_POINTER_MODE_HOST,
                    &value_t,
                    mask_desc.get(),
                    GetBasePtr(mask));
      return;
    }

    // create mlu random generator
    const int device_id = dev_ctx.GetPlace().GetDeviceId();
    auto mlu_gen_random = GetMLURandomGenerator(dev_ctx, device_id, seed_data);

    const float prob = is_upscale ? dropout_prob : 0.0f;
    MLUCnnl::FusedDropout(dev_ctx,
                          mlu_gen_random->get(),
                          x_desc.get(),
                          GetBasePtr(&x),
                          prob,
                          GetBasePtr(&(mlu_gen_random->get_state())),
                          mask_desc.get(),
                          GetBasePtr(mask),
                          out_desc.get(),
                          GetBasePtr(out));
  } else {
    // exec dropout op for inference only.
    if (is_upscale) {
      TensorCopy(dev_ctx, x, false, out);
    } else {
      auto scale = static_cast<T>(1.0f - dropout_prob);
      Tensor scale_tensor;
      scale_tensor.Resize({1});
      dev_ctx.template Alloc<T>(&scale_tensor);
      MLUCnnlTensorDesc scale_desc(scale_tensor);
      MLUCnnl::Fill(dev_ctx,
                    CNNL_POINTER_MODE_HOST,
                    &scale,
                    scale_desc.get(),
                    GetBasePtr(&scale_tensor));

      auto data_type = ToCnnlDataType<T>();
      MLUCnnlOpTensorDesc op_tensor_desc(
          CNNL_OP_TENSOR_MUL, data_type, CNNL_NOT_PROPAGATE_NAN);
      MLUCnnl::OpTensor(dev_ctx,
                        op_tensor_desc.get(),
                        x_desc.get(),
                        GetBasePtr(&x),
                        scale_desc.get(),
                        GetBasePtr(&scale_tensor),
                        out_desc.get(),
                        GetBasePtr(out),
                        data_type);
    }
  }
}

template <typename T, typename Context>
void DropoutGradRawKernel(const Context& dev_ctx,
                          const phi::DenseTensor& mask,
                          const phi::DenseTensor& dout,
                          const phi::Scalar& p,
                          bool is_test,
                          const std::string& mode,
                          phi::DenseTensor* dx) {
  PADDLE_ENFORCE_EQ(is_test,
                    false,
                    phi::errors::InvalidArgument(
                        "GradOp is only callable when is_test is false"));
  auto dropout_prob = p.to<float>();
  dev_ctx.template Alloc<T>(dx);
  MLUCnnlTensorDesc grad_x_desc(*dx);
  if (dropout_prob == 1.) {
    auto value_t = static_cast<T>(0.0f);
    MLUCnnl::Fill(dev_ctx,
                  CNNL_POINTER_MODE_HOST,
                  &value_t,
                  grad_x_desc.get(),
                  GetBasePtr(dx));
    return;
  }
  // cast mask from uint8 to float32/float16
  Tensor cast_mask;
  cast_mask.Resize(mask.dims());
  dev_ctx.template Alloc<T>(&cast_mask);

  MLUCnnlTensorDesc mask_desc(mask);
  MLUCnnlTensorDesc cast_mask_desc(cast_mask);
  cnnlCastDataType_t cast_type =
      GetCastDataType(mask.dtype(), cast_mask.dtype());

  MLUCnnl::Cast(dev_ctx,
                cast_type,
                mask_desc.get(),
                GetBasePtr(&mask),
                cast_mask_desc.get(),
                GetBasePtr(&cast_mask));

  const bool is_upscale = (mode == "upscale_in_train");
  const float scale = is_upscale ? (1.0f / (1.0f - dropout_prob)) : (1.0f);
  auto data_type = ToCnnlDataType<T>();
  MLUCnnlTensorDesc grad_out_desc(dout);
  MLUCnnlOpTensorDesc op_tensor_desc(
      CNNL_OP_TENSOR_MUL, data_type, CNNL_NOT_PROPAGATE_NAN);
  MLUCnnl::OpTensor(dev_ctx,
                    op_tensor_desc.get(),
                    cast_mask_desc.get(),
                    GetBasePtr(&cast_mask),
                    grad_out_desc.get(),
                    GetBasePtr(&dout),
                    grad_x_desc.get(),
                    GetBasePtr(dx),
                    data_type,
                    scale);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(dropout,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::DropoutRawKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(dropout_grad,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::DropoutGradRawKernel,
                          float,
                          phi::dtype::float16) {}
