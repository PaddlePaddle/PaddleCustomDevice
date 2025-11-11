// BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
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
// OF SUCH DAMAGE.

#include <iostream>

#include "kernels/amp/amp_funcs.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

template <typename T, typename Context>
class LazyZerosSDAA {
 public:
  void operator()(const Context& dev_ctx,
                  const std::vector<bool> found_inf_vec,
                  const std::vector<const phi::DenseTensor*>& xs,
                  const std::vector<phi::DenseTensor*>& outs) const {
    if (!xs.size()) {
      return;
    }

    for (size_t i = 0; i < xs.size(); ++i) {
      auto* out = outs[i];
      auto* x = xs[i];
      dev_ctx.template Alloc<T>(out);
      if (!found_inf_vec[0]) {
        VLOG(4) << "-- UpdateLossScaling: Find finite grads. --";
        TensorCopy(dev_ctx, *x, false, out);
      } else {
        VLOG(4) << "-- UpdateLossScaling: Find infinite grads. --";
        sdaa_ops::doMemsetTensor(dev_ctx, static_cast<int>(0), out);
      }
    }
  }
};

template <typename T, typename Context>
void Update(const Context& dev_ctx,
            const std::vector<bool> found_inf_vec,
            const phi::DenseTensor* pre_loss_scaling_tensor,
            const phi::DenseTensor* good_in_tensor,
            const phi::DenseTensor* bad_in_tensor,
            const int incr_every_n_steps,
            const int decr_every_n_nan_or_inf,
            const float incr_ratio,
            const float decr_ratio,
            phi::DenseTensor* updated_loss_scaling_tensor,
            phi::DenseTensor* good_out_tensor,
            phi::DenseTensor* bad_out_tensor) {
  dev_ctx.template Alloc<T>(updated_loss_scaling_tensor);
  dev_ctx.template Alloc<int>(good_out_tensor);
  dev_ctx.template Alloc<int>(bad_out_tensor);

  phi::DenseTensor* pre_loss_scaling_tensor_ =
      const_cast<phi::DenseTensor*>(pre_loss_scaling_tensor);

  if (found_inf_vec[0]) {
    // good_out_data = 0
    sdaa_ops::doFillTensor<T>(
        dev_ctx, static_cast<T>(0), good_out_tensor->dtype(), good_out_tensor);

    // bad_out_data = bad_in_data + 1
    amp_funcs::AddOne<T, Context>(dev_ctx, bad_in_tensor, bad_out_tensor);

    std::vector<int> bad_out_data;
    TensorToVector(dev_ctx, *bad_out_tensor, dev_ctx, &bad_out_data);

    if (bad_out_data[0] >= decr_every_n_nan_or_inf) {
      // updated_loss_scaling_data = pre_loss_scaling_data * decr_ratio
      sdaa_ops::doScaleTensor(dev_ctx,
                              *pre_loss_scaling_tensor,
                              decr_ratio,
                              0.0,
                              false,
                              false,
                              updated_loss_scaling_tensor);

      std::vector<T> new_loss_scaling;
      TensorToVector(
          dev_ctx, *updated_loss_scaling_tensor, dev_ctx, &new_loss_scaling);
      float min_value = 1.0;
      // updated_loss_scaling_data = new_loss_scaling < 1 ? 1 : new_loss_scaling
      if (new_loss_scaling[0] < static_cast<T>(min_value)) {
        sdaa_ops::doUnaryOpTensor(dev_ctx,
                                  *pre_loss_scaling_tensor,
                                  0.0,
                                  UnaryOpMode::POW,
                                  updated_loss_scaling_tensor);
      }

      // bad_out_data = 0
      sdaa_ops::doFillTensor<T>(
          dev_ctx, static_cast<T>(0), bad_out_tensor->dtype(), bad_out_tensor);
    }
  } else {
    // bad_out_data = 0
    sdaa_ops::doFillTensor<T>(
        dev_ctx, static_cast<T>(0), bad_out_tensor->dtype(), bad_out_tensor);

    // good_out_data = good_in_data + 1
    amp_funcs::AddOne<T, Context>(dev_ctx, good_in_tensor, good_out_tensor);

    std::vector<int> good_out_data;
    TensorToVector(dev_ctx, *good_out_tensor, dev_ctx, &good_out_data);

    if (good_out_data[0] >= incr_every_n_steps) {
      // updated_loss_scaling_data = pre_loss_scaling_data * incr_ratio
      sdaa_ops::doScaleTensor(dev_ctx,
                              *pre_loss_scaling_tensor,
                              incr_ratio,
                              0.0,
                              false,
                              false,
                              updated_loss_scaling_tensor);

      std::vector<T> new_loss_scaling;
      TensorToVector(
          dev_ctx, *updated_loss_scaling_tensor, dev_ctx, &new_loss_scaling);

      // updated_loss_scaling_data = new_loss_scaling != nan/inf ?
      // new_loss_scaling : pre_loss_scaling_data
      if (!std::isfinite(new_loss_scaling[0])) {
        sdaa_ops::doScaleTensor(dev_ctx,
                                *pre_loss_scaling_tensor,
                                1.0,
                                0.0,
                                false,
                                false,
                                updated_loss_scaling_tensor);
      }

      // good_out_data = 0
      sdaa_ops::doFillTensor<T>(dev_ctx,
                                static_cast<T>(0),
                                good_out_tensor->dtype(),
                                good_out_tensor);
    }
  }
}

template <typename T, typename Context>
void UpdateLossScaling(const Context& dev_ctx,
                       const std::vector<const phi::DenseTensor*>& xs,
                       const phi::DenseTensor& t_found_inf,
                       const phi::DenseTensor& t_pre_loss_scaling,
                       const phi::DenseTensor& t_good_in,
                       const phi::DenseTensor& t_bad_in,
                       int incr_every_n_steps,
                       int decr_every_n_nan_or_inf,
                       float incr_ratio,
                       float decr_ratio,
                       const phi::Scalar& stop_update,
                       std::vector<phi::DenseTensor*> outs,
                       phi::DenseTensor* updated_loss_scaling,
                       phi::DenseTensor* good_out,
                       phi::DenseTensor* bad_out) {
  VLOG(4) << "Call SDAA UpdateLossScaling";

  auto* found_inf = &t_found_inf;
  PADDLE_ENFORCE_EQ(
      found_inf->numel(),
      1,
      phi::errors::InvalidArgument("FoundInfinite must has only one element."));

  std::vector<bool> found_inf_vec;
  TensorToVector(dev_ctx, *found_inf, dev_ctx, &found_inf_vec);

  LazyZerosSDAA<T, Context>{}(dev_ctx, found_inf_vec, xs, outs);

  auto stop_update_val = stop_update.to<bool>();
  if (stop_update_val) {
    return;
  }

  auto* pre_loss_scaling = &t_pre_loss_scaling;
  auto* good_in = &t_good_in;
  auto* bad_in = &t_bad_in;

  Update<T, Context>(dev_ctx,
                     found_inf_vec,
                     pre_loss_scaling,
                     good_in,
                     bad_in,
                     incr_every_n_steps,
                     decr_every_n_nan_or_inf,
                     incr_ratio,
                     decr_ratio,
                     updated_loss_scaling,
                     good_out,
                     bad_out);
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(update_loss_scaling,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::UpdateLossScaling,
                          float,
                          phi::dtype::float16) {
  kernel->OutputAt(2).SetDataType(phi::DataType::INT32);
  kernel->OutputAt(3).SetDataType(phi::DataType::INT32);
}
