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
template <typename Context, typename T>
void Update(const Context& ctx,
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
  auto place = ctx.GetPlace();
  auto stream = ctx.stream();
  if (found_inf_vec[0]) {
    // good_out_data = 0
    auto g = ctx.template Alloc<int>(good_out_tensor);
    aclrtMemsetAsync(static_cast<void*>(g),
                     good_out_tensor->numel() * sizeof(int),
                     0,
                     good_out_tensor->numel() * sizeof(int),
                     stream);

    // bad_out_data = bad_in_data + 1
    phi::DenseTensor factor_tensor;
    factor_tensor.Resize({1});
    ctx.template Alloc<int>(&factor_tensor);
    FillNpuTensorWithConstant<int>(&factor_tensor, ctx, static_cast<int>(1));
    const auto& runner_p2 = NpuOpRunner(
        "Add", {*bad_in_tensor, factor_tensor}, {*bad_out_tensor}, {});
    runner_p2.Run(stream);

    std::vector<int> bad_out_data;
    TensorToVector(ctx, *bad_out_tensor, ctx, &bad_out_data);
    if (bad_out_data[0] >= decr_every_n_nan_or_inf) {
      const auto& runner_p3 = NpuOpRunner("Power",
                                          {*pre_loss_scaling_tensor},
                                          {*updated_loss_scaling_tensor},
                                          {{"power", static_cast<float>(1)},
                                           {"scale", decr_ratio},
                                           {"shift", static_cast<float>(0)}});

      runner_p3.Run(stream);

      std::vector<T> new_loss_scaling;
      TensorToVector(ctx, *updated_loss_scaling_tensor, ctx, &new_loss_scaling);
      float min_value = 1.0;

      int FLAGS_min_loss_scaling = 0;
      if (getenv("FLAGS_min_loss_scaling")) {
        auto ptr = getenv("FLAGS_min_loss_scaling");
        FLAGS_min_loss_scaling = ptr[0] == '1';
      }
      if (FLAGS_min_loss_scaling > 1) {
        min_value = static_cast<float>(FLAGS_min_loss_scaling);
      }

      if (new_loss_scaling[0] < min_value) {
        // updated_loss_scaling_data = 1
        const auto& runner_p4 =
            NpuOpRunner("Power",
                        {*pre_loss_scaling_tensor},
                        {*updated_loss_scaling_tensor},
                        {{"power", static_cast<float>(1)},
                         {"scale", static_cast<float>(0)},
                         {"shift", static_cast<float>(min_value)}});

        runner_p4.Run(stream);
      }

      // bad_out_data = 0
      auto b = ctx.template Alloc<int>(bad_out_tensor);
      aclrtMemsetAsync(static_cast<void*>(b),
                       bad_out_tensor->numel() * sizeof(int),
                       0,
                       bad_out_tensor->numel() * sizeof(int),
                       stream);
    }
  } else {
    // bad_out_data = 0
    auto b = ctx.template Alloc<int>(bad_out_tensor);
    aclrtMemsetAsync(static_cast<void*>(b),
                     bad_out_tensor->numel() * sizeof(int),
                     0,
                     bad_out_tensor->numel() * sizeof(int),
                     stream);

    // good_out_data = good_in_data + 1
    phi::DenseTensor factor_tensor;
    factor_tensor.Resize({1});
    ctx.template Alloc<int>(&factor_tensor);
    FillNpuTensorWithConstant<int>(&factor_tensor, ctx, static_cast<int>(1));
    const auto& runner_p2 = NpuOpRunner(
        "Add", {*good_in_tensor, factor_tensor}, {*good_out_tensor}, {});
    runner_p2.Run(stream);

    std::vector<int> good_out_data;
    TensorToVector(ctx, *good_out_tensor, ctx, &good_out_data);

    if (good_out_data[0] >= incr_every_n_steps) {
      const auto& runner_p3 = NpuOpRunner("Power",
                                          {*pre_loss_scaling_tensor},
                                          {*updated_loss_scaling_tensor},
                                          {{"power", static_cast<float>(1)},
                                           {"scale", incr_ratio},
                                           {"shift", static_cast<float>(0)}});
      runner_p3.Run(stream);

      std::vector<T> new_loss_scaling;
      TensorToVector(ctx, *updated_loss_scaling_tensor, ctx, &new_loss_scaling);
      if (!std::isfinite(new_loss_scaling[0])) {
        // updated_loss_scaling_data = pre_loss_scaling_data
        const auto& runner_p4 = NpuOpRunner("Power",
                                            {*pre_loss_scaling_tensor},
                                            {*updated_loss_scaling_tensor},
                                            {{"power", static_cast<float>(1)},
                                             {"scale", static_cast<float>(1)},
                                             {"shift", static_cast<float>(0)}});

        runner_p4.Run(stream);
      }
      // good_out_data = 0
      auto g = ctx.template Alloc<int>(good_out_tensor);
      aclrtMemsetAsync(static_cast<void*>(g),
                       good_out_tensor->numel() * sizeof(int),
                       0,
                       good_out_tensor->numel() * sizeof(int),
                       stream);
    }
  }
}

template <typename Context, typename T>
void UpdateLossScalingFunc(const Context& dev_ctx,
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
  Update<Context, T>(dev_ctx,
                     found_inf_vec,
                     pre_loss_scaling_tensor,
                     good_in_tensor,
                     bad_in_tensor,
                     incr_every_n_steps,
                     decr_every_n_nan_or_inf,
                     incr_ratio,
                     decr_ratio,
                     updated_loss_scaling_tensor,
                     good_out_tensor,
                     bad_out_tensor);
}

template <typename Context, typename T>
class LazyZerosNPU {
 public:
  void operator()(const Context& dev_ctx,
                  const std::vector<bool> found_inf_vec,
                  const std::vector<const phi::DenseTensor*>& xs,
                  const std::vector<phi::DenseTensor*>& outs) const {
    if (!xs.size()) {
      return;
    }
    auto place = dev_ctx.GetPlace();
    auto stream = dev_ctx.stream();
    phi::DenseTensor* zero_tensor = nullptr;
    void* zero_ptr = nullptr;
    if (found_inf_vec[0]) {
      int max_num = -1;
      for (size_t i = 0; i < xs.size(); ++i) {
        auto* out = outs[i];
        int num = out->numel();
        if (max_num < num) {
          max_num = num;
          zero_tensor = out;
        }
      }

      dev_ctx.template Alloc<T>(zero_tensor);
      const auto& runner_zeros =
          NpuOpRunner("ZerosLike", {*zero_tensor}, {*zero_tensor});
      runner_zeros.Run(stream);
      // zero_tensor->check_memory_size();
      zero_ptr = zero_tensor->data();
    }

    for (size_t i = 0; i < xs.size(); ++i) {
      auto* out = outs[i];
      auto* x = xs[i];
      auto dst_ptr = dev_ctx.template Alloc<T>(out);
      if (!found_inf_vec[0]) {
        TensorCopy(dev_ctx, *x, false, out);
      } else if (zero_ptr != dst_ptr) {
        auto size = out->numel() * paddle::experimental::SizeOf(out->dtype());
        aclrtMemcpyAsync(
            dst_ptr, size, zero_ptr, size, ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
      }
    }
  }
};

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
                       bool stop_update,
                       phi::DenseTensor* updated_loss_scaling,
                       phi::DenseTensor* good_out,
                       phi::DenseTensor* bad_out,
                       std::vector<phi::DenseTensor*> outs,
                       std::vector<phi::DenseTensor*> stop_update_vec) {
  auto* found_inf = &t_found_inf;
  PADDLE_ENFORCE_EQ(
      found_inf->numel(),
      1,
      phi::errors::InvalidArgument("FoundInfinite must has only one element."));

  std::vector<bool> found_inf_vec;
  TensorToVector(dev_ctx, *found_inf, dev_ctx, &found_inf_vec);

  LazyZerosNPU<Context, T>{}(dev_ctx, found_inf_vec, xs, outs);

  if (stop_update) {
    return;
  }

  auto* pre_loss_scaling = &t_pre_loss_scaling;
  auto* good_in = &t_good_in;
  auto* bad_in = &t_bad_in;

  dev_ctx.template Alloc<T>(updated_loss_scaling);
  dev_ctx.template Alloc<int>(good_out);
  dev_ctx.template Alloc<int>(bad_out);

  UpdateLossScalingFunc<Context, T>(dev_ctx,
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
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::UpdateLossScaling,
                          float,
                          double) {}
