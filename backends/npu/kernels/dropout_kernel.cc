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
  auto dropout_prob = p.to<float>();

  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  if (dropout_prob == 1.) {
    const auto& runner_zeros_out = NpuOpRunner("ZerosLike", {*out}, {*out});
    runner_zeros_out.Run(stream);
    dev_ctx.template Alloc<uint8_t>(mask);
    const auto& runner_zeros_mask = NpuOpRunner("ZerosLike", {*mask}, {*mask});
    runner_zeros_mask.Run(stream);
    return;
  }

  // only achieve the default `upscale_in_train` method
  if (!is_test) {
    //   Tensor tmp_x(x->dtype());
    //   Tensor tmp_out(out->dtype());
    //   tmp_x.ShareDataWith(*x);
    //   tmp_out.ShareDataWith(*out);
    phi::DenseTensor tmp_x(x);
    phi::DenseTensor tmp_out(*out);

    if (x.dims().size() == 1) {
      // DropOutDoMask will get error result when input
      // is 1-D. Make it become 2-D.
      std::vector<int> vec_dim = phi::vectorize<int>(x.dims());
      tmp_x.Resize(phi::make_ddim({vec_dim[0], 1}));
      tmp_out.Resize(phi::make_ddim({vec_dim[0], 1}));
    }

    int seed = 0;
    int seed2 = 0;
    float keep_prob = 1. - dropout_prob;
    if (seed_tensor) {
      std::vector<int> seed_data;
      TensorToVector(dev_ctx, seed_tensor.get(), dev_ctx, &seed_data);
      seed = seed_data[0];
    } else {
      seed = fix_seed ? seed : 0;
    }

    phi::DenseTensor keep_prob_tensor;
    phi::DenseTensorMeta keep_prob_tensor_meta = {x.dtype(), {1}};
    keep_prob_tensor.set_meta(keep_prob_tensor_meta);
    FillNpuTensorWithConstant<T>(
        &keep_prob_tensor, dev_ctx, static_cast<T>(keep_prob));

    dev_ctx.template Alloc<uint8_t>(mask);

    // mask used in `DropOutGenMask` NPU OP is different from
    // the output `Mask`.
    uint32_t length = (x.numel() + 128 - 1) / 128 * 128;
    phi::DenseTensor npu_mask;
    phi::DenseTensorMeta npu_mask_meta = {paddle::experimental::DataType::UINT8,
                                          phi::make_ddim({length / 8})};
    npu_mask.set_meta(npu_mask_meta);
    dev_ctx.template Alloc<uint8_t>(&npu_mask);

    // TODO(pangyoki): `keep_prob` used in `DropOutGenMask` NPU
    // OP must be a scalar with shape[0]. At present, the shape
    // of the `prob` Tensor of this OP is forced to be set to 0
    // in `npu_op_runner.cc`, which needs to be optimized later.
    NpuOpRunner runner_gen_mask;
    runner_gen_mask.SetType("DropOutGenMask")
        .AddInput(dev_ctx, phi::vectorize(tmp_out.dims()))
        .AddInput(keep_prob_tensor)
        .AddOutput(npu_mask)
        .AddAttr("seed", seed)
        .AddAttr("seed2", seed2);
    runner_gen_mask.Run(stream);

    NpuOpRunner runner_dropout;
    runner_dropout.SetType("DropOutDoMask")
        .AddInput(tmp_x)
        .AddInput(npu_mask)
        .AddInput(keep_prob_tensor)
        .AddOutput(tmp_out);
    runner_dropout.Run(stream);

    // cast `out` from float/float16 to bool
    phi::DenseTensor cast_mask;
    phi::DenseTensorMeta cast_mask_meta = {paddle::experimental::DataType::BOOL,
                                           mask->dims()};
    cast_mask.set_meta(cast_mask_meta);
    dev_ctx.template Alloc<bool>(&cast_mask);

    auto dst_dtype_bool = ConvertToNpuDtype(cast_mask.dtype());
    const auto& runner_cast_mask_bool =
        NpuOpRunner("Cast",
                    {*out},
                    {cast_mask},
                    {{"dst_type", static_cast<int>(dst_dtype_bool)}});
    runner_cast_mask_bool.Run(stream);

    // cast cast_mask from bool to uint8
    auto dst_dtype_uint8 = ConvertToNpuDtype(mask->dtype());
    const auto& runner_cast_mask_uint8 =
        NpuOpRunner("Cast",
                    {cast_mask},
                    {*mask},
                    {{"dst_type", static_cast<int>(dst_dtype_uint8)}});
    runner_cast_mask_uint8.Run(stream);
  } else {
    TensorCopy(dev_ctx, x, false, out);
  }
}

template <typename T, typename Context>
void DropoutGradRawKernel(const Context& dev_ctx,
                          const phi::DenseTensor& mask,
                          const phi::DenseTensor& dout,
                          float p,
                          bool is_test,
                          const std::string& mode,
                          phi::DenseTensor* dx) {
  auto dropout_prob = p;

  dev_ctx.template Alloc<T>(dx);
  auto stream = dev_ctx.stream();

  if (dropout_prob == 1.) {
    const auto& runner_zeros = NpuOpRunner("ZerosLike", {*dx}, {*dx});
    runner_zeros.Run(stream);
    return;
  }

  // cast mask from uint8 to float32/float16
  phi::DenseTensor cast_mask;
  phi::DenseTensorMeta cast_mask_meta = {dx->dtype(), mask.dims()};
  cast_mask.set_meta(cast_mask_meta);
  dev_ctx.template Alloc<T>(&cast_mask);

  auto dst_dtype = ConvertToNpuDtype(dx->dtype());
  const auto& runner_cast_mask = NpuOpRunner(
      "Cast", {mask}, {cast_mask}, {{"dst_type", static_cast<int>(dst_dtype)}});
  runner_cast_mask.Run(stream);

  const auto& runner =
      NpuOpRunner("MaskedScale",
                  {dout, cast_mask},
                  {*dx},
                  {{"value", static_cast<float>(1. / (1 - dropout_prob))}});
  runner.Run(stream);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(dropout,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::DropoutRawKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(dropout_grad,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::DropoutGradRawKernel,
                          float,
                          phi::dtype::float16,
                          double) {}
