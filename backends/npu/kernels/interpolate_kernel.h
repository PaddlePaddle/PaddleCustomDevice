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
#pragma once
#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"
#include "kernels/funcs/slice_utils.h"

namespace custom_kernel {

template <typename T, typename Context>
struct InterpolateFunction {
 public:
  explicit InterpolateFunction(const Context& dev_ctx) : dev_ctx(dev_ctx) {
    place = dev_ctx.GetPlace();
    stream = dev_ctx.stream();
  }
  void Arange(int n, phi::DenseTensor* x) {
    if (x->dtype() == phi::DataType::FLOAT16) {
      phi::DenseTensor x_fp32;
      phi::DenseTensorMeta x_fp32_meta = {phi::DataType::FLOAT32, x->dims()};
      x_fp32.set_meta(x_fp32_meta);
      dev_ctx.template Alloc<float>(&x_fp32);

      NpuOpRunner runner;
      // Fix refer to
      // https://gitee.com/ascend/modelzoo/issues/I6K3HN?from=project-issue
      runner.SetType("Range")
          .AddInput(dev_ctx,
                    std::move(std::vector<float>{static_cast<float>(0)}),
                    false)
          .AddInput(dev_ctx,
                    std::move(std::vector<float>{static_cast<float>(n)}),
                    false)
          .AddInput(dev_ctx,
                    std::move(std::vector<float>{static_cast<float>(1)}),
                    false)
          .AddOutput(x_fp32);
      runner.Run(stream);
      Cast(&x_fp32, x);
    } else {
      NpuOpRunner runner;
      // Fix refer to
      // https://gitee.com/ascend/modelzoo/issues/I6K3HN?from=project-issue
      runner.SetType("Range")
          .AddInput(dev_ctx,
                    std::move(std::vector<float>{static_cast<float>(0)}),
                    false)
          .AddInput(dev_ctx,
                    std::move(std::vector<float>{static_cast<float>(n)}),
                    false)
          .AddInput(dev_ctx,
                    std::move(std::vector<float>{static_cast<float>(1)}),
                    false)
          .AddOutput(*x);
      runner.Run(stream);
    }
  }
  void ReduceSum(const phi::DenseTensor* x,
                 phi::DenseTensor* y,
                 const std::vector<int>& dim,
                 bool keep_dims = true) {
    const auto& runner = NpuOpRunner(
        "ReduceSumD", {*x}, {*y}, {{"axes", dim}, {"keep_dims", keep_dims}});
    runner.Run(stream);
  }
  void Add(const phi::DenseTensor* x,
           const phi::DenseTensor* y,
           phi::DenseTensor* z) {
    const auto& runner = NpuOpRunner("AddV2", {*x, *y}, {*z}, {});
    runner.Run(stream);
  }
  void Adds(const phi::DenseTensor* x, float scalar, phi::DenseTensor* y) {
    const auto& runner = NpuOpRunner("Adds", {*x}, {*y}, {{"value", scalar}});
    runner.Run(stream);
  }
  void Mul(const phi::DenseTensor* x,
           const phi::DenseTensor* y,
           phi::DenseTensor* z) {
    const auto& runner = NpuOpRunner("Mul", {*x, *y}, {*z}, {});
    runner.Run(stream);
  }
  void Sub(const phi::DenseTensor* x,
           const phi::DenseTensor* y,
           phi::DenseTensor* z) {
    const auto& runner = NpuOpRunner("Sub", {*x, *y}, {*z}, {});
    runner.Run(stream);
  }
  void Cast(const phi::DenseTensor* x, phi::DenseTensor* y) {
    auto dst_dtype = ConvertToNpuDtype(y->dtype());
    const auto& runner = NpuOpRunner(
        "Cast", {*x}, {*y}, {{"dst_type", static_cast<int>(dst_dtype)}});
    runner.Run(stream);
  }
  void Gather(const phi::DenseTensor* x,
              const phi::DenseTensor* indices,
              const int axis,
              phi::DenseTensor* y) {
    const auto& runner =
        NpuOpRunner("GatherV2D", {*x, *indices}, {*y}, {{"axis", axis}});
    runner.Run(stream);
  }
  void GatherGrad(const phi::DenseTensor* gy,
                  const phi::DenseTensor* indices,
                  const int axis,
                  phi::DenseTensor* gx) {
    //  1  gy swapaxis: axis & 0
    int len = (gy->dims()).size();
    std::vector<int> axis_swap(len);
    for (int i = 0; i < len; i++) {
      axis_swap[i] = i;
    }
    axis_swap[0] = axis;
    axis_swap[axis] = 0;
    auto y_new_shape = gy->dims();
    auto yt = y_new_shape[axis];
    y_new_shape[axis] = y_new_shape[0];
    y_new_shape[0] = yt;
    phi::DenseTensor gy_t;
    gy_t.Resize(y_new_shape);
    dev_ctx.template Alloc<T>(&gy_t);
    Transpose(gy, &gy_t, axis_swap);
    //  2  scatter
    auto x_new_shape = gx->dims();
    auto xt = x_new_shape[axis];
    x_new_shape[axis] = x_new_shape[0];
    x_new_shape[0] = xt;
    phi::DenseTensor gx_zero, gx_t;
    gx_zero.Resize(x_new_shape);
    gx_t.Resize(x_new_shape);
    dev_ctx.template Alloc<T>(&gx_zero);
    dev_ctx.template Alloc<T>(&gx_t);
    FillNpuTensorWithConstant<T>(&gx_zero, dev_ctx, static_cast<T>(0));
    gx_zero.Resize(x_new_shape);
    Scatter(&gx_zero, indices, &gy_t, &gx_t);
    //  3  gx swapaxis: axis, 0
    Transpose(&gx_t, gx, axis_swap);
  }
  void Scatter(const phi::DenseTensor* x,
               const phi::DenseTensor* index,
               const phi::DenseTensor* updates,
               phi::DenseTensor* y) {
    const auto& runner =
        NpuOpRunner("TensorScatterAdd", {*x, *index, *updates}, {*y}, {});
    runner.Run(stream);
  }
  void Transpose(const phi::DenseTensor* x,
                 phi::DenseTensor* y,
                 const std::vector<int>& axis) {
    const auto& runner =
        NpuOpRunner("TransposeD", {*x}, {*y}, {{"perm", axis}});
    runner.Run(stream);
  }
  void Muls(const phi::DenseTensor* x, float scalar, phi::DenseTensor* y) {
    const auto& runner = NpuOpRunner("Muls", {*x}, {*y}, {{"value", scalar}});
    runner.Run(stream);
  }
  void Maximum(const phi::DenseTensor* x,
               const phi::DenseTensor* y,
               phi::DenseTensor* z) {
    const auto& runner = NpuOpRunner("Maximum", {*x, *y}, {*z}, {});
    runner.Run(stream);
  }
  void Minimum(const phi::DenseTensor* x,
               const phi::DenseTensor* y,
               phi::DenseTensor* z) {
    const auto& runner = NpuOpRunner("Minimum", {*x, *y}, {*z}, {});
    runner.Run(stream);
  }
  void Floor(const phi::DenseTensor* x, phi::DenseTensor* y) {
    const auto& runner = NpuOpRunner("Floor", {*x}, {*y}, {});
    runner.Run(stream);
  }

 private:
  phi::Place place;
  aclrtStream stream;
  const Context& dev_ctx;
};

void InterpolateParamCompute(const float scale_h,
                             const float scale_w,
                             const bool align_corners,
                             const int align_mode,
                             const phi::DataLayout& data_layout,
                             const phi::DDim& indim,
                             const phi::DDim& outdim,
                             int* axis_h,
                             int* axis_w,
                             int* in_h,
                             int* in_w,
                             int* out_h,
                             int* out_w,
                             float* ratio_h,
                             float* ratio_w) {
  if (data_layout == phi::DataLayout::kNCHW) {
    *axis_h = 2;
    *axis_w = 3;
  } else {
    *axis_h = 1;
    *axis_w = 2;
  }
  *out_h = outdim[*axis_h];
  *out_w = outdim[*axis_w];
  *in_h = indim[*axis_h];
  *in_w = indim[*axis_w];
  *ratio_h = 0.0f;
  *ratio_w = 0.0f;
  if (*out_h > 1) {
    *ratio_h =
        align_corners
            ? static_cast<float>(*in_h - 1) / (*out_h - 1)
            : (scale_h > 0 ? 1 / scale_h : static_cast<float>(*in_h) / *out_h);
  }
  if (*out_w > 1) {
    *ratio_w =
        align_corners
            ? static_cast<float>(*in_w - 1) / (*out_w - 1)
            : (scale_w > 0 ? 1 / scale_w : static_cast<float>(*in_w) / *out_w);
  }
}

template <typename T, typename Context>
void BilinearParamTensorCompute(const Context& dev_ctx,
                                const phi::DataLayout& data_layout,
                                int in_h,
                                int in_w,
                                int out_h,
                                int out_w,
                                bool align_cond,
                                float ratio_h,
                                float ratio_w,
                                phi::DenseTensor* h0,
                                phi::DenseTensor* h1,
                                phi::DenseTensor* w0,
                                phi::DenseTensor* w1,
                                phi::DenseTensor* coef_h0,
                                phi::DenseTensor* coef_h1,
                                phi::DenseTensor* coef_w0,
                                phi::DenseTensor* coef_w1) {
  InterpolateFunction<T, Context> F(dev_ctx);
  auto place = dev_ctx.GetPlace();
  phi::DenseTensor _h0, _w0;
  _h0.Resize({out_h});
  _w0.Resize({out_w});
  dev_ctx.template Alloc<T>(&_h0);
  dev_ctx.template Alloc<T>(&_w0);
  F.Arange(out_h, &_h0);
  F.Arange(out_w, &_w0);
  if (align_cond) {
    F.Adds(&_h0, static_cast<float>(0.5), &_h0);
    F.Adds(&_w0, static_cast<float>(0.5), &_w0);
    F.Muls(&_h0, ratio_h, &_h0);
    F.Muls(&_w0, ratio_w, &_w0);
    F.Adds(&_h0, static_cast<float>(-0.5), &_h0);
    F.Adds(&_w0, static_cast<float>(-0.5), &_w0);
  } else {
    F.Muls(&_h0, ratio_h, &_h0);
    F.Muls(&_w0, ratio_w, &_w0);
  }

  phi::DenseTensor zero_t;
  phi::DenseTensor one_t;
  zero_t.Resize({1});
  one_t.Resize({1});
  dev_ctx.template Alloc<T>(&zero_t);
  dev_ctx.template Alloc<T>(&one_t);
  FillNpuTensorWithConstant<T>(&zero_t, dev_ctx, static_cast<T>(0));
  FillNpuTensorWithConstant<T>(&one_t, dev_ctx, static_cast<T>(1));
  F.Maximum(&_h0, &zero_t, &_h0);
  F.Maximum(&_w0, &zero_t, &_w0);

  phi::DenseTensor _h0_floor, _w0_floor;
  _h0_floor.Resize({out_h});
  _w0_floor.Resize({out_w});
  dev_ctx.template Alloc<T>(&_h0_floor);
  dev_ctx.template Alloc<T>(&_w0_floor);
  F.Floor(&_h0, &_h0_floor);
  F.Floor(&_w0, &_w0_floor);
  F.Cast(&_h0_floor, h0);
  F.Cast(&_w0_floor, w0);

  phi::DenseTensor one_int;
  one_int.Resize({1});
  dev_ctx.template Alloc<T>(&one_int);
  FillNpuTensorWithConstant<int>(&one_int, dev_ctx, static_cast<int>(1));
  F.Add(h0, &one_int, h1);
  F.Add(w0, &one_int, w1);
  phi::DenseTensor t_max_h, t_max_w;
  t_max_h.Resize({1});
  t_max_w.Resize({1});
  dev_ctx.template Alloc<T>(&t_max_h);
  dev_ctx.template Alloc<T>(&t_max_w);
  FillNpuTensorWithConstant<int>(&t_max_h, dev_ctx, static_cast<int>(in_h - 1));
  FillNpuTensorWithConstant<int>(&t_max_w, dev_ctx, static_cast<int>(in_w - 1));
  F.Minimum(h1, &t_max_h, h1);
  F.Minimum(w1, &t_max_w, w1);

  F.Sub(&_h0, &_h0_floor, coef_h1);
  F.Sub(&_w0, &_w0_floor, coef_w1);
  F.Sub(&one_t, coef_h1, coef_h0);
  F.Sub(&one_t, coef_w1, coef_w0);

  if (data_layout == phi::DataLayout::kNCHW) {
    coef_h0->Resize({out_h, 1});
    coef_h1->Resize({out_h, 1});
  } else {
    coef_h0->Resize({out_h, 1, 1});
    coef_h1->Resize({out_h, 1, 1});
    coef_w0->Resize({out_w, 1});
    coef_w1->Resize({out_w, 1});
  }
}

template <typename T, typename Context>
void BilinearFwdNpu(const Context& dev_ctx,
                    const phi::DenseTensor* input,
                    phi::DenseTensor* output,
                    const float scale_h,
                    const float scale_w,
                    const bool align_corners,
                    const int align_mode,
                    const phi::DataLayout& data_layout) {
  InterpolateFunction<T, Context> F(dev_ctx);
  auto place = dev_ctx.GetPlace();
  auto outdim = output->dims();
  auto indim = input->dims();

  int axis_h, axis_w;
  int out_h, out_w, in_h, in_w;
  float ratio_h, ratio_w;
  InterpolateParamCompute(scale_h,
                          scale_w,
                          align_corners,
                          align_mode,
                          data_layout,
                          indim,
                          outdim,
                          &axis_h,
                          &axis_w,
                          &in_h,
                          &in_w,
                          &out_h,
                          &out_w,
                          &ratio_h,
                          &ratio_w);

  phi::DenseTensor h0, h1, w0, w1;
  h0.Resize({out_h});
  h1.Resize({out_h});
  w0.Resize({out_w});
  w1.Resize({out_w});
  dev_ctx.template Alloc<int>(&h0);
  dev_ctx.template Alloc<int>(&h1);
  dev_ctx.template Alloc<int>(&w0);
  dev_ctx.template Alloc<int>(&w1);
  phi::DenseTensor coef_h0, coef_h1, coef_w0, coef_w1;
  coef_h0.Resize({out_h});
  coef_h1.Resize({out_h});
  coef_w0.Resize({out_w});
  coef_w1.Resize({out_w});
  dev_ctx.template Alloc<T>(&coef_h0);
  dev_ctx.template Alloc<T>(&coef_h1);
  dev_ctx.template Alloc<T>(&coef_w0);
  dev_ctx.template Alloc<T>(&coef_w1);
  bool align_cond = align_mode == 0 && !align_corners;
  BilinearParamTensorCompute<T>(dev_ctx,
                                data_layout,
                                in_h,
                                in_w,
                                out_h,
                                out_w,
                                align_cond,
                                ratio_h,
                                ratio_w,
                                &h0,
                                &h1,
                                &w0,
                                &w1,
                                &coef_h0,
                                &coef_h1,
                                &coef_w0,
                                &coef_w1);

  phi::DenseTensor input_gather_h0, input_gather_h1;
  auto dim_gather_h = indim;
  dim_gather_h[axis_h] = out_h;
  input_gather_h0.Resize(dim_gather_h);
  input_gather_h1.Resize(dim_gather_h);
  dev_ctx.template Alloc<T>(&input_gather_h0);
  dev_ctx.template Alloc<T>(&input_gather_h1);
  F.Gather(input, &h0, axis_h, &input_gather_h0);
  F.Gather(input, &h1, axis_h, &input_gather_h1);

  F.Mul(&input_gather_h0, &coef_h0, &input_gather_h0);
  F.Mul(&input_gather_h1, &coef_h1, &input_gather_h1);
  phi::DenseTensor out_x4;
  out_x4.Resize({4, outdim[0], outdim[1], outdim[2], outdim[3]});
  dev_ctx.template Alloc<T>(&out_x4);
  phi::DenseTensor input_gather_h0_w0 = custom_kernel::Slice(out_x4, 0, 1);
  phi::DenseTensor input_gather_h0_w1 = custom_kernel::Slice(out_x4, 1, 2);
  phi::DenseTensor input_gather_h1_w0 = custom_kernel::Slice(out_x4, 2, 3);
  phi::DenseTensor input_gather_h1_w1 = custom_kernel::Slice(out_x4, 3, 4);
  F.Gather(&input_gather_h0, &w0, axis_w, &input_gather_h0_w0);
  F.Gather(&input_gather_h0, &w1, axis_w, &input_gather_h0_w1);
  F.Gather(&input_gather_h1, &w0, axis_w, &input_gather_h1_w0);
  F.Gather(&input_gather_h1, &w1, axis_w, &input_gather_h1_w1);
  F.Mul(&input_gather_h0_w0, &coef_w0, &input_gather_h0_w0);
  F.Mul(&input_gather_h0_w1, &coef_w1, &input_gather_h0_w1);
  F.Mul(&input_gather_h1_w0, &coef_w0, &input_gather_h1_w0);
  F.Mul(&input_gather_h1_w1, &coef_w1, &input_gather_h1_w1);
  F.ReduceSum(&out_x4, output, std::vector<int>{0}, false);
}

template <typename T, typename Context>
void BilinearBwdNpu(const Context& dev_ctx,
                    const phi::DenseTensor* gout,
                    phi::DenseTensor* gin,
                    const float scale_h,
                    const float scale_w,
                    const bool align_corners,
                    const int align_mode,
                    const phi::DataLayout& data_layout) {
  InterpolateFunction<T, Context> F(dev_ctx);
  auto place = dev_ctx.GetPlace();
  auto outdim = gout->dims();
  auto indim = gin->dims();

  int axis_h, axis_w;
  int out_h, out_w, in_h, in_w;
  float ratio_h, ratio_w;
  InterpolateParamCompute(scale_h,
                          scale_w,
                          align_corners,
                          align_mode,
                          data_layout,
                          indim,
                          outdim,
                          &axis_h,
                          &axis_w,
                          &in_h,
                          &in_w,
                          &out_h,
                          &out_w,
                          &ratio_h,
                          &ratio_w);

  phi::DenseTensor h0, h1, w0, w1;
  h0.Resize({out_h});
  h1.Resize({out_h});
  w0.Resize({out_w});
  w1.Resize({out_w});
  dev_ctx.template Alloc<int>(&h0);
  dev_ctx.template Alloc<int>(&h1);
  dev_ctx.template Alloc<int>(&w0);
  dev_ctx.template Alloc<int>(&w1);
  phi::DenseTensor coef_h0, coef_h1, coef_w0, coef_w1;
  coef_h0.Resize({out_h});
  coef_h1.Resize({out_h});
  coef_w0.Resize({out_w});
  coef_w1.Resize({out_w});
  dev_ctx.template Alloc<T>(&coef_h0);
  dev_ctx.template Alloc<T>(&coef_h1);
  dev_ctx.template Alloc<T>(&coef_w0);
  dev_ctx.template Alloc<T>(&coef_w1);
  bool align_cond = align_mode == 0 && !align_corners;
  BilinearParamTensorCompute<T, Context>(dev_ctx,
                                         data_layout,
                                         in_h,
                                         in_w,
                                         out_h,
                                         out_w,
                                         align_cond,
                                         ratio_h,
                                         ratio_w,
                                         &h0,
                                         &h1,
                                         &w0,
                                         &w1,
                                         &coef_h0,
                                         &coef_h1,
                                         &coef_w0,
                                         &coef_w1);
  phi::DenseTensor gy_w0, gy_w1;
  gy_w0.Resize(outdim);
  gy_w1.Resize(outdim);
  dev_ctx.template Alloc<T>(&gy_w0);
  dev_ctx.template Alloc<T>(&gy_w1);
  F.Mul(gout, &coef_w0, &gy_w0);
  F.Mul(gout, &coef_w1, &gy_w1);

  auto dim_gather_h = indim;
  dim_gather_h[axis_h] = out_h;
  phi::DenseTensor g_gather_w0, g_gather_w1;
  g_gather_w0.Resize(dim_gather_h);
  g_gather_w1.Resize(dim_gather_h);
  dev_ctx.template Alloc<T>(&g_gather_w0);
  dev_ctx.template Alloc<T>(&g_gather_w1);

  w0.Resize({out_w, 1});
  w1.Resize({out_w, 1});
  F.GatherGrad(&gy_w0, &w0, axis_w, &g_gather_w0);
  F.GatherGrad(&gy_w1, &w1, axis_w, &g_gather_w1);

  F.Add(&g_gather_w0, &g_gather_w1, &g_gather_w0);
  F.Mul(&g_gather_w0, &coef_h1, &g_gather_w1);
  F.Mul(&g_gather_w0, &coef_h0, &g_gather_w0);

  phi::DenseTensor gx_0, gx_1;
  gx_0.Resize(indim);
  gx_1.Resize(indim);
  dev_ctx.template Alloc<T>(&gx_0);
  dev_ctx.template Alloc<T>(&gx_1);
  h0.Resize({out_h, 1});
  h1.Resize({out_h, 1});
  F.GatherGrad(&g_gather_w0, &h0, axis_h, &gx_0);
  F.GatherGrad(&g_gather_w1, &h1, axis_h, &gx_1);

  F.Add(&gx_0, &gx_1, gin);
}

}  // namespace custom_kernel
