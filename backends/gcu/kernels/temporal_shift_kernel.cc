// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "kernels/funcs/gcu_kernel_funcs.h"
#include "topsaten/topsaten_define.h"

namespace custom_kernel {

template <typename T, typename Context>
extern void StridedSliceKernel(const Context& dev_ctx,
                               const phi::DenseTensor& x,
                               const std::vector<int>& axes,
                               const phi::IntArray& starts,
                               const phi::IntArray& ends,
                               const phi::IntArray& strides,
                               phi::DenseTensor* out);

template <typename T, typename Context>
extern void ConcatKernel(const Context& dev_ctx,
                         const std::vector<const phi::DenseTensor*>& ins,
                         const phi::Scalar& axis_scalar,
                         phi::DenseTensor* out);

template <typename T, typename Context>
void TemporalShiftKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         int seg_num,
                         float shift_ratio,
                         const std::string& data_format_str,
                         DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("temporal_shift");
  const auto x_dims = x.dims();
  const auto n = x_dims[0] / seg_num;
  const auto t = seg_num;
  dev_ctx.template Alloc<T>(out);
  if (LaunchAOTKernel()) {
    if (data_format_str == "NHWC") {
      const auto c = x_dims[3];
      const auto h = x_dims[1];
      const auto w = x_dims[2];

      auto x5 = x;
      x5.Resize(common::make_ddim({n, t, h, w, c}));
      phi::DenseTensor pad_x;
      auto pad_x_meta = pad_x.meta();
      pad_x_meta.dtype = x.dtype();
      pad_x.set_meta(pad_x_meta);
      pad_x.Resize(common::make_ddim({n, t + 2, h, w, c}));
      dev_ctx.template Alloc<T>(&pad_x);

      std::vector<int64_t> paddings = {0L, 0L, 0L, 0L, 0L, 0L, 1L, 1L, 0L, 0L};
      phi::Scalar pad_value_scalar = 0.0f;

      LAUNCH_TOPSATENOP(topsatenPad,
                        dev_ctx,
                        pad_x,
                        x5,
                        paddings,
                        topsatenPadMode_t(0),
                        pad_value_scalar);

      phi::DenseTensorMeta meta;
      meta.dtype = x.dtype();
      phi::DenseTensor slice1;
      phi::DenseTensor slice2;
      phi::DenseTensor slice3;
      auto c_slice1 = static_cast<int>(c * shift_ratio);
      auto c_slice2 = c / 2 - static_cast<int>(c * shift_ratio);
      auto c_slice3 = c - c / 2;
      meta.dims = common::make_ddim({n, t, h, w, c_slice1});
      meta.strides = phi::DenseTensorMeta::calc_strides(meta.dims);
      slice1.set_meta(meta);
      meta.dims = common::make_ddim({n, t, h, w, c_slice2});
      meta.strides = phi::DenseTensorMeta::calc_strides(meta.dims);
      slice2.set_meta(meta);
      meta.dims = common::make_ddim({n, t, h, w, c_slice3});
      meta.strides = phi::DenseTensorMeta::calc_strides(meta.dims);
      slice3.set_meta(meta);
      dev_ctx.template Alloc<T>(&slice1);
      dev_ctx.template Alloc<T>(&slice2);
      dev_ctx.template Alloc<T>(&slice3);
      std::vector<int> axes = {1, 4};
      custom_kernel::StridedSliceKernel<T, Context>(
          dev_ctx,
          pad_x,
          axes,
          {0, 0},
          {t, static_cast<int>(c * shift_ratio)},
          {1, 1},
          &slice1);
      custom_kernel::StridedSliceKernel<T, Context>(
          dev_ctx,
          pad_x,
          axes,
          {2, static_cast<int>(c * shift_ratio)},
          {t + 2, c / 2},
          {1, 1},
          &slice2);
      custom_kernel::StridedSliceKernel<T, Context>(
          dev_ctx, pad_x, axes, {1, c / 2}, {t + 1, c}, {1, 1}, &slice3);
      out->Resize(common::make_ddim({n, t, h, w, c}));
      custom_kernel::ConcatKernel<T, Context>(
          dev_ctx, {&slice1, &slice2, &slice3}, 4, out);
      out->Resize(x.dims());
    } else {
      auto c = x_dims[1];
      auto h = x_dims[2];
      auto w = x_dims[3];

      auto x5 = x;
      x5.Resize(common::make_ddim({n, t, c, h, w}));
      phi::DenseTensor pad_x;
      auto pad_x_meta = pad_x.meta();
      pad_x_meta.dtype = x.dtype();
      pad_x.set_meta(pad_x_meta);
      pad_x.Resize(common::make_ddim({n, t + 2, c, h, w}));
      dev_ctx.template Alloc<T>(&pad_x);

      std::vector<int64_t> paddings = {0L, 0L, 0L, 0L, 0L, 0L, 1L, 1L, 0L, 0L};
      phi::Scalar pad_value_scalar = 0.0f;

      LAUNCH_TOPSATENOP(topsatenPad,
                        dev_ctx,
                        pad_x,
                        x5,
                        paddings,
                        topsatenPadMode_t(0),
                        pad_value_scalar);

      phi::DenseTensorMeta meta = x.meta();
      phi::DenseTensor slice1;
      phi::DenseTensor slice2;
      phi::DenseTensor slice3;
      auto c_slice1 = static_cast<int>(c * shift_ratio);
      auto c_slice2 = c / 2 - static_cast<int>(c * shift_ratio);
      auto c_slice3 = c - c / 2;
      meta.dims = common::make_ddim({n, t, c_slice1, h, w});
      meta.strides = phi::DenseTensorMeta::calc_strides(meta.dims);
      slice1.set_meta(meta);
      meta.dims = common::make_ddim({n, t, c_slice2, h, w});
      meta.strides = phi::DenseTensorMeta::calc_strides(meta.dims);
      slice2.set_meta(meta);
      meta.dims = common::make_ddim({n, t, c_slice3, h, w});
      meta.strides = phi::DenseTensorMeta::calc_strides(meta.dims);
      slice3.set_meta(meta);
      dev_ctx.template Alloc<T>(&slice1);
      dev_ctx.template Alloc<T>(&slice2);
      dev_ctx.template Alloc<T>(&slice3);
      std::vector<int> axes = {1, 2};
      custom_kernel::StridedSliceKernel<T, Context>(
          dev_ctx,
          pad_x,
          axes,
          {0, 0},
          {t, static_cast<int>(c * shift_ratio)},
          {1, 1},
          &slice1);
      custom_kernel::StridedSliceKernel<T, Context>(
          dev_ctx,
          pad_x,
          axes,
          {2, static_cast<int>(c * shift_ratio)},
          {t + 2, c / 2},
          {1, 1},
          &slice2);
      custom_kernel::StridedSliceKernel<T, Context>(
          dev_ctx, pad_x, axes, {1, c / 2}, {t + 1, c}, {1, 1}, &slice3);
      out->Resize(common::make_ddim({n, t, c, h, w}));
      custom_kernel::ConcatKernel<T, Context>(
          dev_ctx, {&slice1, &slice2, &slice3}, 2, out);
      out->Resize(x.dims());
    }
  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(temporal_shift,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::TemporalShiftKernel,
                          float,
                          double,
                          phi::dtype::bfloat16,
                          phi::dtype::float16) {}
