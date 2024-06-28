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
void SubtractRawKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& y,
                       int axis,
                       phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  const auto& runner = NpuOpRunner("Sub", {x, y}, {*out}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void SubtractKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    phi::DenseTensor* out) {
  int axis = -1;
  DO_COMPATIBILITY(
      aclnnSub,
      (custom_kernel::SubtractRawKernel<T, Context>(dev_ctx, x, y, axis, out)));
  dev_ctx.template Alloc<T>(out);
  aclDataType acl_data_type = ConvertToNpuDtype(x.dtype());
  static const auto aclCreateScalar = GET_OP_API_FUNC(aclCreateScalar);
  auto one = static_cast<T>(1.0);
  aclScalar* acl_scalar_one = aclCreateScalar(&one, acl_data_type);
  EXEC_NPU_CMD(aclnnSub, dev_ctx, x, y, acl_scalar_one, *out);
}

template <typename T, typename Context>
void AclopSubtractGradKernel(const Context& dev_ctx,
                             const phi::DenseTensor& x,
                             const phi::DenseTensor& y,
                             const phi::DenseTensor& dout,
                             int axis,
                             phi::DenseTensor* dx,
                             phi::DenseTensor* dy) {
  auto stream = dev_ctx.stream();

  // NOTE(zhiqiu): It seems npu Sub follow the broadcast sematics with
  // default axis=-1?
  // So, the sub_grad should do reduce if needed.
  // For example, the shape of each variable in elementwise_sub:
  // x, dx: [2, 3, 5]
  // y, dy: [1, 5]
  // out, dout: [2, 3, 5]
  // Then, out = x - y  =>  dx = dout, dy = -dout
  // And, the shape of dy can be computed by two stages reduce,
  // 1. [2, 3, 5] => [3, 5], ReduceSumD on axis = 0, keep_dims = false.
  // 2. [3, 5] => [1, 5], ReduceSumD on axis = 0, keep_dims = true.

  if (dx) {
    dev_ctx.template Alloc<T>(dx);
    // For dx
    // stage 1
    auto reduce_ndim = dout.dims().size() - dx->dims().size();
    std::vector<int> axes;
    for (auto i = 0; i < reduce_ndim; ++i) {
      axes.push_back(i);
    }
    phi::DenseTensor axes_t;
    axes_t.Resize({axes.size()});
    dev_ctx.template Alloc<int>(&axes_t);
    custom_kernel::TensorFromVector(dev_ctx, axes, dev_ctx, &axes_t);

    phi::DenseTensor* tmp_dout = const_cast<phi::DenseTensor*>(&dout);
    phi::DenseTensor reduced_dout;
    if (axes.size() != 0) {
      std::vector<int64_t> reduced_dout_dims;
      for (auto i = reduce_ndim; i < dout.dims().size(); ++i) {
        reduced_dout_dims.push_back(dout.dims()[i]);
      }

      phi::DenseTensorMeta reduced_dout_meta = {
          dx->dtype(), phi::make_ddim(reduced_dout_dims)};
      reduced_dout.set_meta(reduced_dout_meta);
      dev_ctx.template Alloc<T>(&reduced_dout);

      const auto& sum_runner = NpuOpRunner(
          "ReduceSum", {dout, axes_t}, {reduced_dout}, {{"keep_dims", false}});
      sum_runner.Run(stream, true);
      tmp_dout = &reduced_dout;
    }

    // stage 2
    axes.clear();
    for (auto i = 0; i < dx->dims().size(); ++i) {
      if (dx->dims()[i] == 1) {
        axes.push_back(i);
      }
    }
    if (axes.size() != 0) {
      phi::DenseTensor axes_t1;
      axes_t1.Resize({axes.size()});
      dev_ctx.template Alloc<int>(&axes_t1);
      custom_kernel::TensorFromVector(dev_ctx, axes, dev_ctx, &axes_t1);
      // For inplace strategy, dx will be stored in addr of dout, which makes
      // the result of dy wrong.
      if (dx->IsSharedWith(dout)) {
        dx->clear();
        dx->Resize(x.dims());
        dev_ctx.template Alloc<T>(dx);
      }
      const auto& sum_runner = NpuOpRunner(
          "ReduceSum", {*tmp_dout, axes_t1}, {*dx}, {{"keep_dims", true}});
      sum_runner.Run(stream, true);
    } else {
      TensorCopy(dev_ctx, *tmp_dout, false, dx);
    }
  }
  if (dy) {
    dev_ctx.template Alloc<T>(dy);
    // For dy
    // stage 1
    auto reduce_ndim = dout.dims().size() - dy->dims().size();
    std::vector<int> axes;
    for (auto i = 0; i < reduce_ndim; ++i) {
      axes.push_back(i);
    }
    phi::DenseTensor* tmp_dout = const_cast<phi::DenseTensor*>(&dout);
    phi::DenseTensor reduced_dy;
    phi::DenseTensor reduced_dout;

    if (axes.size() != 0) {
      std::vector<int64_t> reduced_dout_dims;
      for (auto i = reduce_ndim; i < dout.dims().size(); ++i) {
        reduced_dout_dims.push_back(dout.dims()[i]);
      }

      phi::DenseTensorMeta reduced_dout_meta = {
          dy->dtype(), phi::make_ddim(reduced_dout_dims)};
      reduced_dout.set_meta(reduced_dout_meta);
      dev_ctx.template Alloc<T>(&reduced_dout);

      phi::DenseTensor axes_t2;
      axes_t2.Resize({axes.size()});
      dev_ctx.template Alloc<int>(&axes_t2);
      custom_kernel::TensorFromVector(dev_ctx, axes, dev_ctx, &axes_t2);
      const auto& sum_runner = NpuOpRunner(
          "ReduceSum", {dout, axes_t2}, {reduced_dout}, {{"keep_dims", false}});
      sum_runner.Run(stream, true);
      tmp_dout = &reduced_dout;
    }

    // stage 2
    axes.clear();
    phi::DenseTensor* tmp_dy = tmp_dout;
    for (auto i = 0; i < dy->dims().size(); ++i) {
      if (dy->dims()[i] == 1) {
        axes.push_back(i);
      }
    }
    if (axes.size() != 0) {
      phi::DenseTensorMeta reduced_dy_meta = {dy->dtype(), dy->dims()};
      reduced_dy.set_meta(reduced_dy_meta);
      dev_ctx.template Alloc<T>(&reduced_dy);

      phi::DenseTensor axes_t3;
      axes_t3.Resize({axes.size()});
      dev_ctx.template Alloc<int>(&axes_t3);
      custom_kernel::TensorFromVector(dev_ctx, axes, dev_ctx, &axes_t3);
      const auto& sum_runner = NpuOpRunner("ReduceSum",
                                           {*tmp_dout, axes_t3},
                                           {reduced_dy},
                                           {{"keep_dims", true}});
      sum_runner.Run(stream, true);

      tmp_dy = &reduced_dy;
    }

    // stage 3, negative
    const auto& neg_runner = NpuOpRunner("Neg", {*tmp_dy}, {*dy}, {});
    neg_runner.Run(stream);
  }
}

template <typename T, typename Context>
void SubtractGradKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& y,
                        const phi::DenseTensor& dout,
                        int axis,
                        phi::DenseTensor* dx,
                        phi::DenseTensor* dy) {
  DO_COMPATIBILITY(aclnnReduceSum,
                   (custom_kernel::AclopSubtractGradKernel<T, Context>(
                       dev_ctx, x, y, dout, axis, dx, dy)));
  auto stream = dev_ctx.stream();
  bool keep_dim;

  // NOTE(zhiqiu): It seems npu Sub follow the broadcast sematics with
  // default axis=-1?
  // So, the sub_grad should do reduce if needed.
  // For example, the shape of each variable in elementwise_sub:
  // x, dx: [2, 3, 5]
  // y, dy: [1, 5]
  // out, dout: [2, 3, 5]
  // Then, out = x - y  =>  dx = dout, dy = -dout
  // And, the shape of dy can be computed by two stages reduce,
  // 1. [2, 3, 5] => [3, 5], ReduceSumD on axis = 0, keep_dims = false.
  // 2. [3, 5] => [1, 5], ReduceSumD on axis = 0, keep_dims = true.'

  if (dx) {
    dev_ctx.template Alloc<T>(dx);
    // For dx
    // stage 1
    auto reduce_ndim = dout.dims().size() - dx->dims().size();
    std::vector<int> axes;
    for (auto i = 0; i < reduce_ndim; ++i) {
      axes.push_back(i);
    }

    phi::DenseTensor* tmp_dout = const_cast<phi::DenseTensor*>(&dout);
    phi::DenseTensor reduced_dout;
    if (axes.size() != 0) {
      std::vector<int64_t> reduced_dout_dims;
      for (auto i = reduce_ndim; i < dout.dims().size(); ++i) {
        reduced_dout_dims.push_back(dout.dims()[i]);
      }

      phi::DenseTensorMeta reduced_dout_meta = {
          dx->dtype(), phi::make_ddim(reduced_dout_dims)};
      reduced_dout.set_meta(reduced_dout_meta);
      dev_ctx.template Alloc<T>(&reduced_dout);

      keep_dim = false;
      auto dtype = ConvertToNpuDtype(reduced_dout.dtype());
      auto axis = phi::IntArray(axes);
      EXEC_NPU_CMD(
          aclnnReduceSum, dev_ctx, dout, axis, keep_dim, dtype, reduced_dout);
      tmp_dout = &reduced_dout;
    }

    // stage 2
    axes.clear();
    for (auto i = 0; i < dx->dims().size(); ++i) {
      if (dx->dims()[i] == 1) {
        axes.push_back(i);
      }
    }
    if (axes.size() != 0) {
      // For inplace strategy, dx will be stored in addr of dout, which makes
      // the result of dy wrong.
      if (dx->IsSharedWith(dout)) {
        dx->clear();
        dx->Resize(x.dims());
        dev_ctx.template Alloc<T>(dx);
      }
      phi::DenseTensor tmp(*dx);
      keep_dim = true;
      auto dtype = ConvertToNpuDtype(dx->dtype());
      auto axis = phi::IntArray(axes);
      EXEC_NPU_CMD(
          aclnnReduceSum, dev_ctx, *tmp_dout, axis, keep_dim, dtype, tmp);
    } else {
      TensorCopy(dev_ctx, *tmp_dout, false, dx);
    }
  }
  if (dy) {
    dev_ctx.template Alloc<T>(dy);
    // For dy
    // stage 1
    auto reduce_ndim = dout.dims().size() - dy->dims().size();
    std::vector<int> axes;
    for (auto i = 0; i < reduce_ndim; ++i) {
      axes.push_back(i);
    }
    phi::DenseTensor* tmp_dout = const_cast<phi::DenseTensor*>(&dout);
    phi::DenseTensor reduced_dy;
    phi::DenseTensor reduced_dout;

    if (axes.size() != 0) {
      std::vector<int64_t> reduced_dout_dims;
      for (auto i = reduce_ndim; i < dout.dims().size(); ++i) {
        reduced_dout_dims.push_back(dout.dims()[i]);
      }

      phi::DenseTensorMeta reduced_dout_meta = {
          dy->dtype(), phi::make_ddim(reduced_dout_dims)};
      reduced_dout.set_meta(reduced_dout_meta);
      dev_ctx.template Alloc<T>(&reduced_dout);

      keep_dim = false;
      auto dtype = ConvertToNpuDtype(reduced_dout.dtype());
      auto axis = phi::IntArray(axes);
      EXEC_NPU_CMD(
          aclnnReduceSum, dev_ctx, dout, axis, keep_dim, dtype, reduced_dout);
      tmp_dout = &reduced_dout;
    }

    // stage 2
    axes.clear();
    phi::DenseTensor* tmp_dy = tmp_dout;
    for (auto i = 0; i < dy->dims().size(); ++i) {
      if (dy->dims()[i] == 1) {
        axes.push_back(i);
      }
    }
    if (axes.size() != 0) {
      phi::DenseTensorMeta reduced_dy_meta = {dy->dtype(), dy->dims()};
      reduced_dy.set_meta(reduced_dy_meta);
      dev_ctx.template Alloc<T>(&reduced_dy);

      keep_dim = true;
      auto dtype = ConvertToNpuDtype(reduced_dy.dtype());
      auto axis = phi::IntArray(axes);
      EXEC_NPU_CMD(aclnnReduceSum,
                   dev_ctx,
                   *tmp_dout,
                   axis,
                   keep_dim,
                   dtype,
                   reduced_dy);

      tmp_dy = &reduced_dy;
    }

    // stage 3, negative
    EXEC_NPU_CMD(aclnnNeg, dev_ctx, *tmp_dy, *dy);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(subtract_raw,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SubtractRawKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(subtract,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SubtractKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(subtract_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SubtractGradKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          double) {}
