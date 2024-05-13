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
#include "paddle/phi/kernels/funcs/tensor_formatter.h"

namespace custom_kernel {

inline phi::DDim GetDimsWithAxis(const phi::DDim& x_dims,
                                 const phi::DDim& y_dims,
                                 const int axis) {
  std::vector<int64_t> y_shape(x_dims.size(), 1);
  for (int i = 0; i < axis; ++i) {
    y_shape[i] = 1;
  }
  for (int i = 0; i < y_dims.size(); ++i) {
    bool is_common_boardcast = x_dims[i + axis] == y_dims[i]
                                   ? true
                                   : (y_dims[i] == 1 || x_dims[i + axis] == 1);
    PADDLE_ENFORCE_EQ(is_common_boardcast,
                      true,
                      phi::errors::InvalidArgument(
                          "Broadcast dimension mismatch. Operands "
                          "could not be broadcast together with the shape of "
                          "X = [%s] and the shape of Y = [%s]. Received [%d] "
                          "in X is not equal to [%d] in Y.",
                          x_dims,
                          y_dims,
                          x_dims[i + axis],
                          y_dims[i]));
    y_shape[i + axis] = y_dims[i];
  }
  for (int i = axis + y_dims.size(); i < x_dims.size(); ++i) {
    y_shape[i] = 1;
  }
  return phi::make_ddim(y_shape);
}

template <typename T, typename Context>
void AddRawKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& y,
                  int axis,
                  phi::DenseTensor* out) {
  if (x.storage_properties_initialized()) {
    auto npu_properties = x.storage_properties<phi::NPUStorageProperties>();
    int64_t storage_format = npu_properties.storage_format;
    AllocNPUTensor<T>(dev_ctx, aclFormat(storage_format), out);
  } else {
    dev_ctx.template Alloc<T>(out);
  }

  auto stream = dev_ctx.stream();

  bool direct_compute = false;
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);

  phi::DenseTensor x_tensor(x), y_tensor(y);
  if (x_dims.size() >= y_dims.size()) {
    y_tensor.Resize(GetDimsWithAxis(x_dims, y_dims, axis));
  } else {
    x_tensor.Resize(GetDimsWithAxis(y_dims, x_dims, axis));
  }
  const auto& runner = NpuOpRunner("Add", {x_tensor, y_tensor}, {*out}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void AclopAddKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    phi::DenseTensor* out) {
  int axis = -1;
  custom_kernel::AddRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void AddKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::DenseTensor& y,
               phi::DenseTensor* out) {
  DO_COMPATIBILITY(
      aclnnAdd,
      (custom_kernel::AclopAddKernel<T, Context>(dev_ctx, x, y, out)));
  if (x.storage_properties_initialized()) {
    custom_kernel::AclopAddKernel<T, Context>(dev_ctx, x, y, out);
    return;
  }
  phi::Scalar alpha = 1;
  if (x.dtype() == phi::DataType::FLOAT32 &&
      (y.dtype() == phi::DataType::BFLOAT16 ||
       y.dtype() == phi::DataType::FLOAT16)) {
    dev_ctx.template Alloc<float>(out);
  } else {
    dev_ctx.template Alloc<T>(out);
  }
  EXEC_NPU_CMD(aclnnAdd, dev_ctx, x, y, alpha, *out);
}

template <typename T, typename Context>
void AddGradAclop(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& y,
                  const phi::DenseTensor& dout,
                  int axis,
                  phi::DenseTensor* dx,
                  phi::DenseTensor* dy) {
  auto stream = dev_ctx.stream();

  axis = (axis == -1 ? std::abs(x.dims().size() - y.dims().size()) : axis);

  if (dx) {
    if (dout.storage_properties_initialized()) {
      auto npu_properties =
          dout.storage_properties<phi::NPUStorageProperties>();
      int64_t storage_format = npu_properties.storage_format;
      AllocNPUTensor<T>(dev_ctx, aclFormat(storage_format), dx);
    } else {
      dev_ctx.template Alloc<T>(dx);
    }

    if (dx->dims() != dout.dims()) {
      std::vector<int> dst_dims_vec;
      std::vector<int> reduce_axes;
      auto src_dims = dx->dims();
      auto dout_dims = dout.dims();

      int src_axis = (src_dims.size() < dout_dims.size() ? axis : 0);
      for (int ax = 0; ax < dout_dims.size(); ++ax) {
        if ((ax < src_axis || ax >= src_axis + src_dims.size()) ||
            (dout_dims[ax] > 1 && src_dims[ax - src_axis] == 1)) {
          reduce_axes.push_back(ax);
        } else {
          dst_dims_vec.push_back(dout_dims[ax]);
        }
      }
      if (!reduce_axes.empty()) {
        phi::DenseTensor tmp(*dx);
        tmp.Resize(phi::make_ddim(dst_dims_vec));
        const auto& runner =
            NpuOpRunner("ReduceSumD",
                        {dout},
                        {tmp},
                        {{"axes", reduce_axes}, {"keep_dims", false}});
        runner.Run(stream);
      }
    } else {
      TensorCopy(dev_ctx, dout, false, dx);
    }
  }
  if (dy) {
    if (dout.storage_properties_initialized()) {
      auto npu_properties =
          dout.storage_properties<phi::NPUStorageProperties>();
      int64_t storage_format = npu_properties.storage_format;
      AllocNPUTensor<T>(dev_ctx, aclFormat(storage_format), dy);
    } else {
      dev_ctx.template Alloc<T>(dy);
    }

    if (dy->dims() != dout.dims()) {
      std::vector<int> dst_dims_vec;
      std::vector<int> reduce_axes;
      auto src_dims = dy->dims();
      auto dout_dims = dout.dims();

      int src_axis = (src_dims.size() < dout_dims.size() ? axis : 0);
      for (int ax = 0; ax < dout_dims.size(); ++ax) {
        if ((ax < src_axis || ax >= src_axis + src_dims.size()) ||
            (dout_dims[ax] > 1 && src_dims[ax - src_axis] == 1)) {
          reduce_axes.push_back(ax);
        } else {
          dst_dims_vec.push_back(dout_dims[ax]);
        }
      }
      if (!reduce_axes.empty()) {
        phi::DenseTensor tmp(*dy);
        tmp.Resize(phi::make_ddim(dst_dims_vec));
        NpuOpRunner runner;
        runner.SetType("ReduceSum");
        runner.AddInput(dout);
        runner.AddInput(dev_ctx, std::move(reduce_axes));
        runner.AddOutput(tmp);
        runner.AddAttr("keep_dims", false);
        runner.Run(stream);
      }
    } else {
      TensorCopy(dev_ctx, dout, false, dy);
    }
  }
}

template <typename T, typename Context>
void AddGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& y,
                   const phi::DenseTensor& dout,
                   int axis,
                   phi::DenseTensor* dx,
                   phi::DenseTensor* dy) {
  DO_COMPATIBILITY(aclnnReduceSum,
                   (custom_kernel::AddGradAclop<T, Context>(
                       dev_ctx, x, y, dout, axis, dx, dy)));
  // aclnnReduceSum will change the dout after calculating dx
  // so we copy dout for later dy calculation
  phi::DenseTensor douty;
  phi::DenseTensorMeta douty_meta = {dout.dtype(), dout.dims()};
  douty.set_meta(douty_meta);
  dev_ctx.template Alloc<T>(&douty);
  TensorCopy(dev_ctx, dout, false, &douty);
  axis = (axis == -1 ? std::abs(x.dims().size() - y.dims().size()) : axis);
  if (dx) {
    dev_ctx.template Alloc<T>(dx);
    if (dx->dims() != dout.dims()) {
      std::vector<int64_t> dst_dims_vec;
      std::vector<int64_t> reduce_axes;
      auto src_dims = dx->dims();
      auto dout_dims = dout.dims();
      int src_axis = (src_dims.size() < dout_dims.size() ? axis : 0);
      for (int ax = 0; ax < dout_dims.size(); ++ax) {
        if ((ax < src_axis || ax >= src_axis + src_dims.size()) ||
            (dout_dims[ax] > 1 && src_dims[ax - src_axis] == 1)) {
          reduce_axes.push_back(ax);
        } else {
          dst_dims_vec.push_back(dout_dims[ax]);
        }
      }
      if (!reduce_axes.empty()) {
        phi::DenseTensor tmp;
        tmp.Resize(phi::make_ddim(dst_dims_vec));
        dev_ctx.template Alloc<T>(&tmp);
        bool keep_dims = false;
        auto dtype = ConvertToNpuDtype(dx->dtype());
        EXEC_NPU_CMD(
            aclnnReduceSum, dev_ctx, dout, reduce_axes, keep_dims, dtype, tmp);
        tmp.Resize(dx->dims());
        TensorCopy(dev_ctx, tmp, false, dx);
      }
    } else {
      TensorCopy(dev_ctx, dout, false, dx);
    }
  }
  if (dy) {
    dev_ctx.template Alloc<T>(dy);
    if (dy->dims() != dout.dims()) {
      std::vector<int64_t> dst_dims_vec;
      std::vector<int64_t> reduce_axes;
      auto src_dims = dy->dims();
      auto dout_dims = dout.dims();
      int src_axis = (src_dims.size() < dout_dims.size() ? axis : 0);
      for (int ax = 0; ax < dout_dims.size(); ++ax) {
        if ((ax < src_axis || ax >= src_axis + src_dims.size()) ||
            (dout_dims[ax] > 1 && src_dims[ax - src_axis] == 1)) {
          reduce_axes.push_back(ax);
        } else {
          dst_dims_vec.push_back(dout_dims[ax]);
        }
      }
      if (!reduce_axes.empty()) {
        phi::DenseTensor tmp;
        tmp.Resize(phi::make_ddim(dst_dims_vec));
        dev_ctx.template Alloc<T>(&tmp);
        bool keep_dims = false;
        auto dtype = ConvertToNpuDtype(dy->dtype());
        EXEC_NPU_CMD(
            aclnnReduceSum, dev_ctx, douty, reduce_axes, keep_dims, dtype, tmp);
        tmp.Resize(dy->dims());
        TensorCopy(dev_ctx, tmp, false, dy);
      }
    } else {
      TensorCopy(dev_ctx, douty, false, dy);
    }
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(add_raw,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::AddRawKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(add,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::AddKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(add_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::AddGradKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          double) {}
