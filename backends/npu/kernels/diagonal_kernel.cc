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

template <typename T>
std::vector<T> ComputeDimStride(const std::vector<T> dim) {
  size_t dim_size = dim.size();
  std::vector<T> dim_strides;
  dim_strides.resize(dim_size);
  for (size_t i = 0; i < dim_size - 1; i++) {
    size_t temp_stride = 1;
    for (size_t j = i + 1; j < dim_size; j++) {
      temp_stride = temp_stride * dim[j];
    }
    dim_strides[i] = temp_stride;
  }
  dim_strides[dim_size - 1] = 1;
  return dim_strides;
}

template <typename T, typename Context>
void DiagonalKernelImpl(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        int offset,
                        int axis1,
                        int axis2,
                        phi::DenseTensor* out) {
  auto input_dim = phi::vectorize(x.dims());
  auto input_dim_size = input_dim.size();
  auto output_dim = phi::vectorize(out->dims());
  auto output_dim_size = output_dim.size();

  // copy to cpu
  phi::DenseTensor output_cpu_t;
  output_cpu_t.Resize(out->dims());
  auto output_data = dev_ctx.template HostAlloc<T>(&output_cpu_t);

  std::vector<T> input_data;
  TensorToVector(dev_ctx, x, dev_ctx, &input_data);

  const int64_t offset_ = offset;
  int64_t axis1_ = axis1 < 0 ? input_dim_size + axis1 : axis1;
  int64_t axis2_ = axis2 < 0 ? input_dim_size + axis2 : axis2;

  std::vector<int64_t> input_stride = ComputeDimStride(input_dim);
  std::vector<int64_t> output_stride = ComputeDimStride(output_dim);

  int64_t out_numel = out->numel();
  for (int64_t idx = 0; idx < out_numel; idx++) {
    std::vector<int64_t> idx_dim(output_dim_size);
    int64_t temp = 0;
    for (size_t i = 0; i < output_dim_size; i++) {
      idx_dim[i] = (idx - temp) / output_stride[i];
      temp = temp + idx_dim[i] * output_stride[i];
    }
    int64_t tmp = idx_dim[output_dim_size - 1];
    std::vector<int64_t> list;
    list.clear();
    int64_t l = std::min(axis1_, axis2_);
    int64_t r = std::max(axis1_, axis2_);
    for (size_t j = 0; j < output_dim_size - 1; j++) {
      list.push_back(idx_dim[j]);
    }
    if (offset_ == 0) {
      list.insert(list.begin() + l, tmp);
      list.insert(list.begin() + r, tmp);
    } else if (offset_ > 0) {
      if (axis1_ < axis2_) {
        list.insert(list.begin() + l, tmp);
        list.insert(list.begin() + r, tmp + offset_);
      } else {
        list.insert(list.begin() + l, tmp + offset_);
        list.insert(list.begin() + r, tmp);
      }
    } else if (offset_ < 0) {
      if (axis1_ < axis2_) {
        list.insert(list.begin() + l, tmp - offset_);
        list.insert(list.begin() + r, tmp);
      } else {
        list.insert(list.begin() + l, tmp);
        list.insert(list.begin() + r, tmp - offset_);
      }
    }

    int64_t input_offset = 0;
    for (size_t i = 0; i < input_dim_size; i++) {
      input_offset = input_offset + list[i] * input_stride[i];
    }
    output_data[idx] = input_data[input_offset];
  }
  dev_ctx.template Alloc<T>(out);
  TensorCopy(dev_ctx, output_cpu_t, true, out);
}

template <typename T, typename Context>
void DiagonalKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    int offset,
                    int axis1,
                    int axis2,
                    phi::DenseTensor* out) {
  auto stream = dev_ctx.stream();
  if (x.dtype() == phi::DataType::FLOAT16) {
    phi::DenseTensor tmp_x, tmp_out;
    tmp_x.Resize(x.dims());
    dev_ctx.template Alloc<float>(&tmp_x);
    tmp_out.Resize(out->dims());
    dev_ctx.template Alloc<float>(&tmp_out);
    const auto& cast_runner1 =
        NpuOpRunner("Cast", {x}, {tmp_x}, {{"dst_type", ACL_FLOAT}});
    cast_runner1.Run(stream);
    DiagonalKernelImpl<float, Context>(
        dev_ctx, tmp_x, offset, axis1, axis2, &tmp_out);

    dev_ctx.template Alloc<T>(out);
    const auto& cast_runner2 =
        NpuOpRunner("Cast", {tmp_out}, {*out}, {{"dst_type", ACL_FLOAT16}});
    cast_runner2.Run(stream);
  } else {
    DiagonalKernelImpl<T, Context>(dev_ctx, x, offset, axis1, axis2, out);
  }
}

template <typename T, typename Context>
void DiagonalGradKernelImpl(const Context& dev_ctx,
                            const phi::DenseTensor& x UNUSED,
                            const phi::DenseTensor& out_grad,
                            int offset,
                            int axis1,
                            int axis2,
                            phi::DenseTensor* in_grad) {
  std::vector<T> dout_data;
  TensorToVector(dev_ctx, out_grad, dev_ctx, &dout_data);
  auto dout_dim = phi::vectorize(out_grad.dims());

  phi::DenseTensor dx;
  dx.Resize(in_grad->dims());
  auto dx_data = dev_ctx.template HostAlloc<T>(&dx);

  auto dx_dim = phi::vectorize(in_grad->dims());
  auto dx_dim_size = dx_dim.size();

  const int64_t offset_ = offset;
  int64_t axis1_ = axis1 < 0 ? dx_dim_size + axis1 : axis1;
  int64_t axis2_ = axis2 < 0 ? dx_dim_size + axis2 : axis2;

  std::vector<int64_t> dout_stride = ComputeDimStride(dout_dim);
  std::vector<int64_t> dx_stride = ComputeDimStride(dx_dim);

  int64_t numel = dx.numel();

  for (int64_t idx = 0; idx < numel; idx++) {
    std::vector<int64_t> idx_dim(dx_dim_size);
    int64_t temp = 0;
    for (size_t i = 0; i < dx_dim_size; i++) {
      idx_dim[i] = (idx - temp) / dx_stride[i];
      temp = temp + idx_dim[i] * dx_stride[i];
    }

    int64_t axis1_dim = idx_dim[axis1_];
    int64_t axis2_dim = idx_dim[axis2_];

    idx_dim.erase(idx_dim.begin() + std::max(axis1_, axis2_));
    idx_dim.erase(idx_dim.begin() + std::min(axis1_, axis2_));

    bool flag = false;
    if (offset_ == 0 && axis1_dim == axis2_dim) {
      idx_dim.push_back(axis1_dim);
      flag = true;
    } else if (offset_ > 0 && (axis1_dim + offset_) == axis2_dim) {
      idx_dim.push_back(axis1_dim);
      flag = true;
    } else if (offset_ < 0 && (axis1_dim + offset_) == axis2_dim) {
      idx_dim.push_back(axis2_dim);
      flag = true;
    }
    if (flag) {
      int64_t idx_output = 0;
      for (size_t i = 0; i < idx_dim.size(); i++) {
        idx_output = idx_output + idx_dim[i] * dout_stride[i];
      }
      dx_data[idx] = dout_data[idx_output];
    } else {
      dx_data[idx] = static_cast<T>(0);
    }
  }
  dev_ctx.template Alloc<T>(in_grad);
  TensorCopy(dev_ctx, dx, true, in_grad);
}

template <typename T, typename Context>
void DiagonalGradKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x UNUSED,
                        const phi::DenseTensor& out_grad,
                        int offset,
                        int axis1,
                        int axis2,
                        phi::DenseTensor* in_grad) {
  auto stream = dev_ctx.stream();
  if (out_grad.dtype() == phi::DataType::FLOAT16) {
    phi::DenseTensor tmp_out_grad, tmp_in_grad;
    tmp_out_grad.Resize(out_grad.dims());
    dev_ctx.template Alloc<float>(&tmp_out_grad);
    tmp_in_grad.Resize(in_grad->dims());
    dev_ctx.template Alloc<float>(&tmp_in_grad);
    const auto& cast_runner1 = NpuOpRunner(
        "Cast", {out_grad}, {tmp_out_grad}, {{"dst_type", ACL_FLOAT}});
    cast_runner1.Run(stream);
    DiagonalGradKernelImpl<float, Context>(
        dev_ctx, x, tmp_out_grad, offset, axis1, axis2, &tmp_in_grad);

    dev_ctx.template Alloc<T>(in_grad);
    const auto& cast_runner2 = NpuOpRunner(
        "Cast", {tmp_in_grad}, {*in_grad}, {{"dst_type", ACL_FLOAT16}});
    cast_runner2.Run(stream);
  } else {
    DiagonalGradKernelImpl<T, Context>(
        dev_ctx, x, out_grad, offset, axis1, axis2, in_grad);
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(diagonal,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::DiagonalKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          bool,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(diagonal_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::DiagonalGradKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          bool,
                          phi::dtype::float16) {}
