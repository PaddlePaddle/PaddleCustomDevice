/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <memory>

#include "npu_enforce.h"
#include "npu_op_runner.h"
#include "runtime.h"

namespace custom_kernel {

/**
 * CPU -> NPU
 * NPU -> CPU
 * NPU -> NPU
*/
template <typename Context>
inline void TensorCopy(const Context& dev_ctx,
                       const phi::DenseTensor& src,
                       bool blocking,
                       phi::DenseTensor* dst,
                       const phi::Place& dst_place = phi::CustomPlace()) {
  auto* src_ptr = src.data();
  const auto& src_place = src.place();
  auto dst_place_ = dst_place;
  if (dst_place_.GetType() != phi::AllocationType::CPU) {
    dst_place_ = dev_ctx.GetPlace();
  }

  VLOG(3) << "TensorCopy " << src.dims() << " from " << src_place << " to "
          << dst_place_;

  dst->Resize(src.dims());
  auto dst_ptr = dst->mutable_data(dst_place_, src.dtype());

  if (src_ptr == dst_ptr) {
    VLOG(3) << "Skip copy the same data async from " << src_place << " to "
            << src_place;
    return;
  }
  VLOG(4) << "src:" << src_ptr << ", dst:" << dst_ptr;

  C_Stream stream = static_cast<C_Stream>(dev_ctx.stream());

  auto size = src.numel() * paddle::experimental::SizeOf(src.dtype());

  if (src_place.GetType() == phi::AllocationType::CPU &&
      dst_place_.GetType() == phi::AllocationType::CUSTOM) {
    if (blocking) {
      MemCpyH2D(nullptr, dst_ptr, src_ptr, size);
    } else {
      AsyncMemCpyH2D(nullptr, stream, dst_ptr, src_ptr, size);
    }
  } else if (src_place.GetType() == phi::AllocationType::CUSTOM &&
             dst_place_.GetType() == phi::AllocationType::CPU) {
    if (blocking) {
      MemCpyD2H(nullptr, dst_ptr, src_ptr, size);
    } else {
      AsyncMemCpyD2H(nullptr, stream, dst_ptr, src_ptr, size);
    }
  } else if (src_place.GetType() == phi::AllocationType::CUSTOM &&
             dst_place_.GetType() == phi::AllocationType::CUSTOM) {
    if (src_place.GetDeviceType() == dst_place_.GetDeviceType()) {
      if (src_place.GetDeviceId() == dst_place_.GetDeviceId()) {
        if (blocking) {
          MemCpyD2D(nullptr, dst_ptr, src_ptr, size);
        } else {
          AsyncMemCpyD2D(nullptr, stream, dst_ptr, src_ptr, size);
        }
      } else {
      }
    } else {
    }
  } else {
  }
}

/**
 * CPU -> NPU
*/
template <typename T>
inline void TensorFromVector(const std::vector<T>& src,
                             const phi::CustomContext& dev_ctx,
                             phi::DenseTensor* dst) {
  auto dst_place = dev_ctx.GetPlace();
  auto src_ptr = static_cast<const void*>(src.data());
  dst->Resize({static_cast<int64_t>(src.size())});
  auto dst_ptr = static_cast<void*>(dev_ctx.template Alloc<T>(dst));
  auto size = src.size() * sizeof(T);
  if (UNLIKELY(size == 0)) return;

  if (dst_place.GetType() == phi::AllocationType::CUSTOM) {
    AsyncMemCpyH2D(nullptr,
                   static_cast<C_Stream>(dev_ctx.stream()),
                   dst_ptr,
                   src_ptr,
                   size);
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "TensorFromVector on %s is not supported.", dst_place));
  }
}

template <>
inline void TensorFromVector<bool>(const std::vector<bool>& src,
                                   const phi::CustomContext& dev_ctx,
                                   phi::DenseTensor* dst) {
  auto dst_place = dev_ctx.GetPlace();
  PADDLE_THROW(phi::errors::Unimplemented(
      "TensorFromVector on %s is not supported.", dst_place));
}

/**
 * CPU -> CPU
 * CPU -> NPU
*/
template <typename T>
inline void TensorFromVector(const std::vector<T>& src,
                             const phi::CPUContext& dev_ctx,
                             phi::DenseTensor* dst) {
  auto dst_place = dev_ctx.GetPlace();
  auto src_ptr = static_cast<const void*>(src.data());
  auto dst_ptr = dst->mutable_data<T>({src.size()}, dst_place);
  auto size = src.size() * sizeof(T);
  if (UNLIKELY(size == 0)) return;

  if (dst_place.GetType() == phi::AllocationType::CPU) {
    VLOG(4) << "src_ptr: " << src_ptr << ", dst_ptr: " << dst_ptr
            << ", size: " << size;
    std::memcpy(dst_ptr, src_ptr, size);
  } else if (dst_place.GetType() == phi::AllocationType::CUSTOM) {
    MemCpyH2D(nullptr, dst_ptr, src_ptr, size);
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "TensorFromVector on %s is not supported.", dst_place));
  }
}

template <>
inline void TensorFromVector<bool>(const std::vector<bool>& src,
                                   const phi::CPUContext& dev_ctx,
                                   phi::DenseTensor* dst) {
  auto dst_place = dev_ctx.GetPlace();
  PADDLE_THROW(phi::errors::Unimplemented(
      "TensorFromVector on %s is not supported.", dst_place));
}

/**
 * CPU -> NPU
*/
template <typename T>
inline void FillNpuTensorWithConstant(phi::DenseTensor* dst,
                                      const phi::CustomContext& dev_ctx,
                                      T val) {
  int numel = dst->numel();
  std::vector<T> src(numel, static_cast<T>(val));
  TensorFromVector(src, dev_ctx, dst);
}

// src - broadcast -> transformed_src
template <typename T, typename Context>
inline void NpuBroadcast(const Context& dev_ctx,
                         const phi::DenseTensor* src,
                         int axis,
                         const phi::DDim& dst_dims,
                         phi::DenseTensor* transformed_src) {
  auto stream = dev_ctx.stream();

  // 1. expand the axis with dim 1
  auto src_dims = src->dims();
  phi::DenseTensor tmp_src;
  tmp_src.ShareDataWith(*src);
  tmp_src.Resize(src_dims);
  for (int i = 0; i < src_dims.size(); ++i) {
    if (src_dims[i] == 1 && dst_dims[i + axis] > 1) {
      phi::DenseTensor tmp_tensor;
      auto tmp_tensor_dims = tmp_src.dims();
      tmp_tensor_dims[i] = dst_dims[i + axis];
      tmp_tensor.Resize(tmp_tensor_dims);
      dev_ctx.template Alloc<T>(&tmp_tensor);
      const auto& runner =
          NpuOpRunner("TileWithAxis",
                      {tmp_src},
                      {tmp_tensor},
                      {{"axis", static_cast<int64_t>(i)},
                       {"tiles", static_cast<int64_t>(dst_dims[i + axis])}});
      runner.Run(stream);
      tmp_src.ShareDataWith(tmp_tensor);
      tmp_src.Resize(tmp_tensor_dims);
    }
  }

  // 2.expand the ahead axis
  auto prev = phi::product(phi::slice_ddim(dst_dims, 0, axis));
  if (prev > 1) {
    phi::DenseTensor tmp_tensor;
    auto tmp_tensor_dims = phi::slice_ddim(dst_dims, 0, axis + src_dims.size());
    tmp_tensor.Resize(tmp_tensor_dims);
    dev_ctx.template Alloc<T>(&tmp_tensor);
    const auto& runner =
        NpuOpRunner("ExpandD",
                    {tmp_src},
                    {tmp_tensor},
                    {{"shape", phi::vectorize<int64_t>(tmp_tensor_dims)}});
    runner.Run(stream);
    tmp_src.ShareDataWith(tmp_tensor);
    tmp_src.Resize(tmp_tensor_dims);
  } else {
    tmp_src.Resize(phi::slice_ddim(dst_dims, 0, axis + src_dims.size()));
  }

  // 3.expand the tail axis
  auto post = phi::product(
      phi::slice_ddim(dst_dims, axis + src_dims.size(), dst_dims.size()));
  if (post > 1) {
    auto src_dims_vec = phi::vectorize<int>(tmp_src.dims());
    src_dims_vec.push_back(1);
    tmp_src.Resize(phi::make_ddim(src_dims_vec));

    phi::DenseTensor tmp_tensor;
    tmp_tensor.Resize(dst_dims);
    dev_ctx.template Alloc<T>(&tmp_tensor);
    const auto& runner =
        NpuOpRunner("TileWithAxis",
                    {tmp_src},
                    {tmp_tensor},
                    {{"axis", static_cast<int64_t>(axis + src_dims.size())},
                     {"tiles", static_cast<int64_t>(post)}});
    runner.Run(stream);
    tmp_src.ShareDataWith(tmp_tensor);
  }
  tmp_src.Resize(dst_dims);
  TensorCopy(dev_ctx, tmp_src, false, transformed_src);
}

template <typename T, typename Context>
inline void NpuElementWiseOpBroadcast(const Context& dev_ctx,
                                      const phi::DenseTensor* x,
                                      const phi::DenseTensor* y,
                                      int axis,
                                      phi::DenseTensor* transformed_x,
                                      phi::DenseTensor* transformed_y) {
  auto x_dims = x->dims();
  auto y_dims = y->dims();
  bool is_xsize_larger = true;
  int max_dim = x_dims.size();
  std::vector<int> dst_dims_vec = phi::vectorize<int>(x_dims);

  if (x_dims.size() < y_dims.size()) {
    is_xsize_larger = false;
    max_dim = y_dims.size();
    dst_dims_vec = phi::vectorize<int>(y_dims);
  }

  axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
  int x_axis = is_xsize_larger ? 0 : axis;
  int y_axis = is_xsize_larger ? axis : 0;

  PADDLE_ENFORCE_GE(
      axis,
      0,
      phi::errors::InvalidArgument(
          "Axis should be great than or equal to 0, but received axis is %d.",
          axis));
  PADDLE_ENFORCE_LT(axis,
                    max_dim,
                    phi::errors::InvalidArgument(
                        "Axis should be less than %d, but received axis is %d.",
                        max_dim,
                        axis));

  for (int i = 0; i < x_dims.size(); ++i) {
    dst_dims_vec[i + x_axis] =
        std::max(dst_dims_vec[i + x_axis], static_cast<int>(x_dims[i]));
  }
  for (int i = 0; i < y_dims.size(); ++i) {
    dst_dims_vec[i + y_axis] =
        std::max(dst_dims_vec[i + y_axis], static_cast<int>(y_dims[i]));
  }

  auto dst_dims = phi::make_ddim(dst_dims_vec);
  NpuBroadcast<T>(dev_ctx, x, x_axis, dst_dims, transformed_x);
  NpuBroadcast<T>(dev_ctx, y, y_axis, dst_dims, transformed_y);
}

static inline int CanonicalAxis(const int axis, const int rank) {
  if (axis < 0) {
    return axis + rank;
  }
  return axis;
}

inline phi::DataLayout StringToDataLayout(const std::string& str) {
  std::string s(str);
  for (size_t i = 0; i < s.size(); ++i) {
    s[i] = toupper(s[i]);
  }

  if (s == "NHWC") {
    return phi::DataLayout::kNHWC;
  } else if (s == "NCHW") {
    return phi::DataLayout::kNCHW;
  } else if (s == "ANYLAYOUT") {
    return phi::DataLayout::kAnyLayout;
  } else if (s == "MKLDNNLAYOUT") {
    return phi::DataLayout::kMKLDNN;
  } else if (s == "SPARSE_COO") {
    return phi::DataLayout::SPARSE_COO;
  } else if (s == "SPARSE_CSR") {
    return phi::DataLayout::SPARSE_CSR;
  } else {
  }
}
}  // namespace custom_kernel
