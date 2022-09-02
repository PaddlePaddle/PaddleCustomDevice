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

#include "kernels/funcs/mlu_baseop.h"

namespace custom_kernel {

/**
 * CPU -> MLU
 * MLU -> CPU
 * MLU -> MLU
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
  void* dst_ptr = nullptr;
  if (dst_place_.GetType() != phi::AllocationType::CPU) {
    dst_ptr = dev_ctx.Alloc(dst, src.dtype());
  } else {
    dst_ptr = dev_ctx.HostAlloc(dst, src.dtype());
  }

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
 * CPU -> MLU
 */
template <typename T>
inline void TensorFromVector(const phi::CustomContext& ctx,
                             const std::vector<T>& src,
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
inline void TensorFromVector<bool>(const phi::CustomContext& ctx,
                                   const std::vector<bool>& src,
                                   const phi::CustomContext& dev_ctx,
                                   phi::DenseTensor* dst) {
  // vector<bool> has no data() member, use array instead.
  // See details:
  // https://stackoverflow.com/questions/46115669/why-does-stdvectorbool-have-no-data/46115714
  bool* array = new bool[src.size()];
  for (unsigned int i = 0; i < src.size(); i++) {
    array[i] = static_cast<bool>(src[i]);
  }

  auto dst_place = dev_ctx.GetPlace();
  auto src_ptr = static_cast<const void*>(array);
  dst->Resize({static_cast<int64_t>(src.size())});
  auto dst_ptr = static_cast<void*>(dev_ctx.template Alloc<bool>(dst));
  auto size = src.size() * sizeof(bool);
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
  delete[] array;
}

/**
 * CPU -> CPU
 * CPU -> MLU
 */
template <typename T>
inline void TensorFromVector(const phi::CustomContext& ctx,
                             const std::vector<T>& src,
                             const phi::CPUContext& dev_ctx,
                             phi::DenseTensor* dst) {
  auto dst_place = dev_ctx.GetPlace();
  auto src_ptr = static_cast<const void*>(src.data());
  dst->Resize({src.size()});
  auto dst_ptr = ctx.template HostAlloc<T>(dst);
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
inline void TensorFromVector<bool>(const phi::CustomContext& ctx,
                                   const std::vector<bool>& src,
                                   const phi::CPUContext& dev_ctx,
                                   phi::DenseTensor* dst) {
  auto dst_place = dev_ctx.GetPlace();
  PADDLE_THROW(phi::errors::Unimplemented(
      "TensorFromVector on %s is not supported.", dst_place));
}

template <typename T>
void TensorFromArray(const phi::CustomContext& ctx,
                     const T* src,
                     const size_t& array_size,
                     const phi::CustomContext& dev_ctx,
                     phi::DenseTensor* dst) {
  auto dst_place = dev_ctx.GetPlace();
  auto src_ptr = static_cast<const void*>(src);
  dst->Resize({static_cast<int64_t>(array_size)});
  auto dst_ptr = static_cast<void*>(dev_ctx.template Alloc<T>(dst));
  auto size = array_size * sizeof(T);

  if (dst_place.GetType() == phi::AllocationType::CUSTOM) {
    AsyncMemCpyH2D(nullptr,
                   static_cast<C_Stream>(dev_ctx.stream()),
                   dst_ptr,
                   src_ptr,
                   size);
  } else {  // NOLINT
    PADDLE_THROW(phi::errors::Unimplemented(
        "TensorFromArray on %s is not supported.", dst_place));
  }
}

/**
 * MLU -> CPU
 */
template <typename T>
inline void TensorToVector(const phi::CustomContext& ctx,
                           const phi::DenseTensor& src,
                           const phi::CustomContext& dev_ctx,
                           std::vector<T>* dst) {
  auto src_ptr = static_cast<const void*>(src.data<T>());
  auto size = src.numel() * sizeof(T);

  dst->resize(src.numel());
  auto dst_ptr = static_cast<void*>(dst->data());

  auto src_place = src.place();

  if (src_place.GetType() == phi::AllocationType::CUSTOM) {
    MemCpyD2H(nullptr, dst_ptr, src_ptr, size);
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "TensorToVector on %s is not supported.", src_place));
  }
}

template <>
inline void TensorToVector<bool>(const phi::CustomContext& ctx,
                                 const phi::DenseTensor& src,
                                 const phi::CustomContext& dev_ctx,
                                 std::vector<bool>* dst) {
  auto src_ptr = static_cast<const void*>(src.data<bool>());
  auto size = src.numel() * sizeof(bool);

  bool* array = new bool[src.numel()];

  phi::CPUPlace dst_place;
  dst->resize(src.numel());
  auto dst_ptr = static_cast<void*>(array);

  auto src_place = src.place();
  if (src_place.GetType() == phi::AllocationType::CUSTOM) {
    MemCpyD2H(nullptr, dst_ptr, src_ptr, size);
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "TensorToVector on %s is not supported.", src_place));
  }
  for (unsigned int i = 0; i < src.numel(); i++) {
    (*dst)[i] = static_cast<bool>(array[i]);
  }
  delete[] array;
}

static inline int CanonicalAxis(const int axis, const int rank) {
  if (axis < 0) {
    return axis + rank;
  }
  return axis;
}

static inline int SizeToAxis(const int axis, phi::DDim dims) {
  int size = 1;
  for (int i = 0; i < axis; i++) {
    size *= dims[i];
  }
  return size;
}

static inline int SizeOutAxis(const int axis, phi::DDim dims) {
  int size = 1;
  for (int i = axis + 1; i < dims.size(); i++) {
    size *= dims[i];
  }
  return size;
}

}  // namespace custom_kernel
