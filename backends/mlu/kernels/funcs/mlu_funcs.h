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

  auto size = src.numel() * phi::SizeOf(src.dtype());
  if (UNLIKELY(size) == 0) {
    return;
  }

  if (src_place.GetType() == phi::AllocationType::CPU &&
      dst_place_.GetType() == phi::AllocationType::CUSTOM) {
    if (blocking) dev_ctx.Wait();
    AsyncMemCpyH2D(nullptr, stream, dst_ptr, src_ptr, size);
  } else if (src_place.GetType() == phi::AllocationType::CUSTOM &&
             dst_place_.GetType() == phi::AllocationType::CPU) {
    AsyncMemCpyD2H(nullptr, stream, dst_ptr, src_ptr, size);
    if (blocking) dev_ctx.Wait();
  } else if (src_place.GetType() == phi::AllocationType::CUSTOM &&
             dst_place_.GetType() == phi::AllocationType::CUSTOM) {
    if (src_place.GetDeviceType() == dst_place_.GetDeviceType()) {
      if (src_place.GetDeviceId() == dst_place_.GetDeviceId()) {
        AsyncMemCpyD2D(nullptr, stream, dst_ptr, src_ptr, size);
        if (blocking) dev_ctx.Wait();
      } else {
        PADDLE_THROW(
            phi::errors::Unimplemented("TensorCopy is not supported."));
      }
    } else {
      PADDLE_THROW(phi::errors::Unimplemented("TensorCopy is not supported."));
    }
  } else if (src_place.GetType() == phi::AllocationType::CPU &&
             dst_place_.GetType() == phi::AllocationType::CPU) {
    std::memcpy(dst_ptr, src_ptr, size);
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

static inline int SizeFromAxis(const int axis, phi::DDim dims) {
  int size = 1;
  for (int i = axis; i < dims.size(); i++) {
    size *= dims[i];
  }
  return size;
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

inline void ExtractNCDWH(const phi::DDim& dims,
                         const phi::DataLayout& data_layout,
                         int* N,
                         int* C,
                         int* D,
                         int* H,
                         int* W) {
  *N = dims[0];

  if (dims.size() == 3) {
    *C = data_layout == phi::DataLayout::kNCHW ? dims[1] : dims[2];
    *D = 1;
    *H = 1;
    *W = data_layout == phi::DataLayout::kNCHW ? dims[2] : dims[1];
  } else if (dims.size() == 4) {
    *C = data_layout == phi::DataLayout::kNCHW ? dims[1] : dims[3];
    *D = 1;
    *H = data_layout == phi::DataLayout::kNCHW ? dims[2] : dims[1];
    *W = data_layout == phi::DataLayout::kNCHW ? dims[3] : dims[2];
  } else {
    *C = data_layout == phi::DataLayout::kNCHW ? dims[1] : dims[4];
    *D = data_layout == phi::DataLayout::kNCHW ? dims[2] : dims[1];
    *H = data_layout == phi::DataLayout::kNCHW ? dims[3] : dims[2];
    *W = data_layout == phi::DataLayout::kNCHW ? dims[4] : dims[3];
  }
}

template <typename T>
inline std::vector<T> get_new_data_from_tensor(
    const phi::CustomContext& dev_ctx,
    const phi::DenseTensor* new_data_tensor) {
  std::vector<T> vec_new_data;
  auto place = new_data_tensor->place();
  phi::DenseTensor cpu_starts_tensor;
  if (place.GetType() == phi::AllocationType::CUSTOM) {
    // if tensor on CUSTOM place, do memcpy to host
    cpu_starts_tensor.Resize(new_data_tensor->dims());
    dev_ctx.template HostAlloc<T>(&cpu_starts_tensor);
    TensorCopy(
        dev_ctx, *new_data_tensor, true, &cpu_starts_tensor, phi::CPUPlace());
  } else {
    // if tensor on CPU place, return ptr
    cpu_starts_tensor = *new_data_tensor;
  }
  auto new_data_ptr = reinterpret_cast<T*>(cpu_starts_tensor.data<T>());
  vec_new_data =
      std::vector<T>(new_data_ptr, new_data_ptr + cpu_starts_tensor.numel());
  return vec_new_data;
}

template <typename T>
inline phi::DenseTensor ReshapeToMatrix(const phi::DenseTensor& src,
                                        T num_col_dims) {
  int rank = src.dims().size();
  PADDLE_ENFORCE_GE(
      rank,
      2,
      phi::errors::InvalidArgument(
          "'ReshapeToMatrix()' is only used for flatten high rank "
          "tensors to matrixs. The dimensions of phi::DenseTensor must be "
          "greater or equal than 2. "
          "But received dimensions of phi::DenseTensor is %d",
          rank));
  if (rank == 2) {
    return src;
  }
  phi::DenseTensor res;
  res = src;
  res.Resize(phi::flatten_to_2d(src.dims(), num_col_dims));
  return res;
}

template <typename T>
class MPTypeTrait {
 public:
  using Type = T;
};

template <>
class MPTypeTrait<phi::dtype::float16> {
 public:
  using Type = float;
};

template <>
class MPTypeTrait<phi::dtype::bfloat16> {
 public:
  using Type = float;
};

}  // namespace custom_kernel
