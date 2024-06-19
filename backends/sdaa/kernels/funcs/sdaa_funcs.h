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

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "paddle/phi/extension.h"
#include "runtime/runtime.h"

namespace custom_kernel {
using Context = phi::CustomContext;
/**
 * CPU -> SDAA
 * SDAA -> CPU
 * SDAA -> SDAA
 */
template <typename Context>
inline void TensorCopy(const Context& dev_ctx,
                       const phi::DenseTensor& src,
                       bool blocking,
                       phi::DenseTensor* dst,
                       const phi::Place& dst_place = phi::CustomPlace()) {
  auto* src_ptr = src.data();
  if (src_ptr == nullptr) {
    return;
  }
  const auto& src_place = src.place();
  auto dst_place_ = dst_place;
  if (dst_place_.GetType() != phi::AllocationType::CPU) {
    dst_place_ = dev_ctx.GetPlace();
  }

  if (&src == dst) {
    if (src_place == dst_place_) {
      VLOG(6) << "Skip copy the same data(" << src_ptr << ") from " << src_place
              << " to " << dst_place_;
    } else {
      VLOG(6) << "Src and dst are the same Tensor, in-place copy data("
              << src_ptr << ") from " << src_place << " to " << dst_place_;
      const phi::DenseTensor src_copy = src;
      TensorCopy(dev_ctx, src_copy, blocking, dst, dst_place_);
    }
    return;
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

  PADDLE_ENFORCE_EQ(
      dst->place(),
      dst_place_,
      phi::errors::Unavailable(
          "The Dst Tensor's place and dst_place do not match, Tensor's place "
          "place is %s, dst_place is %s.",
          dst->place(),
          dst_place_));

  if (src_ptr == dst_ptr && src_place == dst_place_) {
    VLOG(3) << "Skip copy the same data async from " << src_ptr << " in "
            << src_place << " to " << dst_ptr << " in " << dst_place_;
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
    AsyncMemCpyH2D(nullptr, stream, dst_ptr, src_ptr, size);
    if (blocking) {
      dev_ctx.Wait();
    }
  } else if (src_place.GetType() == phi::AllocationType::CUSTOM &&
             dst_place_.GetType() == phi::AllocationType::CPU) {
    AsyncMemCpyD2H(nullptr, stream, dst_ptr, src_ptr, size);
    if (blocking) {
      dev_ctx.Wait();
    }
  } else if (src_place.GetType() == phi::AllocationType::CUSTOM &&
             dst_place_.GetType() == phi::AllocationType::CUSTOM) {
    if (src_place.GetDeviceType() == dst_place_.GetDeviceType()) {
      if (src_place.GetDeviceId() == dst_place_.GetDeviceId()) {
        AsyncMemCpyD2D(nullptr, stream, dst_ptr, src_ptr, size);
        if (blocking) {
          dev_ctx.Wait();
        }
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
 * CPU -> SDAA
 */
template <typename T>
void TensorFromArray(const phi::CustomContext& ctx,
                     const T* src,
                     const size_t& array_size,
                     const phi::CustomContext& dev_ctx,
                     phi::DenseTensor* dst) {
  VLOG(4) << "TensorFromArray start";
  auto dst_place = dev_ctx.GetPlace();
  auto src_ptr = static_cast<const void*>(src);
  dst->Resize({static_cast<int64_t>(array_size)});
  auto dst_ptr = static_cast<void*>(dev_ctx.template Alloc<T>(dst));
  auto size = array_size * sizeof(T);
  if (UNLIKELY(size == 0)) return;

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
 * CPU -> SDAA
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

/**
 * SDAA -> CPU
 */
template <typename T>
inline void TensorToVector(const phi::CustomContext& ctx,
                           const phi::DenseTensor& src,
                           const phi::CustomContext& dev_ctx,
                           std::vector<T>* dst) {
  VLOG(4) << "MemCpyD2H start";

  auto src_ptr = static_cast<const void*>(src.data<T>());
  VLOG(4) << "MemCpyD2H start1";

  auto size = src.numel() * sizeof(T);

  dst->resize(src.numel());
  auto dst_ptr = static_cast<void*>(dst->data());

  auto src_place = src.place();

  if (src_place.GetType() == phi::AllocationType::CUSTOM) {
    AsyncMemCpyD2H(nullptr,
                   static_cast<C_Stream>(dev_ctx.stream()),
                   dst_ptr,
                   src_ptr,
                   size);
    ctx.Wait();
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
    AsyncMemCpyD2H(nullptr,
                   static_cast<C_Stream>(dev_ctx.stream()),
                   dst_ptr,
                   src_ptr,
                   size);
    ctx.Wait();
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "TensorToVector on %s is not supported.", src_place));
  }
  for (unsigned int i = 0; i < src.numel(); i++) {
    (*dst)[i] = static_cast<bool>(array[i]);
  }
  delete[] array;
}
/**
 * vector<DenseTensor*> -> DenseTensor*
 * only used for the number of tensor is one
 */
template <typename T>
inline void TensorFromVectorTensor(
    const phi::CustomContext& dev_ctx,
    const std::vector<const phi::DenseTensor*>& src,
    phi::DenseTensor* dst) {
  int n = src.size();
  dst->Resize({static_cast<int64_t>(n)});
  dev_ctx.template Alloc<T>(dst);
  auto src_place = src[0]->place();
  bool flag = false;
  if (src_place.GetType() == phi::AllocationType::CUSTOM) {
    flag = true;
  }
  for (int i = 0; i < n; ++i) {
    int m = src[i]->numel();
    PADDLE_ENFORCE_EQ(
        m,
        1,
        phi::errors::Unavailable(
            "The Src Tensor's element number must be 1, but accepted %d. ", m));
    if (flag) {
      AsyncMemCpyD2D(nullptr,
                     static_cast<C_Stream>(dev_ctx.stream()),
                     reinterpret_cast<void*>(dst->data<T>() + i),
                     src[i]->data(),
                     sizeof(T));
    } else {
      AsyncMemCpyH2D(nullptr,
                     static_cast<C_Stream>(dev_ctx.stream()),
                     reinterpret_cast<void*>(dst->data<T>() + i),
                     src[i]->data(),
                     sizeof(T));
    }
  }
}

inline static tecodnnHandle_t GetHandleFromCTX(const Context& dev_ctx) {
  return GetHandle(dev_ctx.stream());
}
inline static sdaaStream_t GetStreamFromCTX(const Context& dev_ctx) {
  return GetStream(dev_ctx.stream());
}

static std::string get_blas_data_type_str(tecoblasDataType_t t) {
  switch (t) {
    case TECOBLAS_DATA_FLOAT:
      return "\"TECOBLAS_DATA_FLOAT\"";
    case TECOBLAS_DATA_HALF:
      return "\"TECOBLAS_DATA_HALF\"";
    case TECOBLAS_DATA_DOUBLE:
      return "\"TECOBLAS_DATA_DOUBLE\"";
    case TECOBLAS_DATA_INT8:
      return "\"TECOBLAS_DATA_INT8\"";
    case TECOBLAS_DATA_INT16:
      return "\"TECOBLAS_DATA_INT16\"";
    case TECOBLAS_DATA_INT32:
      return "\"TECOBLAS_DATA_INT32\"";
    case TECOBLAS_DATA_INT64:
      return "\"TECOBLAS_DATA_INT64\"";
    default:
      return "\"UNKNOWN_DATA_TYPE\"";
  }
}

static std::string get_blas_trans_type_str(tecoblasOperation_t t) {
  switch (t) {
    case TECOBLAS_OP_N:
      return "\"TECOBLAS_OP_N\"";
    case TECOBLAS_OP_T:
      return "\"TECOBLAS_OP_T\"";
    case TECOBLAS_OP_C:
      return "\"TECOBLAS_OP_C\"";
    default:
      return "\"UNKNOWN_TRANS_TYPE\"";
  }
}

struct MatmulParam {
  tecoblasOperation_t _transa = TECOBLAS_OP_N;
  tecoblasOperation_t _transb = TECOBLAS_OP_N;
  int _m = 0;
  int _n = 0;
  int _k = 0;
  float _alpha = 0.0;
  tecoblasDataType_t _Atype = TECOBLAS_DATA_FLOAT;
  int _lda = 0;
  int64_t _strideA = 1;
  tecoblasDataType_t _Btype = TECOBLAS_DATA_FLOAT;
  int _ldb = 0;
  int64_t _strideB = 1;
  float _beta = 0.0;
  tecoblasDataType_t _Ctype = TECOBLAS_DATA_FLOAT;
  int _ldc = 0;
  int64_t _strideC = 1;
  int _batchCount = 1;
  tecoblasAPIName_t _apiName;

  std::string get_formated_str() {
    std::stringstream ss;
    ss << "{"
       << "\"transa\": " << get_blas_trans_type_str(_transa)
       << ", \"transb\": " << get_blas_trans_type_str(_transb)
       << ", \"m\": " << _m << ", \"n\": " << _n << ", \"k\": " << _k
       << ", \"alpha\": " << _alpha
       << ", \"Atype\": " << get_blas_data_type_str(_Atype)
       << ", \"lda\": " << _lda << ", \"strideA\": " << _strideA
       << ", \"Btype\": " << get_blas_data_type_str(_Btype)
       << ", \"ldb\": " << _ldb << ", \"strideB\": " << _strideB
       << ", \"beta\": " << _beta
       << ", \"Ctype\": " << get_blas_data_type_str(_Ctype)
       << ", \"ldc\": " << _ldc << ", \"strideC\": " << _strideC
       << ", \"batchCount\": " << _batchCount << "}";

    return ss.str();
  }
};

static void setTBlasWorkspace(const Context& dev_ctx,
                              const struct MatmulParam& param,
                              phi::DenseTensor* workspace) {
  CustomSDAAStream_t stream =
      reinterpret_cast<CustomSDAAStream_t>(dev_ctx.stream());
  tblasHandle_t tblas_handle = stream->tblasHandle;
  static size_t kWorkspaceSize;
  TBLAS_CHECK(tecoblasGetWorkspaceSize(tblas_handle,
                                       param._transa,
                                       param._transb,
                                       param._m,
                                       param._n,
                                       param._k,
                                       param._alpha,
                                       param._Atype,
                                       param._lda,
                                       param._strideA,
                                       param._Btype,
                                       param._ldb,
                                       param._strideB,
                                       param._beta,
                                       param._Ctype,
                                       param._ldc,
                                       param._strideC,
                                       param._batchCount,
                                       param._apiName,
                                       &kWorkspaceSize));

  if (kWorkspaceSize) {
    VLOG(4) << "start to allocate memory for tblas's workspace with size: "
            << kWorkspaceSize / 1024 << " KB.";
    phi::DenseTensorMeta w_meta = {phi::DataType::UINT8,
                                   {static_cast<int64_t>(kWorkspaceSize)}};
    workspace->set_meta(w_meta);
    dev_ctx.template Alloc<uint8_t>(workspace);

    TBLAS_CHECK(
        tecoblasSetWorkspace(tblas_handle, workspace->data(), kWorkspaceSize));
  }
}

inline static tblasHandle_t GetBlasHandleFromCTX(
    const Context& dev_ctx,
    const struct MatmulParam& param,
    phi::DenseTensor* workspace) {
  CustomSDAAStream_t stream =
      reinterpret_cast<CustomSDAAStream_t>(dev_ctx.stream());
  tblasHandle_t& tblas_handle = stream->tblasHandle;
  setTBlasWorkspace(dev_ctx, param, workspace);
  return tblas_handle;
}
template <typename T>
inline void broadcastDims(const phi::DDim& x_dims,
                          const phi::DDim& y_dims,
                          const int& axis,
                          std::vector<T>* x_dims_vec,
                          std::vector<T>* y_dims_vec) {
  auto max_dim = std::max(x_dims.size(), y_dims.size());
  auto min_dim = std::min(x_dims.size(), y_dims.size());
  auto axs = (axis < 0 ? max_dim - min_dim + 1 + axis : axis);
  PADDLE_ENFORCE_LE(
      min_dim + axs,
      max_dim,
      phi::errors::InvalidArgument(
          "Broadcast dimension mismatch. Operands could "
          "not be broadcast together with the shape of X = [%s] and "
          "the shape of Y = [%s] and axis of %d. Min dimension of %d add "
          "absolute value of axis "
          "of %d"
          "is larger than max dimension of %d",
          x_dims,
          y_dims,
          axis,
          min_dim,
          axs,
          max_dim));
  x_dims_vec->clear();
  y_dims_vec->clear();
  x_dims_vec->resize(max_dim, 1);
  y_dims_vec->resize(max_dim, 1);
  if (x_dims.size() > y_dims.size()) {
    std::copy(x_dims.Get(), x_dims.Get() + x_dims.size(), x_dims_vec->begin());
    std::copy(
        y_dims.Get(), y_dims.Get() + y_dims.size(), y_dims_vec->begin() + axs);
  } else {
    std::copy(
        x_dims.Get(), x_dims.Get() + x_dims.size(), x_dims_vec->begin() + axs);
    std::copy(y_dims.Get(), y_dims.Get() + y_dims.size(), y_dims_vec->begin());
  }
}

template <typename T>
inline std::vector<T> findReduceDims(const std::vector<T>& x_dims,
                                     const std::vector<T>& y_dims) {
  PADDLE_ENFORCE_EQ(x_dims.size(),
                    y_dims.size(),
                    phi::errors::InvalidArgument(
                        "findReduceDims requires the size of x_dims and the "
                        "size of y_dims to be equal"
                        "but Received x_dims size of %d and y_dims size of %d",
                        x_dims.size(),
                        y_dims.size()));
  std::vector<T> reduce_dims;
  for (size_t i = 0; i < x_dims.size(); i++) {
    if (x_dims[i] != y_dims[i]) {
      reduce_dims.push_back(i);
    }
  }
  return reduce_dims;
}

/**
 * Fold all the dimensions not to be reduced.
 * Squash the dims between every reduce_dims, and replace the dims with one dim
 * of the product of the dims.
 */
template <typename T>
inline void foldNonReduceDims(const std::vector<T>& x_dims,
                              const std::vector<T>& reduce_dims,
                              std::vector<T>* ref_dims,
                              std::vector<T>* ref_reduce_dims) {
  PADDLE_ENFORCE_NE(
      reduce_dims.size(),
      0,
      phi::errors::InvalidArgument("reduce_dims size should not be zero"));
  ref_dims->clear();
  ref_reduce_dims->clear();
  if (reduce_dims[0] != 0) {
    auto reduced = accumulate(x_dims.begin(),
                              x_dims.begin() + reduce_dims[0],
                              1,
                              std::multiplies<T>());
    ref_dims->push_back(reduced);
  }
  for (size_t i = 0; i < reduce_dims.size() - 1; i++) {
    ref_reduce_dims->push_back(ref_dims->size());
    ref_dims->push_back(x_dims[reduce_dims[i]]);
    if (reduce_dims[i] < reduce_dims[i + 1] - 1) {
      auto reduced = accumulate(x_dims.begin() + reduce_dims[i] + 1,
                                x_dims.begin() + reduce_dims[i + 1],
                                1,
                                std::multiplies<T>());
      ref_dims->push_back(reduced);
    }
  }
  // there's always at least 1 dim to reduce
  ref_reduce_dims->push_back(ref_dims->size());
  ref_dims->push_back(x_dims[reduce_dims.back()]);
  if (reduce_dims.back() < x_dims.size() - 1) {
    auto reduced = accumulate(x_dims.begin() + reduce_dims.back() + 1,
                              x_dims.end(),
                              1,
                              std::multiplies<T>());
    ref_dims->push_back(reduced);
  }
}
template <typename T>
inline phi::DenseTensor build_dummy_tensor(const Context& dev_ctx,
                                           phi::DataType dtype,
                                           phi::DDim input_dims) {
  phi::DenseTensor input_;
  input_.Resize(input_dims);
  dev_ctx.Alloc(&input_, dtype);
  return input_;
}
}  // namespace custom_kernel
