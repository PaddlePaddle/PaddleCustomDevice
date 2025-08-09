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
#ifndef BACKENDS_METAX_GPU_KERNELS_FUNCS_HANDLE_UTILS_H_
#define BACKENDS_METAX_GPU_KERNELS_FUNCS_HANDLE_UTILS_H_
#include <mutex>

#include "paddle/phi/backends/dynload/cusolver.h"

namespace phi {
namespace dynload {

inline bool HasCUSOLVER() {
  std::call_once(cusolver_dso_flag,
                 []() { cusolver_dso_handle = GetCusolverDsoHandle(); });
  return cusolver_dso_handle != nullptr;
}

}  // namespace dynload

AttributeMap dnn_attrs_;

static std::mutex dnn_attr_mutex_;

inline bool HasDnnAttr(const std::string& attr_name) {
  std::lock_guard<std::mutex> lock(dnn_attr_mutex_);
  return dnn_attrs_.count(attr_name) != 0UL;
}
inline const Attribute& GetDnnAttr(const std::string& attr_name) {
  std::lock_guard<std::mutex> lock(dnn_attr_mutex_);
  auto iter = dnn_attrs_.find(attr_name);
  PADDLE_ENFORCE_NE(iter,
                    dnn_attrs_.end(),
                    common::errors::NotFound(
                        "Attribute `%s` is not found in OneDNNContext."));
  return iter->second;
}

inline void SetDnnAttr(const std::string& attr_name, Attribute attr) {
  std::lock_guard<std::mutex> lock(dnn_attr_mutex_);
  dnn_attrs_[attr_name] = attr;
}

inline void ClearDnnAttr() {
  std::lock_guard<std::mutex> lock(dnn_attr_mutex_);
  dnn_attrs_.clear();
}

class DnnWorkspaceHandle {
 public:
  inline DnnWorkspaceHandle(const Allocator* allocator, gpuStream_t stream)
      : allocator_(allocator), stream_(stream) {
    mtx_ = std::make_unique<std::mutex>();
  }

#if defined(PADDLE_WITH_CUSTOM_DEVICE) && defined(MX_DEVICE)
  // 沐曦专用移动构造函数
  DnnWorkspaceHandle(DnnWorkspaceHandle&& other) noexcept
      : allocation_(std::move(other.allocation_)),
        allocator_(other.allocator_),
        stream_(other.stream_),
        mtx_(std::move(other.mtx_)) {
    other.allocator_ = nullptr;
    other.stream_ = nullptr;
  }

  DnnWorkspaceHandle& operator=(DnnWorkspaceHandle&& other) noexcept {
    if (this != &other) {
      allocation_ = std::move(other.allocation_);
      allocator_ = other.allocator_;
      stream_ = other.stream_;
      mtx_ = std::move(other.mtx_);
      other.allocator_ = nullptr;
      other.stream_ = nullptr;
    }
    return *this;
  }
#endif

  inline void RunFunc(const std::function<void(void*)>& cudnn_func,
                      size_t required_workspace_bytes) {
    if (required_workspace_bytes > WorkspaceSize()) {
      ReallocWorkspace(required_workspace_bytes);
    }
    {
      std::lock_guard<std::mutex> guard(*mtx_);
      cudnn_func(allocation_ ? allocation_->ptr() : nullptr);
    }
  }

  /*! \brief Thread which call RunFuncSync() would release gpu memory after
   *  running the function. Currently this function is only used when cudnn
   *  exhaustive searching and callers have to guarantee that the input function
   *  is host blocking */
  void RunFuncSync(const std::function<void(void*)>& cudnn_func,
                   size_t required_workspace_bytes,
                   bool use_cached_allocation = true);

  inline size_t WorkspaceSize() {
    if (allocation_ == nullptr) {
      return 0;
    }
    return allocation_->size();
  }

  void ResetWorkspace();

  TEST_API void ReallocWorkspace(size_t required_workspace_bytes);

  // DnnWorkspaceHandle(const DnnWorkspaceHandle&) = delete;
  // DnnWorkspaceHandle& operator=(const DnnWorkspaceHandle&) = delete;

  DnnWorkspaceHandle(DnnWorkspaceHandle&&) = default;
  // DnnWorkspaceHandle& operator=(DnnWorkspaceHandle&&) = delete;

  DnnWorkspaceHandle& operator=(DnnWorkspaceHandle&&) = default;

 private:
  Allocator::AllocationPtr allocation_{nullptr};
  const Allocator* allocator_{nullptr};  // Not owned
  gpuStream_t stream_{nullptr};          // Not owned
  std::unique_ptr<std::mutex> mtx_;
};

inline static cusolverDnHandle_t cusolver_dn_handle_ = nullptr;
inline std::once_flag flag_cusolver_dn_;

inline void InitCusolverDnHandle(cusolverDnHandle_t* handle,
                                 gpuStream_t stream,
                                 Place place) {
  if (phi::dynload::HasCUSOLVER()) {
    // auto version = phi::dynload::cusolverDnGetVersion();
    PADDLE_RETRY_CUDA_SUCCESS(phi::dynload::cusolverDnCreate(handle));
    PADDLE_RETRY_CUDA_SUCCESS(
        phi::dynload::cusolverDnSetStream(*handle, stream));
  } else {
    *handle = nullptr;
  }
}

inline cusolverDnHandle_t GetCusolverDnHandle(gpuStream_t stream,
                                              GPUPlace place) {
  std::call_once(flag_cusolver_dn_, [&]() {
    if (!cusolver_dn_handle_) {
      InitCusolverDnHandle(&cusolver_dn_handle_, stream, place);
    }
  });
  PADDLE_ENFORCE_NOT_NULL(
      cusolver_dn_handle_,
      common::errors::InvalidArgument(
          "cusolverDn handle is null. Check device initialization."));
  return cusolver_dn_handle_;
}

inline static cudnnHandle_t dnn_handle_ = nullptr;

inline std::once_flag flag_dnn_;

inline void InitDnnHandle(cudnnHandle_t* handle,
                          gpuStream_t stream,
                          Place place) {
  if (phi::dynload::HasCUDNN()) {
    auto version = phi::dynload::cudnnGetVersion();
    auto local_cudnn_major =
        (version < 9000) ? version / 1000 : version / 10000;
    auto local_cudnn_minor =
        (version < 9000) ? (version % 1000) / 100 : (version % 10000) / 100;
    if (version < static_cast<size_t>(CUDNN_VERSION)) {
      std::cout << "ERROR." << std::endl;
    }
    PADDLE_RETRY_CUDA_SUCCESS(phi::dynload::cudnnCreate(handle));
    PADDLE_RETRY_CUDA_SUCCESS(phi::dynload::cudnnSetStream(*handle, stream));
  } else {
    *handle = nullptr;
  }
}

inline cudnnHandle_t GetDnnHandle(gpuStream_t stream, GPUPlace place) {
  std::call_once(flag_dnn_, [&]() {
    if (!dnn_handle_) {
      InitDnnHandle(&dnn_handle_, stream, place);
    }
  });
  PADDLE_ENFORCE_NOT_NULL(
      dnn_handle_,
      common::errors::InvalidArgument(
          "The GPU dnn handle is nullptr. It must not be null."));
  return dnn_handle_;
}

inline static DnnWorkspaceHandle* dnn_workspace_handle_ = nullptr;
inline std::once_flag flag_dnn_workspace_;

inline void InitDnnWorkspaceHandle(DnnWorkspaceHandle** handle,
                                   const phi::Allocator* allocator,
                                   gpuStream_t stream) {
  if (allocator != nullptr) {
    *handle = new DnnWorkspaceHandle(allocator, stream);
  } else {
    *handle = nullptr;
  }
}

inline DnnWorkspaceHandle GetDnnWorkspaceHandle(gpuStream_t stream,
                                                const phi::Allocator* allocator,
                                                Place place) {
  std::call_once(flag_dnn_workspace_, [&]() {
    if (!dnn_workspace_handle_) {
      InitDnnWorkspaceHandle(&dnn_workspace_handle_, allocator, stream);
    }
  });

  PADDLE_ENFORCE_NOT_NULL(
      dnn_workspace_handle_,
      common::errors::InvalidArgument("The GPU DNN workspace handle is null. "
                                      "Check allocator initialization."));
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  return std::move(*dnn_workspace_handle_);
#else
  return *dnn_workspace_handle_;
#endif
  // return *dnn_workspace_handle_;
}

}  // namespace phi
#endif  // BACKENDS_METAX_GPU_KERNELS_FUNCS_HANDLE_UTILS_H_
