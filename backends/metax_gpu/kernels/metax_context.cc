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

#include "kernels/metax_context.h"

namespace phi {
void DnnWorkspaceHandle::RunFuncSync(
    const std::function<void(void*)>& cudnn_func,
    size_t required_workspace_bytes,
    bool use_cached_allocation) {
  bool need_realloc = required_workspace_bytes > WorkspaceSize();
  if (need_realloc && !use_cached_allocation) {
    void* workspace_ptr = nullptr;
    size_t size = ((required_workspace_bytes + 255) >> 8) << 8;
    std::lock_guard<std::mutex> guard(*mtx_);
#ifdef PADDLE_WITH_HIP
    auto status = hipMalloc(&workspace_ptr, size);
#else
    auto status = cudaMalloc(&workspace_ptr, size);
#endif
    if (status == gpuSuccess) {
      cudnn_func(workspace_ptr);
      phi::backends::gpu::GpuStreamSync(stream_);
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(hipFree(workspace_ptr));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(cudaFree(workspace_ptr));
#endif
      return;
    }
  }

  RunFunc(cudnn_func, required_workspace_bytes);
  if (need_realloc) {
    // Release the workspace allocated in this running.
    ResetWorkspace();
  }
}

void DnnWorkspaceHandle::ResetWorkspace() { allocation_ = nullptr; }

void DnnWorkspaceHandle::ReallocWorkspace(size_t required_workspace_bytes) {
  if (required_workspace_bytes <= WorkspaceSize()) return;
  // reset allocation first before re-allocate to save memory
  allocation_.reset();
  allocation_ = allocator_->Allocate(required_workspace_bytes);
}

static std::function<blasLtHandle_t()> blaslt_handle_creator_{nullptr};
static blasLtHandle_t blaslt_handle_{nullptr};
static std::once_flag flag_blaslt_;

static void InitBlasLtHandle(blasLtHandle_t* blaslt_handle) {
#if defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 11060
  mcblasLtCreate(blaslt_handle);
#elif defined(PADDLE_WITH_HIP)
  phi::dynload::hipblasLtCreate(blaslt_handle);
#endif
}

blasLtHandle_t GetBlasLtHandle() {
  std::call_once(flag_blaslt_, [&]() {
    if (!blaslt_handle_) {
      if (!blaslt_handle_creator_)
        InitBlasLtHandle(&blaslt_handle_);
      else
        blaslt_handle_ = blaslt_handle_creator_();
    }
  });
  PADDLE_ENFORCE_NOT_NULL(
      blaslt_handle_,
      common::errors::InvalidArgument(
          "The GPU blasLt handle is nullptr. It must not be null."));
  return blaslt_handle_;
}
}  // namespace phi
