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

#include "iluvatar_context.h"  // NOLINT

#include <memory>
#include <mutex>

// namespace phi {
// bool AllowTF32Cudnn() { return false; }

// void DnnWorkspaceHandle::RunFuncSync(
//     const std::function<void(void*)>& cudnn_func,
//     size_t required_workspace_bytes,
//     bool use_cached_allocation) {
//   bool need_realloc = required_workspace_bytes > WorkspaceSize();
//   if (need_realloc && !use_cached_allocation) {
//     void* workspace_ptr = nullptr;
//     size_t size = ((required_workspace_bytes + 255) >> 8) << 8;
//     std::lock_guard<std::mutex> guard(*mtx_);
// #ifdef PADDLE_WITH_HIP
//     auto status = hipMalloc(&workspace_ptr, size);
// #else
//     auto status = cudaMalloc(&workspace_ptr, size);
// #endif
//     if (status == gpuSuccess) {
//       cudnn_func(workspace_ptr);
//       phi::backends::gpu::GpuStreamSync(stream_);
// #ifdef PADDLE_WITH_HIP
//       PADDLE_ENFORCE_GPU_SUCCESS(hipFree(workspace_ptr));
// #else
//       PADDLE_ENFORCE_GPU_SUCCESS(cudaFree(workspace_ptr));
// #endif
//       return;
//     }
//   }

//   RunFunc(cudnn_func, required_workspace_bytes);
//   if (need_realloc) {
//     // Release the workspace allocated in this running.
//     ResetWorkspace();
//   }
// }

// void DnnWorkspaceHandle::ResetWorkspace() { allocation_ = nullptr; }

// void DnnWorkspaceHandle::ReallocWorkspace(size_t required_workspace_bytes) {
//   if (required_workspace_bytes <= WorkspaceSize()) return;
//   // reset allocation first before re-allocate to save memory
//   allocation_.reset();
//   allocation_ = allocator_->Allocate(required_workspace_bytes);
// }
// }  // namespace phi

namespace iluvatar {
IluvatarContext::~IluvatarContext() {
  if (ixinfer_handle_) {
    cuinferDestroy(ixinfer_handle_);
  }
}
cuinferHandle_t IluvatarContext::getIxInferHandle() {
  if (!ixinfer_handle_) {
    cuinferCreate(&ixinfer_handle_);
  }
  return ixinfer_handle_;
}

IluvatarContext* getContextInstance() {
  static IluvatarContext context;
  return &context;
}
}  // namespace iluvatar
