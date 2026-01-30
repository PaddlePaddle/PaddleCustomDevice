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

#include "kernels/custom_kernel/custom_context.h"

namespace phi {
// void DnnWorkspaceHandle::RunFuncSync(
//     const std::function<void(void*)>& cudnn_func,
//     size_t required_workspace_bytes,
//     bool use_cached_allocation) {
//   bool need_realloc = required_workspace_bytes > WorkspaceSize();
//   if (need_realloc && !use_cached_allocation) {
//     void* workspace_ptr = nullptr;
//     size_t size = ((required_workspace_bytes + 255) >> 8) << 8;
//     std::lock_guard<std::mutex> guard(*mtx_);
//     auto status = cudaMalloc(&workspace_ptr, size);
//     if (status == gpuSuccess) {
//       cudnn_func(workspace_ptr);
//       phi::backends::gpu::GpuStreamSync(stream_);
//       PADDLE_ENFORCE_GPU_SUCCESS(cudaFree(workspace_ptr));
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
}  // namespace phi
