// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include <cuda.h>

#include <iostream>
#include <string>
#include <vector>

#include "paddle/phi/backends/device_ext.h"

namespace paddle {
namespace custom_device {
namespace metax {

// Load module: equivalent to cuModuleLoad
C_Status MetaxModuleLoad(void* dev_ptr, const char* path, void** mod_out) {
  CUmodule module;
  CUresult err = cuModuleLoad(&module, path);
  if (err != CUDA_SUCCESS) return C_Status::C_FAILED;

  *mod_out = reinterpret_cast<void*>(module);
  return C_Status::C_SUCCESS;
}

// Unload module
C_Status MetaxModuleUnload(void* dev_ptr, void* module_handle) {
  cuModuleUnload((CUmodule)module_handle);
  return C_Status::C_SUCCESS;
}

// Get kernel function address: equivalent to cuModuleGetFunction
C_Status MetaxGetKernelAddress(void* dev_ptr,
                               void* module_handle,
                               const char* func_name,
                               void** func_out) {
  CUfunction func;
  CUresult err = cuModuleGetFunction(&func, (CUmodule)module_handle, func_name);
  if (err != CUDA_SUCCESS) return C_Status::C_FAILED;

  *func_out = reinterpret_cast<void*>(func);
  return C_Status::C_SUCCESS;
}

// Launch kernel: equivalent to cuLaunchKernel
C_Status MetaxLaunchKernel(void* dev_ptr,
                           void* func_ptr,
                           void** args,
                           int num_args,
                           int gx,
                           int gy,
                           int gz,
                           int bx,
                           int by,
                           int bz,
                           int shm,
                           void* stream) {
  // Note: args is typically a void*[] and may require argument marshaling
  CUresult err = cuLaunchKernel((CUfunction)func_ptr,
                                gx,
                                gy,
                                gz,
                                bx,
                                by,
                                bz,
                                shm,
                                (CUstream)stream,
                                args,
                                nullptr);
  if (err != CUDA_SUCCESS) return C_Status::C_FAILED;
  return C_Status::C_SUCCESS;
}

}  // namespace metax
}  // namespace custom_device
}  // namespace paddle
