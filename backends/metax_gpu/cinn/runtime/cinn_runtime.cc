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

#include "paddle/phi/backends/device_ext.h"
#include <cuda.h> 
#include <iostream>
#include <vector>
#include <string>

namespace paddle {
namespace custom_device {
namespace metax {

// 【实现1】加载模块：相当于 cudaModuleLoad
C_Status MetaxModuleLoad(void* dev_ptr, const char* path, void** mod_out) {
    CUmodule module;
    CUresult err = cuModuleLoad(&module, path);
    if (err != CUDA_SUCCESS) return C_Status::C_FAILED;
    
    *mod_out = (void*)module;
    return C_Status::C_SUCCESS;
}

// 【实现2】卸载模块
C_Status MetaxModuleUnload(void* dev_ptr, void* module_handle) {
    cuModuleUnload((CUmodule)module_handle);
    return C_Status::C_SUCCESS;
}

// 【实现3】获取函数地址：相当于 cudaModuleGetFunction
C_Status MetaxGetKernelAddress(void* dev_ptr, void* module_handle, const char* func_name, void** func_out) {
    CUfunction func;
    CUresult err = cuModuleGetFunction(&func, (CUmodule)module_handle, func_name);
    if (err != CUDA_SUCCESS) return C_Status::C_FAILED;
    
    *func_out = (void*)func;
    return C_Status::C_SUCCESS;
}

// 【实现4】启动核函数：相当于 cudaLaunchKernel
C_Status MetaxLaunchKernel(void* dev_ptr, void* func_ptr, void** args, int num_args,
                           int gx, int gy, int gz, 
                           int bx, int by, int bz, 
                           int shm, void* stream) {
    // 注意：args 这里通常是 void*[]，可能需要处理一下参数封装
    CUresult err = cuLaunchKernel((CUfunction)func_ptr,
                                  gx, gy, gz, 
                                  bx, by, bz,
                                  shm, 
                                  (CUstream)stream, 
                                  args, 
                                  nullptr);
    if (err != CUDA_SUCCESS) return C_Status::C_FAILED;
    return C_Status::C_SUCCESS;
}

} // namespace metax
} // namespace custom_device
} // namespace paddle