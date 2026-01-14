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

#include "cinn_interface.h"
#include <cstring> // For memset
#include <iostream>

namespace paddle {
namespace custom_device {
namespace metax {

// ============================================================
// 外部函数声明 (External Function Declarations)
// 这些函数需要在对应的子目录文件中实现 (.cc)
// ============================================================

// --- 来自 compiler/compiler.cc ---
// 负责调用 mxcc 将 CINN 生成的源代码编译为二进制
extern C_Status MetaxCompile(void* dev_ptr,
                             const char* code,
                             char* out_path,
                             size_t len);

// 负责提供沐曦 GPU 运行时的基础源码 (类似 cuda_device_runtime.cu)
extern const char* MetaxGetRuntimeSource(void* dev_ptr);


// --- 来自 runtime/cinn_runtime.cc ---
// 负责加载编译好的二进制模块 (.mx / .so)
extern C_Status MetaxModuleLoad(void* dev_ptr,
                                const char* path,
                                void** mod_out);

// 负责卸载模块
extern C_Status MetaxModuleUnload(void* dev_ptr,
                                  void* module_handle);

// 负责从模块中查找核函数地址
extern C_Status MetaxGetKernelAddress(void* dev_ptr,
                                      void* module_handle,
                                      const char* func_name,
                                      void** func_out);

// 负责启动核函数 (Launch Kernel)
extern C_Status MetaxLaunchKernel(void* dev_ptr,
                                  void* func_ptr,
                                  void** args,
                                  int num_args,
                                  int gx, int gy, int gz,
                                  int bx, int by, int bz,
                                  int shm,
                                  void* stream);


// --- 来自 passes/pass_manager.cc ---
// 负责应用自定义的图优化 Pass
extern C_Status MetaxApplyCustomPass(void* dev_ptr,
                                     void* ir_module);


// ============================================================
// 接口初始化实现 (Interface Initialization)
// ============================================================

// 静态实例，确保在插件生命周期内有效
static C_CinnInterface metax_cinn_impl;

void InitCinnInterface(C_DeviceInterface* device_interface) {
    // 1. 安全起见，先清零
    std::memset(&metax_cinn_impl, 0, sizeof(C_CinnInterface));

    // 2. 设置结构体大小 (用于版本校验)
    metax_cinn_impl.size = sizeof(C_CinnInterface);

    // 3. 设置上下文指针 (可选)
    // 如果你的实现需要全局状态，可以指向一个结构体；否则设为 nullptr
    metax_cinn_impl.dev_ptr = nullptr;

    // 4. 挂载 Compiler Toolchain 接口
    metax_cinn_impl.compile = MetaxCompile;
    metax_cinn_impl.get_runtime_source = MetaxGetRuntimeSource;

    // 5. 挂载 Runtime Strategy 接口
    metax_cinn_impl.module_load = MetaxModuleLoad;
    metax_cinn_impl.module_unload = MetaxModuleUnload;
    metax_cinn_impl.get_kernel_address = MetaxGetKernelAddress;
    metax_cinn_impl.launch_kernel = MetaxLaunchKernel;

    // 6. 挂载 Compile Strategy 接口
    metax_cinn_impl.apply_custom_pass = MetaxApplyCustomPass;

    // 7. 【关键】将填好的表挂载到 Paddle 主设备接口上
    if (device_interface) {
        device_interface->cinn_interface = &metax_cinn_impl;
        // VLOG(3) << "[MetaX] CINN Interface initialized successfully.";
    } else {
        std::cerr << "[MetaX] Error: device_interface is null during CINN init." << std::endl;
    }
}

} // namespace metax
} // namespace custom_device
} // namespace paddle