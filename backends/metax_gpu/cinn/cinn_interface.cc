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

#include "cinn/cinn_interface.h"

#include <cstring>  // For memset
#include <iostream>

namespace paddle {
namespace custom_device {
namespace metax {

// ============================================================
// External Function Declarations
// These functions must be implemented in the corresponding subdirectory files
// (.cc).
// ============================================================

// --- From compiler/compiler.cc ---
// Invokes the mxcc toolchain to compile CINN-generated source code into a
// binary
extern C_Status MetaxCompile(void* dev_ptr,
                             const char* code,
                             char* out_path,
                             size_t len);

// Provides the MetaX GPU device runtime source code
extern const char* MetaxGetRuntimeSource(void* dev_ptr);

// --- From runtime/cinn_runtime.cc ---
// Loads a compiled binary module (.mx / .so)
extern C_Status MetaxModuleLoad(void* dev_ptr,
                                const char* path,
                                void** mod_out);

// Unloads a module
extern C_Status MetaxModuleUnload(void* dev_ptr, void* module_handle);

// Retrieves the kernel function address from a loaded module
extern C_Status MetaxGetKernelAddress(void* dev_ptr,
                                      void* module_handle,
                                      const char* func_name,
                                      void** func_out);

// Launches a kernel function
extern C_Status MetaxLaunchKernel(void* dev_ptr,
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
                                  void* stream);

// --- From passes/pass_manager.cc ---
// Applies custom graph optimization passes
extern C_Status MetaxApplyCustomPass(void* dev_ptr, void* ir_module);

// ============================================================
// Interface Initialization
// ============================================================

// Static instance, valid throughout the plugin lifetime
static C_CinnInterface metax_cinn_impl;

void InitCinnInterface(C_DeviceInterface* device_interface) {
  // 1. Zero-initialize for safety
  std::memset(&metax_cinn_impl, 0, sizeof(C_CinnInterface));

  // 2. Set struct size (used for version validation)
  metax_cinn_impl.size = sizeof(C_CinnInterface);

  // 3. Set context pointer (optional)
  // Point to a global state struct if your implementation needs one; otherwise
  // nullptr
  metax_cinn_impl.dev_ptr = nullptr;

  // 4. Register Compiler Toolchain interface
  metax_cinn_impl.compile = MetaxCompile;
  metax_cinn_impl.get_runtime_source = MetaxGetRuntimeSource;

  // 5. Register Runtime Strategy interface
  metax_cinn_impl.module_load = MetaxModuleLoad;
  metax_cinn_impl.module_unload = MetaxModuleUnload;
  metax_cinn_impl.get_kernel_address = MetaxGetKernelAddress;
  metax_cinn_impl.launch_kernel = MetaxLaunchKernel;

  // 6. Register Compilation Strategy interface
  metax_cinn_impl.apply_custom_pass = MetaxApplyCustomPass;

  // 7. Attach the populated dispatch table to the Paddle device interface
  if (device_interface) {
    device_interface->cinn_interface = &metax_cinn_impl;
  } else {
    std::cerr << "[MetaX] Error: device_interface is null during CINN init."
              << std::endl;
  }
}

}  // namespace metax
}  // namespace custom_device
}  // namespace paddle
