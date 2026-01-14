#include "paddle/phi/backends/device_ext.h"
#include <iostream>

namespace paddle {
namespace custom_device {
namespace metax {

// 负责应用自定义的图优化 Pass
// 目前阶段先留空，直接返回成功
C_Status MetaxApplyCustomPass(void* dev_ptr, void* ir_module) {
    // VLOG(3) << "[MetaX] MetaxApplyCustomPass called (No-op)";
    return C_Status::C_SUCCESS;
}

} // namespace metax
} // namespace custom_device
} // namespace paddle