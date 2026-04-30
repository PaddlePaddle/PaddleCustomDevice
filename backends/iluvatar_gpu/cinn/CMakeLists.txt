# =============================================================================
# CINN Plugin for Iluvatar (IXUCA) Backend
# =============================================================================

# 1. 查找 IXUCA 路径 为了在 runtime/cinn_runtime.cc 或 compiler.cc 中能 #include
#   <cuda_runtime.h> 我们需要把天数 SDK 的头文件路径加进来
set(IXUCA_PATH $ENV{IXUCA_PATH})
if(NOT IXUCA_PATH)
  set(IXUCA_PATH "/usr/local/corex") # 默认回退路径
  message(
    STATUS "[Iluvatar CINN] IXUCA_PATH not set, using default: ${IXUCA_PATH}")
else()
  message(STATUS "[Iluvatar CINN] Found IXUCA_PATH: ${IXUCA_PATH}")
endif()

# 1. 定义源文件列表 这里必须包含所有涉及到 CINN 实现的 .cc 文件
set(CINN_SRCS
    cinn_interface.cc # 总入口，负责 InitCinnInterface
    compiler/compiler.cc # 【关键】负责 IluvatarCompile 和 IluvatarGetRuntimeSource
    runtime/cinn_runtime.cc # 负责 IluvatarModuleLoad, IluvatarLaunchKernel
    passes/pass_manager.cc # 负责 IluvatarApplyCustomPass
)

# 1. 创建 OBJECT 库 使用 OBJECT 模式，只编译出 .o 文件，不生成 .a 或 .so 这样上一级的 CMake 可以直接抓取这些 .o
#   文件链接进最终的 plugin.so
add_library(iluvatar_cinn_obj OBJECT ${CINN_SRCS})

# 1. 配置头文件搜索路径
target_include_directories(
  iluvatar_cinn_obj
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} # 允许引用当前目录头文件 (cinn_interface.h)
          ${CMAKE_CURRENT_SOURCE_DIR}/../ # 允许引用上层头文件 (如 common/)
          ${IXUCA_PATH}/include # 【关键】允许引用 <maca_runtime.h>
          ${PADDLE_SOURCE_DIR} # 【新增】必须加这个！否则找不到 paddle/phi/...
          # Paddle 的头文件路径通常由外部环境 (Paddle_DIR) 自动包含
)

# 1. 编译选项设置 CINN 组件通常依赖 C++17 标准
set_property(TARGET iluvatar_cinn_obj PROPERTY CXX_STANDARD 17)

# 开启 PIC (Position Independent Code) 因为这些 .o 文件最终要被链接进动态库，必须开启此选项
set_property(TARGET iluvatar_cinn_obj PROPERTY POSITION_INDEPENDENT_CODE ON)

# 如果 compiler.cc 需要使用 filesystem 等库，可能需要链接 stdc++fs (视 GCC 版本而定) 但因为是 OBJECT
# 库，链接操作推迟到父级进行
