# =============================================================================
# CINN Plugin for MetaX (MACA) Backend
# =============================================================================

# 1. Locate MACA SDK path To allow #include <maca_runtime.h> in
#   runtime/cinn_runtime.cc or compiler.cc, we need to add the MetaX SDK header
#   search path.
set(MACA_PATH $ENV{MACA_PATH})
if(NOT MACA_PATH)
  set(MACA_PATH "/opt/maca") # Default fallback path
  message(STATUS "[MetaX CINN] MACA_PATH not set, using default: ${MACA_PATH}")
else()
  message(STATUS "[MetaX CINN] Found MACA_PATH: ${MACA_PATH}")
endif()

# 1. Define source file list All .cc files involved in the CINN implementation
#   must be included here.
set(CINN_SRCS
    cinn_interface.cc # Main entry point, responsible for InitCinnInterface
    compiler/compiler.cc # Implements MetaxCompile and MetaxGetRuntimeSource
    runtime/cinn_runtime.cc # Implements MetaxModuleLoad, MetaxLaunchKernel
    passes/pass_manager.cc # Implements MetaxApplyCustomPass
)

# 1. Create OBJECT library Use OBJECT mode to compile into .o files only, without
#   generating .a or .so. This allows the parent CMake to directly collect these
#   .o files and link them into the final plugin.so.
add_library(metax_cinn_obj OBJECT ${CINN_SRCS})

# 1. Configure header search paths
target_include_directories(
  metax_cinn_obj
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} # Allow referencing headers in current
                                      # directory (cinn_interface.h)
          ${CMAKE_CURRENT_SOURCE_DIR}/../ # Allow referencing parent-level
                                          # headers (e.g., common/)
          ${MACA_PATH}/include # Allow referencing <maca_runtime.h>
          ${PADDLE_SOURCE_DIR} # Allow referencing paddle/phi/... headers
          # Paddle header paths are typically auto-included via the external
          # environment (Paddle_DIR)
)

# 1. Compiler options The CINN component typically requires C++17 standard
set_property(TARGET metax_cinn_obj PROPERTY CXX_STANDARD 17)

# Enable PIC (Position Independent Code) Required because these .o files will
# ultimately be linked into a shared library
set_property(TARGET metax_cinn_obj PROPERTY POSITION_INDEPENDENT_CODE ON)
